"""Document building + chunking for the clinic dataset, shared by both pipelines.

The dataset is a relational clinic schema:

    patients, doctors, departments, appointments,
    medical_records, prescriptions, billing

:func:`build_documents` flattens those tables into retrieval-friendly documents:

* one **patient** document per patient, aggregating demographics + their
  appointments, medical records, prescriptions and billing (with doctor and
  department names resolved), and
* one reference document per **doctor** and per **department**.

:func:`chunk` then splits them and writes ``app/<pipeline>/chunks.jsonl``. Both
pipelines chunk the same way; the only difference is that ``advanced`` prefers
LLM-enriched documents (from ``app.advanced.preprocess``) when they exist.

Run it for a pipeline with::

    uv run python -m app.common.chunking basic
    uv run python -m app.common.chunking advanced
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from langchain_core.documents import Document

from app.common.chunks import create_chunks
from app.common.chunks import load_chunks as load_stored_chunks
from app.common.chunks import save_chunks, save_parents
from app.common.documents import clean_value, load_csv_tables
from app.common.models import parse_model_overrides, set_model_overrides
from app.common.paths import APP_ROOT, DATASET_DIR

PIPELINES = ("basic", "advanced")

# Patient demographic columns -> human label (skips any column not present).
PATIENT_FIELDS = [
    ("full_name", "Full name"),
    ("gender", "Gender"),
    ("date_of_birth", "Date of birth"),
    ("blood_type", "Blood type"),
    ("phone", "Phone"),
    ("email", "Email"),
    ("city", "City"),
    ("insurance_provider", "Insurance provider"),
    ("emergency_contact_phone", "Emergency contact phone"),
]


def _val(row: dict, column: str) -> str:
    return clean_value(row.get(column, ""))


def _records_by(table: pd.DataFrame, key: str) -> dict[str, list[dict]]:
    """Group a table's rows (as dicts) by a foreign-key column."""
    grouped: dict[str, list[dict]] = {}
    if key not in table.columns:
        return grouped
    for record in table.to_dict("records"):
        grouped.setdefault(clean_value(record.get(key, "")), []).append(record)
    return grouped


def _index_by(table: pd.DataFrame, key: str) -> dict[str, dict]:
    return {clean_value(r.get(key, "")): r for r in table.to_dict("records")}


def doctor_label(
    doctor_id: str, doctors: dict[str, dict], departments: dict[str, dict]
) -> str:
    """e.g. ``Dr. Sara Kulkarni (Laryngology, Cardiology dept)``."""
    doctor = doctors.get(doctor_id)
    if not doctor:
        return doctor_id or "unknown doctor"
    name = _val(doctor, "full_name") or doctor_id
    specialization = _val(doctor, "specialization")
    department = departments.get(_val(doctor, "department_id"), {})
    department_name = _val(department, "department_name")
    extra = ", ".join(
        part
        for part in (
            specialization,
            f"{department_name} dept" if department_name else "",
        )
        if part
    )
    return f"{name} ({extra})" if extra else name


def _demographics_lines(patient: dict) -> list[str]:
    lines = []
    for column, label in PATIENT_FIELDS:
        value = _val(patient, column)
        if value:
            lines.append(f"- {label}: {value}")
    return lines


def _prescription_phrase(row: dict) -> str:
    return ", ".join(
        bit
        for bit in (
            " ".join(p for p in (_val(row, "medication"), _val(row, "dosage")) if p),
            _val(row, "frequency"),
            (
                f"for {_val(row, 'duration_days')} days"
                if _val(row, "duration_days")
                else ""
            ),
            f"{_val(row, 'refills')} refills" if _val(row, "refills") else "",
        )
        if bit
    )


def _billing_phrase(row: dict) -> str:
    return (
        f"consultation INR {_val(row, 'consultation_fee_inr')}, "
        f"lab INR {_val(row, 'lab_fee_inr')}, "
        f"medicine INR {_val(row, 'medicine_fee_inr')}, "
        f"total INR {_val(row, 'total_amount_inr')} "
        f"({_val(row, 'payment_method')}, {_val(row, 'payment_status')}, "
        f"insurance claimed {_val(row, 'insurance_claimed')})"
    )


def _visit_block(
    appt: dict | None,
    record: dict | None,
    rx_rows: list[dict],
    bill: dict | None,
    doctors,
    departments,
) -> str:
    """One coherent encounter: the appointment plus its medical record, that
    record's prescriptions, and its bill. ``appt`` may be None for a stray
    medical record with no matching appointment."""
    source = appt or record or {}
    date = (
        _val(appt, "appointment_datetime") if appt else _val(record or {}, "visit_date")
    )
    lines = [
        f"### Visit on {date} with "
        f"{doctor_label(_val(source, 'doctor_id'), doctors, departments)}".rstrip()
    ]
    if appt:
        detail = ", ".join(
            bit
            for bit in (
                _val(appt, "appointment_type"),
                f"status {_val(appt, 'status')}" if _val(appt, "status") else "",
                f'reason "{_val(appt, "reason")}"' if _val(appt, "reason") else "",
                (
                    f"room {_val(appt, 'room_number')}"
                    if _val(appt, "room_number")
                    else ""
                ),
            )
            if bit
        )
        if detail:
            booked = _val(appt, "created_at")
            lines.append(detail + (f" (booked {booked})" if booked else ""))
    if record:
        follow_up = _val(record, "follow_up_required")
        follow_up_date = _val(record, "follow_up_date")
        if follow_up_date:
            follow_up = f"{follow_up} on {follow_up_date}".strip()
        lines.append(
            f"Diagnosis: {_val(record, 'diagnosis')}. "
            f"Treatment: {_val(record, 'treatment_plan')}. "
            f"Follow-up: {follow_up}."
        )
    if rx_rows:
        lines.append(
            "Prescribed: " + "; ".join(_prescription_phrase(rx) for rx in rx_rows) + "."
        )
    if bill:
        lines.append("Billing: " + _billing_phrase(bill) + ".")
    return "\n".join(lines)


def _section(title: str, lines: list[str]) -> str:
    if not lines:
        return ""
    return f"\n## {title} ({len(lines)})\n" + "\n".join(lines)


def _patient_header(patient: dict) -> str:
    """Compact identity line prepended to every visit child.

    The old whole-patient char-splitting lost the patient's name after the first
    chunk, so later visit chunks were unmatchable. Anchoring identity into each
    child fixes both the embedding and the keyword-substring retrieval metric.
    """
    patient_id = _val(patient, "patient_id")
    name = _val(patient, "full_name") or patient_id
    bits = [f"Patient: {name} ({patient_id})"]
    for column in ("gender", "date_of_birth", "city", "blood_type", "insurance_provider"):
        value = _val(patient, column)
        if value:
            bits.append(value)
    return " | ".join(bits)


def _patient_row_indices(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Pre-group every table once so per-patient assembly is O(1) lookups."""
    return {
        "doctors": _index_by(tables.get("doctors", pd.DataFrame()), "doctor_id"),
        "departments": _index_by(
            tables.get("departments", pd.DataFrame()), "department_id"
        ),
        "appointments": _records_by(
            tables.get("appointments", pd.DataFrame()), "patient_id"
        ),
        "records": _records_by(
            tables.get("medical_records", pd.DataFrame()), "patient_id"
        ),
        "prescriptions": _records_by(
            tables.get("prescriptions", pd.DataFrame()), "patient_id"
        ),
        "billing": _records_by(tables.get("billing", pd.DataFrame()), "patient_id"),
    }


def _patient_encounters(
    patient_id: str, idx: dict[str, Any]
) -> tuple[list[tuple[str, dict]], list[str]]:
    """One patient's encounters as ``(visit_text, visit_meta)`` pairs, plus any
    orphan prescription/bill lines.

    Shared by :func:`build_patient_documents` (which joins the blocks into one
    parent blob) and :func:`build_visit_documents` (one child Document each), so
    parent and child text stay perfectly consistent.
    """
    doctors, departments = idx["doctors"], idx["departments"]
    appt_rows = idx["appointments"].get(patient_id, [])
    record_rows = idx["records"].get(patient_id, [])
    rx_rows = idx["prescriptions"].get(patient_id, [])
    bill_rows = idx["billing"].get(patient_id, [])

    record_by_appt = {_val(r, "appointment_id"): r for r in record_rows}
    rx_by_record: dict[str, list[dict]] = {}
    for rx in rx_rows:
        rx_by_record.setdefault(_val(rx, "record_id"), []).append(rx)
    bill_by_appt = {_val(b, "appointment_id"): b for b in bill_rows}

    used_records: set[str] = set()
    used_rx: set[str] = set()
    used_bills: set[str] = set()

    blocks: list[tuple[str, dict]] = []
    for appt in sorted(appt_rows, key=lambda a: _val(a, "appointment_datetime")):
        appt_id = _val(appt, "appointment_id")
        record = record_by_appt.get(appt_id)
        rec_id = _val(record, "record_id") if record else ""
        rxs = rx_by_record.get(rec_id, []) if rec_id else []
        bill = bill_by_appt.get(appt_id)

        if record:
            used_records.add(rec_id)
        used_rx.update(_val(rx, "prescription_id") for rx in rxs)
        if bill:
            used_bills.add(_val(bill, "bill_id"))

        meta = {
            "visit_date": _val(appt, "appointment_datetime"),
            "appointment_id": appt_id,
            "record_id": rec_id or appt_id,
            "doctor_id": _val(appt, "doctor_id"),
        }
        blocks.append((_visit_block(appt, record, rxs, bill, doctors, departments), meta))

    # Safety net: medical records not reachable from any appointment.
    for record in record_rows:
        rec_id = _val(record, "record_id")
        if rec_id in used_records:
            continue
        used_records.add(rec_id)
        rxs = rx_by_record.get(rec_id, [])
        used_rx.update(_val(rx, "prescription_id") for rx in rxs)
        meta = {
            "visit_date": _val(record, "visit_date"),
            "appointment_id": "",
            "record_id": rec_id,
            "doctor_id": _val(record, "doctor_id"),
        }
        blocks.append((_visit_block(None, record, rxs, None, doctors, departments), meta))

    orphan_lines = [
        f"- Prescription: {_prescription_phrase(rx)}"
        for rx in rx_rows
        if _val(rx, "prescription_id") not in used_rx
    ] + [
        f"- Bill: {_billing_phrase(bill)}"
        for bill in bill_rows
        if _val(bill, "bill_id") not in used_bills
    ]
    return blocks, orphan_lines


def build_patient_documents(tables: dict[str, pd.DataFrame]) -> list[Document]:
    """One large blob per patient (all visits). Used as the retrieval *parent*."""
    idx = _patient_row_indices(tables)
    documents: list[Document] = []
    for patient in tables["patients"].to_dict("records"):
        patient_id = _val(patient, "patient_id")
        name = _val(patient, "full_name") or patient_id
        blocks, orphan_lines = _patient_encounters(patient_id, idx)
        visit_blocks = [text for text, _ in blocks]

        body = "\n".join(
            part
            for part in [
                f"# Patient Record: {name} ({patient_id})",
                "\n## Demographics\n" + "\n".join(_demographics_lines(patient)),
                _section("Visits", visit_blocks),
                _section("Unlinked records", orphan_lines),
            ]
            if part
        )
        row_count = sum(
            len(idx[key].get(patient_id, []))
            for key in ("appointments", "records", "prescriptions", "billing")
        )
        documents.append(
            Document(
                page_content=body,
                metadata={
                    "doc_type": "patient",
                    "patient_id": patient_id,
                    "patient_name": name,
                    "record_id": patient_id,
                    "source_csv_names": "patients.csv",
                    "visit_count": len(visit_blocks),
                    "row_count": row_count,
                },
            )
        )
    return documents


def build_visit_documents(tables: dict[str, pd.DataFrame]) -> list[Document]:
    """One document per encounter, identity-anchored and linked to its patient
    parent via ``parent_id``. This is the small, precise unit we embed."""
    idx = _patient_row_indices(tables)
    documents: list[Document] = []
    for patient in tables["patients"].to_dict("records"):
        patient_id = _val(patient, "patient_id")
        name = _val(patient, "full_name") or patient_id
        header = _patient_header(patient)
        blocks, orphan_lines = _patient_encounters(patient_id, idx)

        for position, (text, meta) in enumerate(blocks, start=1):
            documents.append(
                Document(
                    page_content=f"{header}\n\n{text}",
                    metadata={
                        "doc_type": "visit",
                        "patient_id": patient_id,
                        "patient_name": name,
                        "parent_id": patient_id,
                        "record_id": meta["record_id"] or f"{patient_id}#v{position}",
                        "visit_date": meta["visit_date"],
                        "doctor_id": meta["doctor_id"],
                        "source_csv_names": (
                            "appointments.csv,medical_records.csv,"
                            "prescriptions.csv,billing.csv"
                        ),
                        "row_count": 1,
                    },
                )
            )

        # Keep orphan rows retrievable as their own (identity-anchored) child.
        if orphan_lines:
            documents.append(
                Document(
                    page_content=(
                        f"{header}\n\n## Unlinked records\n" + "\n".join(orphan_lines)
                    ),
                    metadata={
                        "doc_type": "visit",
                        "patient_id": patient_id,
                        "patient_name": name,
                        "parent_id": patient_id,
                        "record_id": f"{patient_id}#unlinked",
                        "visit_date": "",
                        "doctor_id": "",
                        "source_csv_names": "prescriptions.csv,billing.csv",
                        "row_count": 1,
                    },
                )
            )
    return documents


def build_doctor_documents(tables: dict[str, pd.DataFrame]) -> list[Document]:
    if "doctors" not in tables:
        return []
    departments = _index_by(tables.get("departments", pd.DataFrame()), "department_id")
    documents: list[Document] = []
    for doctor in tables["doctors"].to_dict("records"):
        doctor_id = _val(doctor, "doctor_id")
        name = _val(doctor, "full_name") or doctor_id
        department = departments.get(_val(doctor, "department_id"), {})
        department_name = _val(department, "department_name") or _val(
            doctor, "department_id"
        )
        body = "\n".join(
            [
                f"# Doctor Profile: {name} ({doctor_id})",
                f"- Specialization: {_val(doctor, 'specialization')}",
                f"- Department: {department_name} ({_val(doctor, 'department_id')})",
                f"- Gender: {_val(doctor, 'gender')}",
                f"- Years of experience: {_val(doctor, 'years_experience')}",
                f"- Consultation fee (INR): {_val(doctor, 'consultation_fee_inr')}",
                f"- Phone: {_val(doctor, 'phone')}",
                f"- Email: {_val(doctor, 'email')}",
            ]
        )
        documents.append(
            Document(
                page_content=body,
                metadata={
                    "doc_type": "doctor",
                    "patient_id": "",
                    "patient_name": "",
                    "record_id": doctor_id,
                    "source_csv_names": "doctors.csv",
                    "row_count": 1,
                },
            )
        )
    return documents


def build_department_documents(tables: dict[str, pd.DataFrame]) -> list[Document]:
    if "departments" not in tables:
        return []
    doctors_by_dept = _records_by(
        tables.get("doctors", pd.DataFrame()), "department_id"
    )
    documents: list[Document] = []
    for department in tables["departments"].to_dict("records"):
        department_id = _val(department, "department_id")
        name = _val(department, "department_name") or department_id
        staff = doctors_by_dept.get(department_id, [])
        staff_lines = [
            f"- {_val(d, 'full_name')} ({_val(d, 'specialization')})"
            for d in sorted(staff, key=lambda d: _val(d, "full_name"))
        ]
        body = "\n".join(
            [
                f"# Department: {name} ({department_id})",
                f"- Floor: {_val(department, 'floor')}",
                f"- Phone extension: {_val(department, 'phone_extension')}",
                _section("Doctors", staff_lines).lstrip("\n"),
            ]
        )
        documents.append(
            Document(
                page_content=body,
                metadata={
                    "doc_type": "department",
                    "patient_id": "",
                    "patient_name": "",
                    "record_id": department_id,
                    "source_csv_names": "departments.csv",
                    "row_count": len(staff),
                },
            )
        )
    return documents


# --------------------------------------------------------------------------------------
# Rollup (aggregate) documents
#
# Questions like "how many patients are in Kolkata" or "which doctor has the largest
# appointment load" are answered by counts/totals that appear in NO single visit — they
# are computed over many rows. RAG can only retrieve text that exists, so we pre-compute
# those aggregates at build time and embed them as ordinary documents. Each rollup is its
# own parent, so retrieval returns it verbatim to the chat model.
# --------------------------------------------------------------------------------------


def _counts(values: list) -> list[tuple[str, int]]:
    """Non-empty value counts, ordered by frequency then label."""
    counter: dict[str, int] = {}
    for raw in values:
        value = clean_value(raw)
        if value:
            counter[value] = counter.get(value, 0) + 1
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))


def _status_phrase(status_counts: list[tuple[str, int]]) -> str:
    return ", ".join(f"{count} {status.lower()}" for status, count in status_counts)


def _rollup_doc(
    record_id: str,
    title: str,
    lines: list[str],
    source_csv_names: str,
    *,
    patient_id: str = "",
    patient_name: str = "",
) -> Document:
    body = "\n".join([f"# {title}", *lines])
    return Document(
        page_content=body,
        metadata={
            "doc_type": "rollup",
            "patient_id": patient_id,
            "patient_name": patient_name,
            "record_id": record_id,
            "parent_id": record_id,
            "source_csv_names": source_csv_names,
            "row_count": len(lines),
        },
    )


def _distribution_doc(
    record_id: str,
    title: str,
    table: pd.DataFrame,
    column: str,
    source_csv_names: str,
) -> Document | None:
    """A single 'count by <column>' rollup (e.g. patients per city)."""
    if table.empty or column not in table.columns:
        return None
    pairs = _counts(table[column].tolist())
    if not pairs:
        return None
    lines = [f"- {label}: {count}" for label, count in pairs]
    lines.append(f"- Total: {sum(count for _, count in pairs)}")
    return _rollup_doc(record_id, title, lines, source_csv_names)


def _to_float(value: object) -> float:
    try:
        return float(clean_value(value) or 0)
    except ValueError:
        return 0.0


def build_rollup_documents(tables: dict[str, pd.DataFrame]) -> list[Document]:
    """Pre-computed aggregate documents: global distributions, rankings, and short
    per-doctor / per-patient summaries. Each is a self-contained, self-parent doc."""
    empty = pd.DataFrame()
    patients = tables.get("patients", empty)
    appointments = tables.get("appointments", empty)
    billing = tables.get("billing", empty)
    doctors = tables.get("doctors", empty)
    doctor_index = _index_by(doctors, "doctor_id")
    departments = _index_by(tables.get("departments", empty), "department_id")

    docs: list[Document] = []

    # --- global "count by X" distributions ------------------------------------
    distributions = [
        ("rollup:patients_by_city", "Patient count by city", patients, "city", "patients.csv"),
        ("rollup:patients_by_insurance", "Patient count by insurance provider", patients, "insurance_provider", "patients.csv"),
        ("rollup:patients_by_blood_type", "Patient count by blood type", patients, "blood_type", "patients.csv"),
        ("rollup:patients_by_gender", "Patient count by gender", patients, "gender", "patients.csv"),
        ("rollup:appointments_by_status", "Appointment count by status", appointments, "status", "appointments.csv"),
        ("rollup:appointments_by_type", "Appointment count by type", appointments, "appointment_type", "appointments.csv"),
        ("rollup:bills_by_status", "Bill count by payment status", billing, "payment_status", "billing.csv"),
        ("rollup:bills_by_method", "Bill count by payment method", billing, "payment_method", "billing.csv"),
        ("rollup:doctors_by_specialization", "Doctor count by specialization", doctors, "specialization", "doctors.csv"),
    ]
    for record_id, title, table, column, source in distributions:
        doc = _distribution_doc(record_id, title, table, column, source)
        if doc is not None:
            docs.append(doc)

    # Doctor count by department, with department names resolved.
    if not doctors.empty and "department_id" in doctors.columns:
        pairs = _counts(doctors["department_id"].tolist())
        if pairs:
            lines = []
            for department_id, count in pairs:
                name = clean_value(departments.get(department_id, {}).get("department_name")) or department_id
                lines.append(f"- {name} ({department_id}): {count} doctors")
            lines.append(f"- Total: {sum(count for _, count in pairs)}")
            docs.append(
                _rollup_doc(
                    "rollup:doctors_by_department",
                    "Doctor count by department",
                    lines,
                    "doctors.csv,departments.csv",
                )
            )

    # --- per-doctor appointment load + busiest-doctors ranking ----------------
    doctor_loads: list[tuple[str, str, int, list[tuple[str, int]]]] = []
    for doctor_id, rows in _records_by(appointments, "doctor_id").items():
        if not doctor_id:
            continue
        name = clean_value(doctor_index.get(doctor_id, {}).get("full_name")) or doctor_id
        doctor_loads.append((doctor_id, name, len(rows), _counts([r.get("status", "") for r in rows])))
    doctor_loads.sort(key=lambda item: -item[2])
    for doctor_id, name, total, status_counts in doctor_loads:
        docs.append(
            _rollup_doc(
                f"rollup:doctor_load:{doctor_id}",
                f"Appointment load — {name}",
                [f"- {name} ({doctor_id}): {total} appointments — {_status_phrase(status_counts)}"],
                "appointments.csv,doctors.csv",
            )
        )
    if doctor_loads:
        lines = [
            f"- {name} ({doctor_id}): {total} appointments — {_status_phrase(status_counts)}"
            for doctor_id, name, total, status_counts in doctor_loads[:15]
        ]
        docs.append(
            _rollup_doc(
                "rollup:doctors_by_appointment_load",
                "Doctors ranked by appointment load (busiest first)",
                lines,
                "appointments.csv,doctors.csv",
            )
        )

    # --- per-patient summary (appointment counts + billing totals) ------------
    appts_by_patient = _records_by(appointments, "patient_id")
    bills_by_patient = _records_by(billing, "patient_id")
    patient_bill_totals: list[tuple[str, str, int, float]] = []
    for patient in patients.to_dict("records"):
        patient_id = clean_value(patient.get("patient_id"))
        if not patient_id:
            continue
        name = clean_value(patient.get("full_name")) or patient_id
        city = clean_value(patient.get("city"))
        appts = appts_by_patient.get(patient_id, [])
        bills = bills_by_patient.get(patient_id, [])
        lines: list[str] = []
        if appts:
            status_counts = _counts([r.get("status", "") for r in appts])
            lines.append(f"- Appointments: {len(appts)} total — {_status_phrase(status_counts)}")
        if bills:
            total_billed = sum(_to_float(b.get("total_amount_inr")) for b in bills)
            city_suffix = f" from {city}" if city else ""
            lines.append(f"- Bills: {len(bills)} totaling INR {total_billed:,.0f}")
            patient_bill_totals.append((patient_id, f"{name}{city_suffix}", len(bills), total_billed))
        if lines:
            docs.append(
                _rollup_doc(
                    f"rollup:patient_summary:{patient_id}",
                    f"Summary — {name} ({patient_id})",
                    lines,
                    "appointments.csv,billing.csv",
                    patient_id=patient_id,
                    patient_name=name,
                )
            )
    if patient_bill_totals:
        patient_bill_totals.sort(key=lambda item: -item[3])
        lines = [
            f"- {label}: {bill_count} bills totaling INR {total:,.0f}"
            for _, label, bill_count, total in patient_bill_totals[:15]
        ]
        docs.append(
            _rollup_doc(
                "rollup:patients_by_total_billed",
                "Patients ranked by total billed amount (highest first)",
                lines,
                "billing.csv,patients.csv",
            )
        )

    return docs


def _as_own_parent(documents: list[Document]) -> list[Document]:
    """Small, self-contained docs (doctor/department, enriched blobs) are their
    own parent: point ``parent_id`` at ``record_id`` if not already set."""
    for document in documents:
        document.metadata.setdefault(
            "parent_id", document.metadata.get("record_id", "")
        )
    return documents


def build_parent_documents(tables: dict[str, pd.DataFrame]) -> list[Document]:
    """The large, context-rich docs returned to the generator: one blob per
    patient plus one per doctor/department. Stored in ``parents.jsonl`` and
    looked up at query time — never embedded directly."""
    documents = (
        build_patient_documents(tables)
        + build_doctor_documents(tables)
        + build_department_documents(tables)
    )
    return _as_own_parent(documents)


def build_child_documents(tables: dict[str, pd.DataFrame]) -> list[Document]:
    """The small units we embed: per-visit patient children plus the (already
    small) doctor/department docs, which act as their own parents."""
    return (
        build_visit_documents(tables)
        + _as_own_parent(build_doctor_documents(tables))
        + _as_own_parent(build_department_documents(tables))
    )


def build_documents(csv_dir: Path = DATASET_DIR) -> list[Document]:
    """Parent documents for the dataset (kept for enrichment / back-compat)."""
    tables = load_csv_tables(csv_dir)
    if "patients" not in tables:
        raise KeyError("patients.csv is required to build documents")
    documents = build_parent_documents(tables)
    print(f"Built {len(documents)} documents from {Path(csv_dir).resolve()}")
    return documents


# --------------------------------------------------------------------------------------
# Chunking (shared by both pipelines)
# --------------------------------------------------------------------------------------


def chunks_path_for(pipeline: str) -> Path:
    """Where a pipeline's stored (child) chunks live: ``app/<pipeline>/chunks.jsonl``."""
    return APP_ROOT / pipeline / "chunks.jsonl"


def parents_path_for(pipeline: str) -> Path:
    """Where a pipeline's parent documents live: ``app/<pipeline>/parents.jsonl``."""
    return APP_ROOT / pipeline / "parents.jsonl"


def source_parent_documents(
    pipeline: str,
    tables: dict[str, pd.DataFrame],
    csv_dir: Path = DATASET_DIR,
) -> list[Document]:
    """Parent docs returned as context. ``advanced`` prefers LLM-enriched docs.

    Enriched docs carry through their original ``record_id``/``patient_id``
    metadata, so ``parent_id`` still lines up with the visit children.
    """
    if pipeline == "advanced":
        from app.advanced.preprocess import ENRICHED_DOCS_PATH, load_enriched_documents

        if ENRICHED_DOCS_PATH.exists():
            print(f"Using enriched parents from {ENRICHED_DOCS_PATH}")
            return _as_own_parent(load_enriched_documents())
        print("No enriched documents found; using raw parents")
    return build_parent_documents(tables)


def load_chunks(pipeline: str, path: Path | None = None) -> list[Document]:
    return load_stored_chunks(
        path or chunks_path_for(pipeline),
        command_hint=f"uv run python -m app.common.chunking {pipeline}",
    )


def chunk(
    pipeline: str,
    *,
    csv_dir: Path = DATASET_DIR,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    chunks_path: Path | None = None,
) -> list[Document]:
    """Embed small per-visit *children*; persist large *parents* for query-time
    expansion. Children keep ``parent_id`` through the splitter, so a child that
    overflows ``chunk_size`` still resolves back to its patient parent.
    """
    tables = load_csv_tables(csv_dir)
    if "patients" not in tables:
        raise KeyError("patients.csv is required to build documents")

    # Rollups are self-parent: embedded as children AND stored as parents so a
    # retrieved rollup is returned to the model verbatim (with its exact numbers).
    rollups = build_rollup_documents(tables)
    children = build_child_documents(tables) + rollups
    parents = source_parent_documents(pipeline, tables, csv_dir) + rollups

    chunks = create_chunks(
        children,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    save_chunks(chunks, chunks_path or chunks_path_for(pipeline))
    save_parents(parents, parents_path_for(pipeline))
    return chunks


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in PIPELINES:
        raise SystemExit(
            f"usage: python -m app.common.chunking <{'|'.join(PIPELINES)}> [key=value ...]"
        )
    pipeline, overrides = args[0], args[1:]
    # Chunking is deterministic (no LLM); overrides are accepted for a uniform CLI
    # but only affect embedding/chat models used by later stages.
    set_model_overrides(pipeline, parse_model_overrides(overrides))
    chunk(pipeline)
    print(f"{pipeline.title()} chunking complete")


if __name__ == "__main__":
    main()
