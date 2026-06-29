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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from langchain_core.documents import Document

from app.common.chunks import create_chunks
from app.common.chunks import load_chunks as load_stored_chunks
from app.common.chunks import save_chunks
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


def build_patient_documents(tables: dict[str, pd.DataFrame]) -> list[Document]:
    patients = tables["patients"]
    doctors = _index_by(tables.get("doctors", pd.DataFrame()), "doctor_id")
    departments = _index_by(tables.get("departments", pd.DataFrame()), "department_id")

    appointments = _records_by(tables.get("appointments", pd.DataFrame()), "patient_id")
    records = _records_by(tables.get("medical_records", pd.DataFrame()), "patient_id")
    prescriptions = _records_by(
        tables.get("prescriptions", pd.DataFrame()), "patient_id"
    )
    billing = _records_by(tables.get("billing", pd.DataFrame()), "patient_id")

    documents: list[Document] = []
    for patient in patients.to_dict("records"):
        patient_id = _val(patient, "patient_id")
        name = _val(patient, "full_name") or patient_id

        appt_rows = appointments.get(patient_id, [])
        record_rows = records.get(patient_id, [])
        rx_rows = prescriptions.get(patient_id, [])
        bill_rows = billing.get(patient_id, [])

        # Index this patient's rows so each appointment can pull in its own
        # medical record, that record's prescriptions, and its bill.
        record_by_appt = {_val(r, "appointment_id"): r for r in record_rows}
        rx_by_record: dict[str, list[dict]] = {}
        for rx in rx_rows:
            rx_by_record.setdefault(_val(rx, "record_id"), []).append(rx)
        bill_by_appt = {_val(b, "appointment_id"): b for b in bill_rows}

        used_records: set[str] = set()
        used_rx: set[str] = set()
        used_bills: set[str] = set()

        visit_blocks: list[str] = []
        for appt in sorted(appt_rows, key=lambda a: _val(a, "appointment_datetime")):
            record = record_by_appt.get(_val(appt, "appointment_id"))
            rec_id = _val(record, "record_id") if record else ""
            rxs = rx_by_record.get(rec_id, []) if rec_id else []
            bill = bill_by_appt.get(_val(appt, "appointment_id"))

            if record:
                used_records.add(rec_id)
            used_rx.update(_val(rx, "prescription_id") for rx in rxs)
            if bill:
                used_bills.add(_val(bill, "bill_id"))

            visit_blocks.append(
                _visit_block(appt, record, rxs, bill, doctors, departments)
            )

        # Safety net: surface anything not reachable from an appointment so no
        # row is silently dropped (the current dataset has no such orphans).
        for record in record_rows:
            rec_id = _val(record, "record_id")
            if rec_id in used_records:
                continue
            used_records.add(rec_id)
            rxs = rx_by_record.get(rec_id, [])
            used_rx.update(_val(rx, "prescription_id") for rx in rxs)
            visit_blocks.append(
                _visit_block(None, record, rxs, None, doctors, departments)
            )

        orphan_lines = [
            f"- Prescription: {_prescription_phrase(rx)}"
            for rx in rx_rows
            if _val(rx, "prescription_id") not in used_rx
        ] + [
            f"- Bill: {_billing_phrase(bill)}"
            for bill in bill_rows
            if _val(bill, "bill_id") not in used_bills
        ]

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
                    "row_count": len(appt_rows)
                    + len(record_rows)
                    + len(rx_rows)
                    + len(bill_rows),
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


def build_documents(csv_dir: Path = DATASET_DIR) -> list[Document]:
    tables = load_csv_tables(csv_dir)
    if "patients" not in tables:
        raise KeyError("patients.csv is required to build documents")
    documents = (
        build_patient_documents(tables)
        + build_doctor_documents(tables)
        + build_department_documents(tables)
    )
    print(f"Built {len(documents)} documents from {Path(csv_dir).resolve()}")
    return documents


# --------------------------------------------------------------------------------------
# Chunking (shared by both pipelines)
# --------------------------------------------------------------------------------------


def chunks_path_for(pipeline: str) -> Path:
    """Where a pipeline's stored chunks live: ``app/<pipeline>/chunks.jsonl``."""
    return APP_ROOT / pipeline / "chunks.jsonl"


def source_documents(pipeline: str, csv_dir: Path = DATASET_DIR) -> list[Document]:
    """Raw documents, except ``advanced`` prefers LLM-enriched docs when present."""
    if pipeline == "advanced":
        from app.advanced.preprocess import ENRICHED_DOCS_PATH, load_enriched_documents

        if ENRICHED_DOCS_PATH.exists():
            print(f"Using enriched documents from {ENRICHED_DOCS_PATH}")
            return load_enriched_documents()
        print("No enriched documents found; using raw documents")
    return build_documents(csv_dir)


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
    documents = source_documents(pipeline, csv_dir)
    chunks = create_chunks(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    save_chunks(chunks, chunks_path or chunks_path_for(pipeline))
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
