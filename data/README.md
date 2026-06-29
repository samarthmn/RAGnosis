# RAGnosis Dataset

A small, fully synthetic **relational clinic dataset** used as the knowledge base for
the RAGnosis pipelines. No real people — names, contacts, and records are all fabricated.

The data lives as CSV files in [`dataset/`](dataset/), one table per file, plus an
evaluation question bank in [`eval-questions.json`](eval-questions.json).

## Tables

| File | Rows | Primary key | Description |
| --- | --- | --- | --- |
| `patients.csv` | 1,200 | `patient_id` | Demographics + contact + insurance for each patient. |
| `doctors.csv` | 120 | `doctor_id` | Doctor profiles: specialization, department, fee, experience. |
| `departments.csv` | 12 | `department_id` | Hospital departments (name, floor, phone extension). |
| `appointments.csv` | 2,200 | `appointment_id` | Scheduled/visited appointments. |
| `medical_records.csv` | 700 | `record_id` | Diagnosis, treatment plan, and follow-up per visit. |
| `prescriptions.csv` | 400 | `prescription_id` | Medication, dosage, frequency, duration, refills. |
| `billing.csv` | 368 | `bill_id` | Consultation/lab/medicine fees, totals, payment status. |

### Columns

- **patients** — `patient_id, full_name, gender, date_of_birth, blood_type, phone, email, city, insurance_provider, emergency_contact_phone`
- **doctors** — `doctor_id, full_name, gender, specialization, department_id, phone, email, years_experience, consultation_fee_inr`
- **departments** — `department_id, department_name, floor, phone_extension`
- **appointments** — `appointment_id, patient_id, doctor_id, appointment_datetime, appointment_type, status, reason, room_number, created_at`
- **medical_records** — `record_id, appointment_id, patient_id, doctor_id, visit_date, diagnosis, treatment_plan, follow_up_required, follow_up_date`
- **prescriptions** — `prescription_id, record_id, patient_id, doctor_id, medication, dosage, frequency, duration_days, refills`
- **billing** — `bill_id, appointment_id, patient_id, bill_date, consultation_fee_inr, lab_fee_inr, medicine_fee_inr, total_amount_inr, payment_method, payment_status, insurance_claimed`

## Relationships

```
departments ─< doctors
patients ─< appointments >─ doctors
appointments ─< medical_records >─ patients, doctors
medical_records ─< prescriptions >─ patients, doctors
appointments ─< billing >─ patients
```

- `doctors.department_id` → `departments.department_id`
- `appointments.patient_id` → `patients.patient_id`, `appointments.doctor_id` → `doctors.doctor_id`
- `medical_records.appointment_id` → `appointments.appointment_id` (plus `patient_id`, `doctor_id`)
- `prescriptions.record_id` → `medical_records.record_id` (plus `patient_id`, `doctor_id`)
- `billing.appointment_id` → `appointments.appointment_id` (plus `patient_id`)

## How RAGnosis uses it

At chunking time (`app/common/chunking_v2.py`) these tables are flattened into
retrieval-friendly documents:

- one **patient** document per patient — demographics plus every visit (appointment +
  medical record + prescriptions + billing), with doctor and department names resolved;
- one reference document per **doctor** and per **department**.

These documents are then chunked, embedded, and stored in a Chroma vector DB. See the
[root README](../README.md) for the full pipeline.

`eval-questions.json` holds the graded question bank (easy/medium/hard, with keywords,
reference answers, and expected source tables) that `app/evaluator.py` scores against.

## Legacy explorer

`index.html`, `serve.sh`, and `metadata.json` are a leftover single-page explorer for a
previous **FHIR/NDJSON** dataset. They do **not** work with the CSV tables above and are
kept only for reference.
