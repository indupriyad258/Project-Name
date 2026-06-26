# Hospital Inpatient Registration Mock Data

Python-generated mock dataset of hospital **inpatient registrations** spanning **pre-COVID (2018–2019)** and **post-COVID (2022–2024)** years. The data is intentionally imperfect so you can practice deduplication, auditing, cleaning, and validation.

## Quick start

```bash
pip install -r requirements.txt
python generate_inpatient_data.py
python clean_and_validate.py
```

This creates:

| File | Description |
|------|-------------|
| `data/inpatient_registrations_raw.csv` | Main dataset with injected quality issues |
| `data/inpatient_registrations_clean_reference.csv` | Small clean benchmark (100 rows) |
| `data/inpatient_registrations_audited.csv` | Output after running the cleaning pipeline |

## Dataset schema

| Column | Description |
|--------|-------------|
| `patient_id` | Hospital patient identifier (reused on readmissions) |
| `registration_id` | Unique per admission event (should be unique) |
| `registration_timestamp` | When registration was recorded |
| `registered_by` | Staff ID (format `REG-####`) |
| `first_name`, `last_name` | Patient name |
| `date_of_birth` | ISO date (some invalid formats injected) |
| `gender` | Mixed encodings: M/F, Male/Female, etc. |
| `street_address`, `city`, `state`, `zip_code` | Address fields |
| `phone`, `email` | Contact info (mixed formats) |
| `insurance_type` | Payer category |
| `admission_date`, `discharge_date` | Stay dates |
| `length_of_stay_days` | Length of stay |
| `primary_diagnosis_icd10` | ICD-10 code |
| `diagnosis_description` | Plain-language diagnosis |
| `department` | Admitting department |
| `admission_type` | Emergency / Elective / Transfer |
| `covid_related` | Yes/No |
| `mental_health_admission` | Yes/No |
| `telehealth_intake_used` | Yes/No (post-COVID only) |
| `registration_year` | Calendar year |
| `era` | `pre_covid` or `post_covid` |

## Pre- vs post-COVID trends (built into the generator)

| Metric | Pre-COVID (2018–2019) | Post-COVID (2022–2024) |
|--------|------------------------|-------------------------|
| Volume | Baseline (~400/year) | ~12% higher (backlog + new demand) |
| Emergency admissions | ~52% | ~58% |
| Elective admissions | ~38% | ~28% (delayed care) |
| Mental health share | Lower (F-codes ~3–5%) | Higher (anxiety, depression, PTSD, SUD) |
| COVID-related | N/A | Long COVID, post-COVID syndrome, RSV |
| Telehealth intake | No | ~22% of post-COVID registrations |
| Top conditions | Pneumonia, MI, appendicitis, UTI | Post-COVID syndrome, anxiety, depression, diabetes complications |

## Injected data quality issues (for practice)

The raw file includes problems you can detect and fix:

1. **Exact duplicate rows** — full copy-paste duplicates
2. **Near-duplicates** — same patient, different `registration_id`, name casing/whitespace drift
3. **Duplicate `registration_id`** — should be unique per admission
4. **Missing values** — email, phone, insurance, diagnosis, etc.
5. **Invalid formats** — emails, ZIP codes, DOB strings, phone numbers
6. **Logical errors** — discharge before admission, future admission dates
7. **Invalid ICD-10 codes** — truncated or fake codes
8. **Invalid staff IDs** — `admin`, empty, out-of-range
9. **Inconsistent `era` labels** — `POST-COVID`, `unknown`
10. **Whitespace** — leading/trailing spaces in names

## Example exercises

### 1. Remove duplicates

```python
import pandas as pd

df = pd.read_csv("data/inpatient_registrations_raw.csv")
print("Exact dupes:", df.duplicated().sum())
df_clean = df.drop_duplicates()
```

### 2. Find near-duplicates

```python
df["name_key"] = (
    df["first_name"].str.strip().str.upper() + "|"
    + df["last_name"].str.strip().str.upper() + "|"
    + df["date_of_birth"].astype(str)
)
near = df[df.duplicated("name_key", keep=False)].sort_values("name_key")
```

### 3. Run the audit pipeline

```bash
python clean_and_validate.py -i data/inpatient_registrations_raw.csv -o data/inpatient_registrations_audited.csv
```

### 4. Compare eras

```python
from clean_and_validate import compare_eras, load_raw

df = load_raw("data/inpatient_registrations_raw.csv")
print(compare_eras(df))
```

### 5. Custom validation rules

Extend `validate_records()` in `clean_and_validate.py` with your own hospital rules (e.g. Medicaid requires valid ZIP, pediatric RSV only under age 18).

## Generator options

```bash
python generate_inpatient_data.py --records-per-year 600 -o data/custom_raw.csv
```

## Notes

- Data is **synthetic** — no real patients.
- ICD-10 codes are illustrative, not clinical advice.
- COVID gap years (2020–2021) are omitted to keep the pre/post contrast clear; extend `generate_inpatient_data.py` if you want pandemic peak years included.
# Project-Name
