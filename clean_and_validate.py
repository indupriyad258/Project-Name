"""
Data cleaning, validation, and audit utilities for inpatient registration data.

Use alongside generate_inpatient_data.py output for hands-on exercises.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

ICD10_PATTERN = re.compile(r"^[A-TV-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})?$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ZIP_PATTERN = re.compile(r"^\d{5}(-\d{4})?$")
VALID_GENDERS = {"M", "F", "Male", "Female", "male", "female", "NB"}
VALID_STAFF_PATTERN = re.compile(r"^REG-\d{4}$")


def load_raw(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=True)


def normalize_gender(value: str | float) -> str | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    mapping = {
        "m": "M", "male": "M", "f": "F", "female": "F", "nb": "NB",
        "M": "M", "F": "F", "NB": "NB",
    }
    return mapping.get(str(value).strip().lower(), str(value).strip())


def standardize_phone(value: str | float) -> str | None:
    if pd.isna(value):
        return None
    digits = re.sub(r"\D", "", str(value))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return None
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    cleaned = df.drop_duplicates()
    print(f"Removed {before - len(cleaned)} exact duplicate rows")
    return cleaned


def flag_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows that likely represent the same patient (different registration)."""
    work = df.copy()
    work["first_name_norm"] = work["first_name"].str.strip().str.upper()
    work["last_name_norm"] = work["last_name"].str.strip().str.upper()
    work["dob_norm"] = work["date_of_birth"].str.strip()
    work["near_duplicate_group"] = work.groupby(
        ["first_name_norm", "last_name_norm", "dob_norm"]
    ).ngroup()
    group_sizes = work["near_duplicate_group"].value_counts()
    suspicious_groups = group_sizes[group_sizes > 1].index
    work["is_near_duplicate"] = work["near_duplicate_group"].isin(suspicious_groups)
    print(f"Flagged {work['is_near_duplicate'].sum()} near-duplicate registrations")
    return work


def validate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean validation columns for audit reporting."""
    out = df.copy()
    today = date.today().isoformat()

    out["valid_email"] = out["email"].apply(
        lambda x: bool(EMAIL_PATTERN.match(str(x))) if pd.notna(x) else False
    )
    out["valid_zip"] = out["zip_code"].apply(
        lambda x: bool(ZIP_PATTERN.match(str(x))) if pd.notna(x) else False
    )
    out["valid_icd10"] = out["primary_diagnosis_icd10"].apply(
        lambda x: bool(ICD10_PATTERN.match(str(x))) if pd.notna(x) else False
    )
    out["valid_staff_id"] = out["registered_by"].apply(
        lambda x: bool(VALID_STAFF_PATTERN.match(str(x))) if pd.notna(x) else False
    )
    out["valid_gender"] = out["gender"].apply(
        lambda x: str(x).strip() in VALID_GENDERS if pd.notna(x) else False
    )

    out["admission_dt"] = pd.to_datetime(out["admission_date"], errors="coerce")
    out["discharge_dt"] = pd.to_datetime(out["discharge_date"], errors="coerce")
    out["dob_dt"] = pd.to_datetime(out["date_of_birth"], errors="coerce")

    out["valid_date_sequence"] = (
        out["admission_dt"].notna()
        & out["discharge_dt"].notna()
        & (out["discharge_dt"] >= out["admission_dt"])
    )
    out["valid_admission_not_future"] = out["admission_dt"].apply(
        lambda x: x.date().isoformat() <= today if pd.notna(x) else False
    )
    out["valid_los"] = pd.to_numeric(out["length_of_stay_days"], errors="coerce").between(1, 90)
    out["unique_registration_id"] = ~out["registration_id"].duplicated(keep=False)

    out["passes_validation"] = (
        out["valid_email"] | out["email"].isna()
    ) & out["valid_icd10"] & out["valid_date_sequence"] & out["valid_los"] & out["unique_registration_id"]

    return out


def audit_report(df: pd.DataFrame) -> dict:
    validated = validate_records(df)
    report = {
        "total_rows": len(validated),
        "exact_duplicates": int(validated.duplicated().sum()),
        "duplicate_registration_ids": int(validated["registration_id"].duplicated().sum()),
        "invalid_emails": int((~validated["valid_email"] & validated["email"].notna()).sum()),
        "invalid_icd10": int((~validated["valid_icd10"] & validated["primary_diagnosis_icd10"].notna()).sum()),
        "invalid_date_sequence": int((~validated["valid_date_sequence"]).sum()),
        "future_admissions": int((~validated["valid_admission_not_future"]).sum()),
        "invalid_length_of_stay": int((~validated["valid_los"]).sum()),
        "invalid_staff_ids": int((~validated["valid_staff_id"] & validated["registered_by"].notna()).sum()),
        "rows_passing_all_checks": int(validated["passes_validation"].sum()),
        "pass_rate_pct": round(validated["passes_validation"].mean() * 100, 2),
    }
    return report


def compare_eras(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize how patient flow and needs shifted pre- vs post-COVID."""
    clean_era = df[df["era"].isin(["pre_covid", "post_covid"])].copy()
    summary = clean_era.groupby("era").agg(
        registrations=("registration_id", "count"),
        unique_patients=("patient_id", "nunique"),
        avg_length_of_stay=("length_of_stay_days", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        pct_emergency=("admission_type", lambda s: (s == "Emergency").mean() * 100),
        pct_mental_health=("mental_health_admission", lambda s: (s == "Yes").mean() * 100),
        pct_covid_related=("covid_related", lambda s: (s == "Yes").mean() * 100),
        pct_telehealth_intake=("telehealth_intake_used", lambda s: (s == "Yes").mean() * 100),
    ).round(2)
    return summary


def clean_pipeline(input_path: str, output_path: str) -> None:
    df = load_raw(input_path)
    print(f"Loaded {len(df):,} rows from {input_path}")

    df = remove_exact_duplicates(df)
    df = flag_near_duplicates(df)

    df["gender_normalized"] = df["gender"].apply(normalize_gender)
    df["phone_standardized"] = df["phone"].apply(standardize_phone)
    df["first_name"] = df["first_name"].str.strip()
    df["last_name"] = df["last_name"].str.strip()

    report = audit_report(df)
    print("\n=== Audit Report ===")
    for key, value in report.items():
        print(f"  {key}: {value}")

    print("\n=== Era Comparison (pre vs post COVID) ===")
    print(compare_eras(df).to_string())

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nWrote cleaned/audited dataset to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean and audit inpatient registration data.")
    parser.add_argument(
        "-i", "--input",
        default="data/inpatient_registrations_raw.csv",
    )
    parser.add_argument(
        "-o", "--output",
        default="data/inpatient_registrations_audited.csv",
    )
    args = parser.parse_args()
    clean_pipeline(args.input, args.output)
