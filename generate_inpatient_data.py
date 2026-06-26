"""
Generate mock hospital inpatient registration data (2018–2024).

Designed for data cleaning, deduplication, validation, and audit exercises.
Includes intentional quality issues: duplicates, missing values, invalid formats,
inconsistent encodings, and logical errors.
"""

from __future__ import annotations

import argparse
import random
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker("en_US")
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Diagnosis profiles: illness mix shifts pre- vs post-COVID
# ---------------------------------------------------------------------------

PRE_COVID_DIAGNOSES = [
    ("J18.9", "Pneumonia, unspecified organism", "Pulmonary", 0.08),
    ("I21.9", "Acute myocardial infarction, unspecified", "Cardiology", 0.07),
    ("K35.80", "Unspecified acute appendicitis", "General Surgery", 0.05),
    ("N39.0", "Urinary tract infection, site not specified", "Internal Medicine", 0.06),
    ("E11.65", "Type 2 diabetes with hyperglycemia", "Endocrinology", 0.05),
    ("I50.9", "Heart failure, unspecified", "Cardiology", 0.06),
    ("J44.1", "COPD with acute exacerbation", "Pulmonary", 0.05),
    ("S72.001A", "Fracture of unspecified part of neck of right femur", "Orthopedics", 0.04),
    ("K80.20", "Calculus of gallbladder without obstruction", "General Surgery", 0.04),
    ("G93.1", "Anoxic brain damage, not elsewhere classified", "Neurology", 0.02),
    ("F10.239", "Alcohol dependence with withdrawal, unspecified", "Psychiatry", 0.03),
    ("J06.9", "Acute upper respiratory infection, unspecified", "Internal Medicine", 0.05),
    ("I63.9", "Cerebral infarction, unspecified", "Neurology", 0.04),
    ("C78.00", "Secondary malignant neoplasm of lung", "Oncology", 0.03),
    ("O80", "Encounter for full-term uncomplicated delivery", "Obstetrics", 0.04),
]

POST_COVID_DIAGNOSES = [
    ("U09.9", "Post COVID-19 condition, unspecified", "Pulmonary", 0.09),
    ("F41.1", "Generalized anxiety disorder", "Psychiatry", 0.08),
    ("F32.1", "Major depressive disorder, single episode, moderate", "Psychiatry", 0.07),
    ("J12.82", "Pneumonia due to coronavirus disease 2019", "Pulmonary", 0.04),
    ("E11.69", "Type 2 diabetes with other specified complication", "Endocrinology", 0.07),
    ("I50.9", "Heart failure, unspecified", "Cardiology", 0.06),
    ("F10.20", "Alcohol dependence, uncomplicated", "Psychiatry", 0.05),
    ("F11.20", "Opioid dependence, uncomplicated", "Psychiatry", 0.04),
    ("J06.9", "Acute upper respiratory infection, unspecified", "Internal Medicine", 0.04),
    ("J18.9", "Pneumonia, unspecified organism", "Pulmonary", 0.05),
    ("I21.9", "Acute myocardial infarction, unspecified", "Cardiology", 0.05),
    ("R53.83", "Other fatigue (long COVID symptom cluster)", "Internal Medicine", 0.06),
    ("F43.10", "Post-traumatic stress disorder, unspecified", "Psychiatry", 0.04),
    ("J21.0", "Acute bronchiolitis due to RSV", "Pediatrics", 0.03),
    ("K35.80", "Unspecified acute appendicitis", "General Surgery", 0.03),
    ("N17.9", "Acute kidney failure, unspecified", "Nephrology", 0.04),
    ("Z86.16", "Personal history of COVID-19", "Internal Medicine", 0.03),
]

ADMISSION_TYPES_PRE = {"Emergency": 0.52, "Elective": 0.38, "Transfer": 0.10}
ADMISSION_TYPES_POST = {"Emergency": 0.58, "Elective": 0.28, "Transfer": 0.14}

INSURANCE_TYPES = [
    "Medicare",
    "Medicaid",
    "Private - PPO",
    "Private - HMO",
    "Self-Pay",
    "Workers Comp",
    "Tricare",
    "Uninsured",
]

STAFF_IDS = [f"REG-{i:04d}" for i in range(1, 45)]
INVALID_STAFF_IDS = ["REG-9999", "admin", "TEMP", "", "REG-0045"]

GENDER_ENCODINGS = ["M", "F", "Male", "Female", "male", "female", "M", "F", "NB", ""]

PHONE_FORMATS = [
    lambda p: p,
    lambda p: p.replace("-", ""),
    lambda p: f"({p[:3]}) {p[3:6]}-{p[6:]}",
    lambda p: f"+1-{p}",
    lambda p: p.replace("-", " "),
    lambda p: "000-000-0000",
    lambda p: "invalid",
]


def weighted_choice(items: list[tuple], weight_idx: int = -1):
    codes = [item[0] for item in items]
    weights = [item[weight_idx] for item in items]
    idx = random.choices(range(len(items)), weights=weights, k=1)[0]
    return items[idx]


def random_admission_date(year: int) -> date:
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    if year == date.today().year:
        end = min(end, date.today())
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def generate_patient_core() -> dict:
    dob = fake.date_of_birth(minimum_age=1, maximum_age=95)
    first = fake.first_name()
    last = fake.last_name()
    return {
        "first_name": first,
        "last_name": last,
        "date_of_birth": dob,
        "gender_raw": random.choice(GENDER_ENCODINGS),
        "street_address": fake.street_address(),
        "city": fake.city(),
        "state": fake.state_abbr(),
        "zip_code": fake.zipcode(),
        "phone_raw": fake.numerify("###-###-####"),
        "email_raw": fake.email(),
    }


def build_record(
    patient_id: str,
    registration_id: str,
    year: int,
    era: str,
    core: dict | None = None,
    force_diagnosis: tuple | None = None,
) -> dict:
    core = core or generate_patient_core()
    is_post = era == "post_covid"

    if force_diagnosis:
        icd, desc, dept, _ = force_diagnosis
    else:
        icd, desc, dept, _ = weighted_choice(
            POST_COVID_DIAGNOSES if is_post else PRE_COVID_DIAGNOSES
        )

    admission = random_admission_date(year)
    los = max(1, int(np.random.lognormal(mean=1.6 if is_post else 1.4, sigma=0.7)))
    discharge = admission + timedelta(days=los)

    admission_mix = ADMISSION_TYPES_POST if is_post else ADMISSION_TYPES_PRE
    admission_type = random.choices(
        list(admission_mix.keys()), weights=list(admission_mix.values()), k=1
    )[0]

    covid_flag = "Yes" if is_post and random.random() < 0.18 else "No"
    if icd.startswith("U09") or icd.startswith("J12.82") or icd == "Z86.16":
        covid_flag = "Yes"

    mental_health = dept == "Psychiatry" or icd.startswith("F")
    telehealth_intake = is_post and random.random() < 0.22

    insurance = random.choices(
        INSURANCE_TYPES,
        weights=[0.28, 0.18, 0.22, 0.12, 0.06, 0.03, 0.04, 0.07],
        k=1,
    )[0]
    if is_post and insurance == "Uninsured" and random.random() < 0.3:
        insurance = "Medicaid"

    reg_time = datetime.combine(admission, datetime.min.time()) + timedelta(
        hours=random.randint(0, 23), minutes=random.randint(0, 59)
    )

    phone = random.choice(PHONE_FORMATS)(core["phone_raw"])

    return {
        "patient_id": patient_id,
        "registration_id": registration_id,
        "registration_timestamp": reg_time.isoformat(sep=" "),
        "registered_by": random.choice(STAFF_IDS),
        "first_name": core["first_name"],
        "last_name": core["last_name"],
        "date_of_birth": core["date_of_birth"].isoformat(),
        "gender": core["gender_raw"],
        "street_address": core["street_address"],
        "city": core["city"],
        "state": core["state"],
        "zip_code": core["zip_code"],
        "phone": phone,
        "email": core["email_raw"],
        "insurance_type": insurance,
        "admission_date": admission.isoformat(),
        "discharge_date": discharge.isoformat(),
        "length_of_stay_days": los,
        "primary_diagnosis_icd10": icd,
        "diagnosis_description": desc,
        "department": dept,
        "admission_type": admission_type,
        "covid_related": covid_flag,
        "mental_health_admission": "Yes" if mental_health else "No",
        "telehealth_intake_used": "Yes" if telehealth_intake else "No",
        "registration_year": year,
        "era": era,
    }


def inject_quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Introduce realistic data quality problems for cleaning exercises."""
    df = df.copy()
    n = len(df)

    # --- Exact duplicates (copy full rows) ---
    dup_idx = df.sample(n=min(35, n // 50), random_state=1).index
    exact_dupes = df.loc[dup_idx].copy()
    df = pd.concat([df, exact_dupes], ignore_index=True)

    # --- Near-duplicates: same person, new registration_id, slight name drift ---
    near_idx = df.sample(n=min(45, n // 40), random_state=2).index
    near_dupes = df.loc[near_idx].copy()
    for i, row in near_dupes.iterrows():
        near_dupes.at[i, "registration_id"] = str(uuid.uuid4())[:12].upper()
        if random.random() < 0.5:
            near_dupes.at[i, "first_name"] = row["first_name"].upper()
        if random.random() < 0.3:
            near_dupes.at[i, "last_name"] = row["last_name"] + " "
        if random.random() < 0.4:
            near_dupes.at[i, "phone"] = row["phone"].replace("-", "") if "-" in str(row["phone"]) else row["phone"]
    df = pd.concat([df, near_dupes], ignore_index=True)

    # --- Missing values ---
    missing_cols = {
        "email": 0.08,
        "phone": 0.05,
        "insurance_type": 0.04,
        "street_address": 0.03,
        "primary_diagnosis_icd10": 0.02,
        "registered_by": 0.03,
        "gender": 0.04,
        "date_of_birth": 0.02,
    }
    for col, rate in missing_cols.items():
        mask = df.sample(frac=rate, random_state=hash(col) % 100).index
        df.loc[mask, col] = np.nan

    # --- Invalid / inconsistent formats ---
    bad_email_idx = df.sample(n=min(25, len(df) // 100), random_state=3).index
    df.loc[bad_email_idx, "email"] = random.choice(
        ["not-an-email", "patient@", "@hospital.com", "john..doe@gmail.com"]
    )

    bad_zip_idx = df.sample(n=min(30, len(df) // 80), random_state=4).index
    df.loc[bad_zip_idx, "zip_code"] = random.choice(["00000", "ABCDE", "1234", ""])

    bad_dob_idx = df.sample(n=min(20, len(df) // 120), random_state=5).index
    df.loc[bad_dob_idx, "date_of_birth"] = random.choice(
        ["2030-01-01", "invalid", "01/15/1980", "1800-00-00"]
    )

    # --- Logical errors: discharge before admission ---
    logic_idx = df.sample(n=min(18, len(df) // 150), random_state=6).index
    for idx in logic_idx:
        adm = pd.to_datetime(df.at[idx, "admission_date"], errors="coerce")
        if pd.notna(adm):
            df.at[idx, "discharge_date"] = (adm - timedelta(days=random.randint(1, 5))).date().isoformat()

    # --- Negative / zero length of stay mismatch ---
    los_idx = df.sample(n=min(15, len(df) // 160), random_state=7).index
    df.loc[los_idx, "length_of_stay_days"] = random.choice([0, -1, -3, 999])

    # --- Invalid ICD codes ---
    icd_idx = df.sample(n=min(22, len(df) // 130), random_state=8).index
    df.loc[icd_idx, "primary_diagnosis_icd10"] = random.choice(
        ["XYZ.99", "J18", "F41", "COVID", ""]
    )

    # --- Invalid staff IDs ---
    staff_idx = df.sample(n=min(20, len(df) // 140), random_state=9).index
    df.loc[staff_idx, "registered_by"] = [random.choice(INVALID_STAFF_IDS) for _ in staff_idx]

    # --- Future admission dates ---
    future_idx = df.sample(n=min(12, len(df) // 200), random_state=10).index
    for idx in future_idx:
        df.at[idx, "admission_date"] = (date.today() + timedelta(days=random.randint(30, 400))).isoformat()

    # --- Duplicate registration_id (should be unique) ---
    reg_dup_idx = df.sample(n=min(8, len(df) // 250), random_state=11).index
    existing_reg = df["registration_id"].iloc[0]
    df.loc[reg_dup_idx, "registration_id"] = existing_reg

    # --- Inconsistent era vs year ---
    era_idx = df.sample(n=min(10, len(df) // 220), random_state=12).index
    for idx in era_idx:
        df.at[idx, "era"] = random.choice(["pre_covid", "post_covid", "POST-COVID", "unknown"])

    # --- Whitespace in names ---
    ws_idx = df.sample(n=min(40, len(df) // 60), random_state=13).index
    for idx in ws_idx:
        df.at[idx, "first_name"] = f"  {df.at[idx, 'first_name']}  "
        if random.random() < 0.5:
            df.at[idx, "last_name"] = f"{df.at[idx, 'last_name']}\t"

    return df.sample(frac=1, random_state=99).reset_index(drop=True)


def generate_dataset(
    pre_covid_years: list[int] | None = None,
    post_covid_years: list[int] | None = None,
    records_per_year: int = 400,
) -> pd.DataFrame:
    pre_covid_years = pre_covid_years or [2018, 2019]
    post_covid_years = post_covid_years or [2022, 2023, 2024]

    records: list[dict] = []
    patient_counter = 100000

    for year in pre_covid_years:
        for _ in range(records_per_year):
            patient_id = f"PAT-{patient_counter}"
            patient_counter += 1
            records.append(
                build_record(
                    patient_id=patient_id,
                    registration_id=str(uuid.uuid4())[:12].upper(),
                    year=year,
                    era="pre_covid",
                )
            )

    for year in post_covid_years:
        # Slightly higher volume post-COVID (backlog + new conditions)
        volume = int(records_per_year * (1.12 if year >= 2022 else 1.0))
        for _ in range(volume):
            patient_id = f"PAT-{patient_counter}"
            patient_counter += 1
            records.append(
                build_record(
                    patient_id=patient_id,
                    registration_id=str(uuid.uuid4())[:12].upper(),
                    year=year,
                    era="post_covid",
                )
            )

    # Re-admissions: same patient_id, different registration
    readmission_sources = random.sample(records, k=min(80, len(records) // 20))
    for src in readmission_sources:
        new_year = random.choice(post_covid_years if src["era"] == "post_covid" else pre_covid_years)
        core = {
            "first_name": src["first_name"],
            "last_name": src["last_name"],
            "date_of_birth": datetime.strptime(src["date_of_birth"], "%Y-%m-%d").date()
            if src["date_of_birth"] and "-" in str(src["date_of_birth"])
            else fake.date_of_birth(minimum_age=18, maximum_age=90),
            "gender_raw": src["gender"],
            "street_address": src["street_address"],
            "city": src["city"],
            "state": src["state"],
            "zip_code": src["zip_code"],
            "phone_raw": fake.numerify("###-###-####"),
            "email_raw": src["email"] if pd.notna(src.get("email")) else fake.email(),
        }
        records.append(
            build_record(
                patient_id=src["patient_id"],
                registration_id=str(uuid.uuid4())[:12].upper(),
                year=new_year,
                era="post_covid" if new_year >= 2022 else "pre_covid",
                core=core,
            )
        )

    df = pd.DataFrame(records)
    df = inject_quality_issues(df)
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== Inpatient Registration Dataset Summary ===")
    print(f"Total rows (including duplicates/issues): {len(df):,}")
    print(f"Unique patient_id: {df['patient_id'].nunique():,}")
    print(f"Unique registration_id: {df['registration_id'].nunique():,}")
    print("\nRegistrations by year:")
    print(df.groupby("registration_year").size().to_string())
    print("\nTop diagnoses by era:")
    for era in ["pre_covid", "post_covid"]:
        subset = df[df["era"].str.lower().str.contains("post", na=False) if era == "post_covid"
                    else df["era"].str.lower().eq("pre_covid")]
        if era == "pre_covid":
            subset = df[df["era"] == "pre_covid"]
        top = subset["primary_diagnosis_icd10"].value_counts().head(5)
        print(f"\n  {era}:")
        print(top.to_string())
    print("\nMental health admissions by era:")
    for era in ["pre_covid", "post_covid"]:
        subset = df[df["era"] == era]
        rate = (subset["mental_health_admission"] == "Yes").mean() * 100
        print(f"  {era}: {rate:.1f}%")
    print("\nKnown data quality issue counts (for audit practice):")
    print(f"  Missing email: {df['email'].isna().sum()}")
    print(f"  Missing phone: {df['phone'].isna().sum()}")
    print(f"  Duplicate registration_id: {df['registration_id'].duplicated().sum()}")
    print(f"  Exact duplicate rows: {df.duplicated().sum()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock inpatient registration data.")
    parser.add_argument(
        "-o", "--output",
        default="data/inpatient_registrations_raw.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--records-per-year",
        type=int,
        default=400,
        help="Base registration count per calendar year",
    )
    parser.add_argument(
        "--clean-copy",
        default="data/inpatient_registrations_clean_reference.csv",
        help="Optional clean reference file (no injected issues)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(records_per_year=args.records_per_year)
    df.to_csv(output_path, index=False)
    print(f"Wrote raw dataset: {output_path} ({len(df):,} rows)")

    # Clean reference for validation benchmarking (small sample, no issues)
    clean_records = []
    pid = 900000
    for year in [2019, 2023]:
        era = "pre_covid" if year < 2020 else "post_covid"
        for _ in range(50):
            pid += 1
            clean_records.append(
                build_record(
                    patient_id=f"PAT-{pid}",
                    registration_id=str(uuid.uuid4())[:12].upper(),
                    year=year,
                    era=era,
                )
            )
    clean_df = pd.DataFrame(clean_records)
    clean_path = Path(args.clean_copy)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(clean_path, index=False)
    print(f"Wrote clean reference: {clean_path} ({len(clean_df):,} rows)")

    print_summary(df)


if __name__ == "__main__":
    main()
