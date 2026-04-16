"""
preprocess_submit_datasets.py
==============================
Green-Ops CI/CD Framework — Pre-Submit & Post-Submit Dataset Preprocessor

Reads the pre_submit_dataset and post_submit_dataset CSVs and produces:
  • presubmit_clean.csv         — cleaned + feature-engineered pre-submit data
  • postsubmit_clean.csv        — cleaned + feature-engineered post-submit data
  • combined_submit.csv         — merged dataset with delta features
  • eda_report.txt              — quick exploratory data analysis summary

Schema detected from your screenshots:
  test_duration | build | test_name | test_result (PASSED / FAILED)

Usage:
    pip install pandas scikit-learn
    python preprocess_submit_datasets.py \
        --presubmit  pre_submit_dataset.csv \
        --postsubmit post_submit_dataset.csv
"""

import argparse
import hashlib
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("greenops.preprocess")

OUTPUT_DIR = Path("./greenops_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOADING & BASIC VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_COLUMNS = {"test_duration", "build", "test_name", "test_result"}


def load_dataset(path: str, label: str) -> pd.DataFrame:
    """
    Load a pre/post submit CSV with robust parsing.
    Handles large files efficiently using chunked dtype inference.
    Tries comma, pipe, and tab separators automatically.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")

    log.info("Loading %s from %s ...", label, path)

    for sep in [",", "|", "\t"]:
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                engine="python",
                on_bad_lines="skip",
                low_memory=False,
            )
            df.columns = df.columns.str.strip().str.lower()
            if EXPECTED_COLUMNS.issubset(set(df.columns)):
                log.info("  Loaded %d rows x %d cols (sep=%r)", len(df), len(df.columns), sep)
                return df
        except Exception:
            continue

    # Fallback: pandas auto-sniffer
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    log.info("  Loaded %d rows x %d cols (auto-detected separator)", len(df), len(df.columns))
    return df


def validate_schema(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Check and coerce schema to expected types."""
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        log.warning("%s is missing columns: %s", label, missing)

    # Coerce test_duration to float
    if "test_duration" in df.columns:
        df["test_duration"] = pd.to_numeric(df["test_duration"], errors="coerce")

    # Coerce build to string (contains SHA hashes)
    if "build" in df.columns:
        df["build"] = df["build"].astype(str).str.strip()

    # Normalize test_result: PASSED -> 1, FAILED -> 0
    if "test_result" in df.columns:
        df["test_result_raw"] = df["test_result"].astype(str).str.strip().str.upper()
        df["test_result"] = df["test_result_raw"].map({"PASSED": 1, "FAILED": 0})

    # Strip whitespace from test_name
    if "test_name" in df.columns:
        df["test_name"] = df["test_name"].astype(str).str.strip()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Full cleaning pipeline:
      - Drop exact duplicates
      - Handle missing values
      - Remove impossible duration values
      - Detect and flag outliers
    """
    original_len = len(df)

    # 1. Drop fully duplicate rows
    df = df.drop_duplicates()
    log.info("[%s] Dropped %d exact duplicates", label, original_len - len(df))

    # 2. Drop rows where all key fields are null
    key_cols = [c for c in ["test_duration", "build", "test_name", "test_result"]
                if c in df.columns]
    df = df.dropna(subset=key_cols, how="all")

    # 3. Clean test_duration
    if "test_duration" in df.columns:
        neg_mask = df["test_duration"] < 0
        if neg_mask.sum() > 0:
            log.warning("[%s] Found %d negative durations - setting to NaN",
                        label, neg_mask.sum())
            df.loc[neg_mask, "test_duration"] = np.nan

        # Group-aware imputation: fill with median per test_name
        if df["test_duration"].isna().sum() > 0:
            df["test_duration"] = df.groupby("test_name")["test_duration"].transform(
                lambda x: x.fillna(x.median())
            )
            # Any remaining NaN (test seen only once) -> global median
            global_median = df["test_duration"].median()
            df["test_duration"] = df["test_duration"].fillna(global_median)

    # 4. Flag duration outliers using IQR (useful feature for XGBoost)
    if "test_duration" in df.columns:
        Q1 = df["test_duration"].quantile(0.25)
        Q3 = df["test_duration"].quantile(0.75)
        IQR = Q3 - Q1
        df["is_duration_outlier"] = (
            (df["test_duration"] < Q1 - 3 * IQR) |
            (df["test_duration"] > Q3 + 3 * IQR)
        ).astype(int)

    # 5. Drop rows where test_result is still NaN (unparseable values)
    if "test_result" in df.columns:
        before = len(df)
        df = df.dropna(subset=["test_result"])
        dropped = before - len(df)
        if dropped > 0:
            log.warning("[%s] Dropped %d rows with unparseable test_result", label, dropped)
        df["test_result"] = df["test_result"].astype(int)

    log.info("[%s] Clean shape: %d rows x %d cols", label, len(df), len(df.columns))
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def parse_build_components(build_str: str) -> dict:
    """
    Your build column contains composite strings of SHA hashes separated by
    underscores or slashes. Extract structural components.
    Example: '4e560f4d05/8c21ea298f/a64d32f263/37dc2de87d/...'
    """
    parts = re.split(r"[/_]", build_str)
    return {
        "build_seg_0": parts[0] if len(parts) > 0 else "",
        "build_seg_1": parts[1] if len(parts) > 1 else "",
        "build_seg_2": parts[2] if len(parts) > 2 else "",
        "build_seg_3": parts[3] if len(parts) > 3 else "",
        "build_depth":  len(parts),
    }


def extract_test_name_features(test_name: str) -> dict:
    """
    Parse test_name into semantic components for NER-style feature extraction.
    Detects module, class, method, and role from the test name string.
    """
    is_unit        = int(bool(re.search(r"unit|Unit", test_name)))
    is_integration = int(bool(re.search(r"integrat|Integration|IT", test_name)))
    is_e2e         = int(bool(re.search(r"e2e|E2E|end.to.end", test_name)))
    is_perf        = int(bool(re.search(r"perf|Perf|benchmark|Benchmark|load|Load", test_name)))
    is_smoke       = int(bool(re.search(r"smoke|Smoke|sanity|Sanity", test_name)))

    parts      = [p for p in re.split(r"[./:#_\-]", test_name) if p]
    name_hash  = int(hashlib.md5(test_name.encode()).hexdigest()[:8], 16)

    return {
        "test_is_unit":         is_unit,
        "test_is_integration":  is_integration,
        "test_is_e2e":          is_e2e,
        "test_is_perf":         is_perf,
        "test_is_smoke":        is_smoke,
        "test_name_length":     len(test_name),
        "test_name_depth":      len(parts),
        "test_name_hash":       name_hash,
    }


def engineer_features(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Full feature engineering pipeline for pre/post submit datasets."""
    log.info("[%s] Engineering features ...", label)

    # ── Duration features ─────────────────────────────────────────────────────
    if "test_duration" in df.columns:
        df["duration_log"]    = np.log1p(df["test_duration"])
        df["duration_zscore"] = (
            (df["test_duration"] - df["test_duration"].mean()) /
            (df["test_duration"].std() + 1e-8)
        )

        # Per-test historical stats (flakiness and performance signals)
        grp = df.groupby("test_name")["test_duration"]
        df["test_duration_mean"]  = grp.transform("mean")
        df["test_duration_std"]   = grp.transform("std").fillna(0)
        # >1 = slower than usual, <1 = faster — key signal for flakiness
        df["test_duration_ratio"] = (
            df["test_duration"] / (df["test_duration_mean"] + 1e-8)
        )

    # ── Test result aggregation per test_name (historical failure rate) ───────
    if "test_result" in df.columns:
        grp_res = df.groupby("test_name")["test_result"]
        df["test_pass_rate"]      = grp_res.transform("mean")
        df["test_failure_rate"]   = 1 - df["test_pass_rate"]
        df["test_total_runs"]     = grp_res.transform("count")
        df["test_total_failures"] = grp_res.transform(lambda x: (x == 0).sum())

        # Flakiness: 0 = perfectly stable, 1 = maximally flaky (50/50 pass rate)
        df["test_flakiness_score"] = 1 - (2 * abs(df["test_pass_rate"] - 0.5))

    # ── Build-level aggregation ───────────────────────────────────────────────
    if "test_result" in df.columns and "build" in df.columns:
        grp_build = df.groupby("build")["test_result"]
        df["build_pass_rate"]     = grp_build.transform("mean")
        df["build_test_count"]    = grp_build.transform("count")
        df["build_failure_count"] = grp_build.transform(lambda x: (x == 0).sum())
        df["build_has_failure"]   = (df["build_failure_count"] > 0).astype(int)

    # ── Test name semantic features ───────────────────────────────────────────
    log.info("[%s] Parsing test name features ...", label)
    name_feats = df["test_name"].apply(extract_test_name_features).apply(pd.Series)
    df = pd.concat([df, name_feats], axis=1)

    # ── Build path features ───────────────────────────────────────────────────
    log.info("[%s] Parsing build path features ...", label)
    build_feats = df["build"].apply(parse_build_components).apply(pd.Series)
    df = pd.concat([df, build_feats], axis=1)

    # ── Label encode build SHA segments for ML ────────────────────────────────
    for col in ["build_seg_0", "build_seg_1", "build_seg_2", "build_seg_3"]:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

    # ── Normalize duration for model input ────────────────────────────────────
    if "test_duration" in df.columns:
        scaler = MinMaxScaler()
        df["duration_normalized"] = scaler.fit_transform(df[["test_duration"]])

    log.info("[%s] Feature engineering complete. Shape: %d x %d",
             label, len(df), len(df.columns))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — COMBINE PRE + POST SUBMIT
# ─────────────────────────────────────────────────────────────────────────────

def combine_datasets(pre_df: pd.DataFrame, post_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge pre-submit and post-submit on (build, test_name) to create delta
    features — the difference in pass rate and duration between pre and post.

    These delta features are critical for the XGBoost Pf (probability of
    failure) gatekeeper:
      - A test that passed pre-submit but failed post-submit = regression
      - regression_detected = 1 is the ground truth training label
    """
    log.info("Combining pre-submit and post-submit datasets ...")

    def aggregate(df, suffix):
        return (
            df.groupby(["build", "test_name"])
            .agg(
                duration_mean  = ("test_duration", "mean"),
                duration_std   = ("test_duration", "std"),
                pass_rate      = ("test_result",   "mean"),
                failure_count  = ("test_result",   lambda x: (x == 0).sum()),
                total_runs     = ("test_result",   "count"),
            )
            .reset_index()
            .rename(columns={
                "duration_mean":  f"duration_mean_{suffix}",
                "duration_std":   f"duration_std_{suffix}",
                "pass_rate":      f"pass_rate_{suffix}",
                "failure_count":  f"failure_count_{suffix}",
                "total_runs":     f"total_runs_{suffix}",
            })
        )

    pre_agg  = aggregate(pre_df,  "pre")
    post_agg = aggregate(post_df, "post")

    combined = pre_agg.merge(post_agg, on=["build", "test_name"], how="outer")

    # Delta features (pre -> post changes)
    combined["delta_pass_rate"] = (
        combined["pass_rate_post"] - combined["pass_rate_pre"]
    )   # negative = test degraded after submit

    combined["delta_duration"] = (
        combined["duration_mean_post"] - combined["duration_mean_pre"]
    )   # positive = test got slower after submit

    # Ground truth label for XGBoost Pf model
    combined["regression_detected"] = (
        (combined["pass_rate_pre"]  >= 0.8) &   # was passing pre-submit
        (combined["pass_rate_post"] <  0.5)      # now failing post-submit
    ).astype(int)

    combined["only_in_pre"]  = combined["pass_rate_post"].isna().astype(int)
    combined["only_in_post"] = combined["pass_rate_pre"].isna().astype(int)

    combined = combined.fillna({
        "delta_pass_rate":     0,
        "delta_duration":      0,
        "regression_detected": 0,
    })

    log.info("Combined dataset shape: %d rows x %d cols",
             len(combined), len(combined.columns))
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — EDA REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_eda_report(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    output_path: Path,
):
    """Write a plain-text EDA summary for quick inspection."""
    lines = []

    def section(title):
        lines.append("\n" + "=" * 60)
        lines.append(title)
        lines.append("=" * 60)

    section("PRE-SUBMIT DATASET")
    lines.append(f"Shape             : {pre_df.shape}")
    lines.append(f"Unique builds     : {pre_df['build'].nunique()}")
    lines.append(f"Unique test names : {pre_df['test_name'].nunique()}")
    lines.append(f"Pass rate         : {pre_df['test_result'].mean():.2%}")
    lines.append(f"Failure rate      : {1 - pre_df['test_result'].mean():.2%}")
    lines.append(f"Missing durations : {pre_df['test_duration'].isna().sum()}")
    lines.append(f"\nDuration stats (seconds):")
    lines.append(pre_df["test_duration"].describe().to_string())
    lines.append(f"\nTop 10 most frequently FAILING tests (pre-submit):")
    lines.append(
        pre_df[pre_df["test_result"] == 0]["test_name"]
        .value_counts().head(10).to_string()
    )

    section("POST-SUBMIT DATASET")
    lines.append(f"Shape             : {post_df.shape}")
    lines.append(f"Unique builds     : {post_df['build'].nunique()}")
    lines.append(f"Unique test names : {post_df['test_name'].nunique()}")
    lines.append(f"Pass rate         : {post_df['test_result'].mean():.2%}")
    lines.append(f"Failure rate      : {1 - post_df['test_result'].mean():.2%}")
    lines.append(f"Missing durations : {post_df['test_duration'].isna().sum()}")
    lines.append(f"\nDuration stats (seconds):")
    lines.append(post_df["test_duration"].describe().to_string())
    lines.append(f"\nTop 10 most frequently FAILING tests (post-submit):")
    lines.append(
        post_df[post_df["test_result"] == 0]["test_name"]
        .value_counts().head(10).to_string()
    )

    section("COMBINED / DELTA FEATURES")
    lines.append(f"Shape                           : {combined_df.shape}")
    lines.append(f"Regression events (pre pass -> post fail): "
                 f"{combined_df['regression_detected'].sum()}")
    lines.append(f"Only in pre-submit              : {combined_df['only_in_pre'].sum()}")
    lines.append(f"Only in post-submit             : {combined_df['only_in_post'].sum()}")
    if "delta_pass_rate" in combined_df.columns:
        lines.append(f"\nDelta pass rate stats:")
        lines.append(combined_df["delta_pass_rate"].describe().to_string())

    section("CLASS BALANCE (for XGBoost Pf model)")
    lines.append("Pre-submit:")
    lines.append(
        pre_df["test_result"].value_counts()
        .rename({1: "PASSED", 0: "FAILED"}).to_string()
    )
    lines.append("\nPost-submit:")
    lines.append(
        post_df["test_result"].value_counts()
        .rename({1: "PASSED", 0: "FAILED"}).to_string()
    )

    pre_vc  = pre_df["test_result"].value_counts(normalize=True)
    post_vc = post_df["test_result"].value_counts(normalize=True)
    lines.append(f"\nPre-submit  minority class : {pre_vc.min():.2%}")
    lines.append(f"Post-submit minority class : {post_vc.min():.2%}")
    lines.append("\nNote: If minority class < 10%, use scale_pos_weight in XGBoost")
    if len(pre_vc) > 1:
        lines.append(f"  Recommended scale_pos_weight (pre)  ~ {pre_vc.max()/pre_vc.min():.1f}")
    if len(post_vc) > 1:
        lines.append(f"  Recommended scale_pos_weight (post) ~ {post_vc.max()/post_vc.min():.1f}")

    section("FEATURE COLUMNS GENERATED")
    skip = {"build", "test_name", "test_result_raw"}
    for col in [c for c in pre_df.columns if c not in skip]:
        lines.append(f"  * {col}")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    log.info("EDA report saved -> %s", output_path)
    print(report)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess pre_submit and post_submit datasets for Green-Ops"
    )
    parser.add_argument(
        "--presubmit",
        default="pre_submit_dataset.csv",
        help="Path to pre_submit_dataset CSV",
    )
    parser.add_argument(
        "--postsubmit",
        default="post_submit_dataset.csv",
        help="Path to post_submit_dataset CSV",
    )
    parser.add_argument(
        "--outdir",
        default="./greenops_output",
        help="Output directory",
    )
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = Path(args.outdir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    pre_raw  = load_dataset(args.presubmit,  "pre_submit")
    post_raw = load_dataset(args.postsubmit, "post_submit")

    # Validate schema
    pre_raw  = validate_schema(pre_raw,  "pre_submit")
    post_raw = validate_schema(post_raw, "post_submit")

    # Clean
    pre_clean  = clean_dataset(pre_raw,  "pre_submit")
    post_clean = clean_dataset(post_raw, "post_submit")

    # Feature engineering
    pre_feat  = engineer_features(pre_clean,  "pre_submit")
    post_feat = engineer_features(post_clean, "post_submit")

    # Combine into delta-feature dataset
    combined = combine_datasets(pre_feat, post_feat)

    # Save outputs
    pre_out      = OUTPUT_DIR / "presubmit_clean.csv"
    post_out     = OUTPUT_DIR / "postsubmit_clean.csv"
    combined_out = OUTPUT_DIR / "combined_submit.csv"
    eda_out      = OUTPUT_DIR / "eda_report.txt"

    pre_feat.to_csv(pre_out,      index=False)
    post_feat.to_csv(post_out,    index=False)
    combined.to_csv(combined_out, index=False)

    log.info("Saved -> %s", pre_out)
    log.info("Saved -> %s", post_out)
    log.info("Saved -> %s", combined_out)

    generate_eda_report(pre_feat, post_feat, combined, eda_out)

    print("\n" + "=" * 60)
    print("ALL OUTPUTS WRITTEN TO:", OUTPUT_DIR.resolve())
    print("  presubmit_clean.csv   - cleaned + features (pre-submit)")
    print("  postsubmit_clean.csv  - cleaned + features (post-submit)")
    print("  combined_submit.csv   - delta features for XGBoost Pf model")
    print("  eda_report.txt        - exploratory data analysis summary")


if __name__ == "__main__":
    main()