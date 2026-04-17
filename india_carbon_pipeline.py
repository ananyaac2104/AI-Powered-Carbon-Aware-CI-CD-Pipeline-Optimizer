"""
india_carbon_pipeline.py
=========================
Green-Ops CI/CD Framework — India State-Level Carbon Intensity Pipeline

This script does 4 things in sequence:
  1. PROCESS  — Reads the Ember India yearly dataset (india_yearly_full_release_long_format.csv)
                and computes per-state carbon intensity (gCO2/kWh) from:
                  - Direct CO2 intensity values from the dataset (primary)
                  - Fuel-mix weighted calculation as cross-check
  2. EXPORT   — Writes india_state_carbon.csv with per-state intensity + fuel breakdown
  3. STORE    — Loads the CSV into SQLite (greenops.db) tables:
                  state_carbon_intensity   — one row per state per year
                  datacenter_carbon        — mapped intensity per DC region
  4. ESTIMATE — CarbonEstimator class reads from DB, classifies by datacenter
                provider (AWS / Azure / GCP), and computes module carbon cost

Datacenter mappings:
  AWS   → Mumbai (Maharashtra), Hyderabad (Telangana)
  Azure → Pune (Maharashtra), Mumbai (Maharashtra), Chennai (Tamil Nadu)
  GCP   → Mumbai (Maharashtra), Delhi (Delhi)

Data source:
  Ember - India Yearly Full Release (Long Format), 2024
  https://ember-energy.org/data/
  License: CC BY 4.0

Usage:
  pip install pandas sqlite3 scipy
  python india_carbon_pipeline.py --csv india_yearly_full_release_long_format.csv
  python india_carbon_pipeline.py --csv india_yearly_full_release_long_format.csv --provider azure --region Chennai
"""

import argparse
import ast as pyast
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("greenops.india_carbon")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DATA_SOURCE_CITATION = (
    "Ember - India Yearly Full Release (Long Format), 2024. "
    "https://ember-energy.org/data/ | License: CC BY 4.0"
)

# Emission factors per fuel type (gCO2/kWh) — IPCC AR6 lifecycle median values
# Used to cross-check the dataset's own CO2 intensity values
FUEL_EMISSION_FACTORS = {
    "Coal":             820,   # gCO2/kWh — IPCC AR6 lifecycle
    "Gas":              490,   # gCO2/kWh
    "Other Fossil":     650,   # gCO2/kWh — weighted average oil/other
    "Nuclear":           12,   # gCO2/kWh — lifecycle including construction
    "Hydro":              4,   # gCO2/kWh — lifecycle
    "Solar":             41,   # gCO2/kWh — utility PV lifecycle
    "Wind":              11,   # gCO2/kWh — onshore lifecycle
    "Bioenergy":        230,   # gCO2/kWh — combustion with land use
    "Other Renewables":  20,   # gCO2/kWh — geothermal/tidal avg
}

# ── Datacenter region → Indian state mappings ─────────────────────────────────
# City → state that hosts it
CITY_TO_STATE = {
    "Mumbai":    "Maharashtra",
    "Pune":      "Maharashtra",
    "Hyderabad": "Telangana",
    "Chennai":   "Tamil Nadu",
    "Delhi":     "Delhi",
}

# Provider → list of (city, state) pairs
DATACENTER_REGIONS = {
    "aws": [
        {"city": "Mumbai",    "state": "Maharashtra", "zone": "ap-south-1"},
        {"city": "Hyderabad", "state": "Telangana",   "zone": "ap-south-2"},
    ],
    "azure": [
        {"city": "Pune",      "state": "Maharashtra", "zone": "centralindia"},
        {"city": "Mumbai",    "state": "Maharashtra", "zone": "westindia"},
        {"city": "Chennai",   "state": "Tamil Nadu",  "zone": "southindia"},
    ],
    "gcp": [
        {"city": "Mumbai",    "state": "Maharashtra", "zone": "asia-south1"},
        {"city": "Delhi",     "state": "Delhi",       "zone": "asia-south2"},
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PROCESS EMBER DATASET
# ─────────────────────────────────────────────────────────────────────────────

def process_ember_dataset(csv_path: str, year: int = 2024) -> pd.DataFrame:
    """
    Read the Ember India long-format CSV and produce a clean per-state
    carbon intensity table for the requested year.

    Columns in output:
        state, state_code, year,
        co2_intensity_gco2_kwh,      ← direct from dataset (primary)
        co2_intensity_weighted,       ← fuel-mix weighted calc (cross-check)
        total_generation_gwh,
        coal_pct, gas_pct, hydro_pct, solar_pct, wind_pct,
        nuclear_pct, bioenergy_pct, other_fossil_pct, other_renewables_pct,
        fossil_pct, clean_pct,
        data_source
    """
    log.info("Loading Ember dataset from %s ...", csv_path)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Filter to the requested year, exclude India Total and Others aggregates
    df_year = df[
        (df["Year"] == year) &
        (~df["State"].isin(["India Total", "Others"]))
    ].copy()

    states = sorted(df_year["State"].unique())
    log.info("Processing %d states for year %d ...", len(states), year)

    records = []

    for state in states:
        s = df_year[df_year["State"] == state]
        state_code = s["State code"].iloc[0] if len(s) > 0 else ""

        # ── 1. Direct CO2 intensity from dataset ─────────────────────────────
        co2_direct_row = s[
            (s["Variable"] == "CO2 intensity") &
            (s["Unit"] == "gCO2/kWh")
        ]
        co2_direct = float(co2_direct_row["Value"].iloc[0]) \
            if len(co2_direct_row) > 0 else None

        # ── 2. Fuel mix percentages ───────────────────────────────────────────
        fuel_pct = {}
        fuel_rows = s[
            (s["Category"] == "Electricity generation") &
            (s["Subcategory"] == "Fuel") &
            (s["Unit"] == "%")
        ]
        for _, row in fuel_rows.iterrows():
            fuel_pct[row["Variable"]] = float(row["Value"] or 0)

        # ── 3. Weighted CO2 intensity from fuel mix ───────────────────────────
        weighted = 0.0
        for fuel, ef in FUEL_EMISSION_FACTORS.items():
            pct = fuel_pct.get(fuel, 0.0)
            weighted += (pct / 100.0) * ef

        # ── 4. Total generation ───────────────────────────────────────────────
        total_gen_row = s[
            (s["Category"] == "Electricity generation") &
            (s["Variable"] == "Total Generation") &
            (s["Unit"] == "GWh")
        ]
        total_gen = float(total_gen_row["Value"].iloc[0]) \
            if len(total_gen_row) > 0 else None

        # ── 5. Aggregate fuel shares ──────────────────────────────────────────
        agg_rows = s[
            (s["Category"] == "Electricity generation") &
            (s["Subcategory"] == "Aggregate fuel") &
            (s["Unit"] == "%")
        ]
        agg = {row["Variable"]: float(row["Value"] or 0)
               for _, row in agg_rows.iterrows()}

        # Use direct CO2 intensity as primary; weighted as fallback
        final_intensity = co2_direct if co2_direct is not None else round(weighted, 2)

        records.append({
            "state":                   state,
            "state_code":              state_code,
            "year":                    year,
            "co2_intensity_gco2_kwh":  round(final_intensity, 2),
            "co2_intensity_weighted":  round(weighted, 2),
            "total_generation_gwh":    round(total_gen, 2) if total_gen else None,
            "coal_pct":                round(fuel_pct.get("Coal", 0), 2),
            "gas_pct":                 round(fuel_pct.get("Gas", 0), 2),
            "hydro_pct":               round(fuel_pct.get("Hydro", 0), 2),
            "solar_pct":               round(fuel_pct.get("Solar", 0), 2),
            "wind_pct":                round(fuel_pct.get("Wind", 0), 2),
            "nuclear_pct":             round(fuel_pct.get("Nuclear", 0), 2),
            "bioenergy_pct":           round(fuel_pct.get("Bioenergy", 0), 2),
            "other_fossil_pct":        round(fuel_pct.get("Other Fossil", 0), 2),
            "other_renewables_pct":    round(fuel_pct.get("Other Renewables", 0), 2),
            "fossil_pct":              round(agg.get("Fossil", 0), 2),
            "clean_pct":               round(agg.get("Clean", 0), 2),
            "data_source":             DATA_SOURCE_CITATION,
        })

    result_df = pd.DataFrame(records).sort_values(
        "co2_intensity_gco2_kwh", ascending=False
    ).reset_index(drop=True)

    log.info("Processed %d state records.", len(result_df))
    return result_df


def build_datacenter_carbon_df(state_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a datacenter-level carbon intensity table by mapping
    each DC city to its host state's intensity.

    For providers with multiple regions, the default intensity
    is the AVERAGE across all their regions.
    """
    state_lookup = state_df.set_index("state")["co2_intensity_gco2_kwh"].to_dict()
    records = []

    for provider, regions in DATACENTER_REGIONS.items():
        intensities = []
        for r in regions:
            intensity = state_lookup.get(r["state"], None)
            if intensity is None:
                log.warning("No intensity found for %s (%s)", r["state"], provider)
                continue
            intensities.append(intensity)
            records.append({
                "provider":         provider.upper(),
                "city":             r["city"],
                "state":            r["state"],
                "zone":             r["zone"],
                "co2_intensity":    intensity,
                "is_provider_avg":  False,
            })

        # Provider-level average row
        if intensities:
            records.append({
                "provider":         provider.upper(),
                "city":             "ALL_REGIONS_AVG",
                "state":            "+".join(r["state"] for r in regions),
                "zone":             "default",
                "co2_intensity":    round(sum(intensities) / len(intensities), 2),
                "is_provider_avg":  True,
            })

    dc_df = pd.DataFrame(records)
    log.info("Datacenter carbon table: %d rows", len(dc_df))
    return dc_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — EXPORT CSV
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(state_df: pd.DataFrame, dc_df: pd.DataFrame, outdir: str) -> tuple:
    """Save processed DataFrames to CSV files."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    state_csv_path = out / "india_state_carbon.csv"
    dc_csv_path    = out / "india_datacenter_carbon.csv"

    state_df.to_csv(state_csv_path, index=False)
    dc_df.to_csv(dc_csv_path, index=False)

    log.info("Exported state CSV  → %s", state_csv_path)
    log.info("Exported DC CSV     → %s", dc_csv_path)

    print(f"\n{'='*60}")
    print("India State Carbon Intensity (gCO2/kWh) — Ember 2024 Data")
    print(f"{'='*60}")
    print(f"{'State':<42} {'gCO2/kWh':>10}  {'Fossil%':>8}  {'Clean%':>8}")
    print(f"  {'-'*72}")
    for _, row in state_df.iterrows():
        print(f"  {row['state']:<40} {row['co2_intensity_gco2_kwh']:>10.1f}  "
              f"{row['fossil_pct']:>7.1f}%  {row['clean_pct']:>7.1f}%")

    print(f"\n{'='*60}")
    print("Datacenter Carbon Intensity")
    print(f"{'='*60}")
    print(f"  {'Provider':<8} {'City':<20} {'Zone':<20} {'gCO2/kWh':>10}  {'Avg?':>6}")
    print(f"  {'-'*70}")
    for _, row in dc_df.iterrows():
        avg_flag = "← DEFAULT" if row["is_provider_avg"] else ""
        print(f"  {row['provider']:<8} {row['city']:<20} {row['zone']:<20} "
              f"{row['co2_intensity']:>10.1f}  {avg_flag}")

    return str(state_csv_path), str(dc_csv_path)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — STORE IN SQLITE
# ─────────────────────────────────────────────────────────────────────────────

def store_in_sqlite(
    state_df:  pd.DataFrame,
    dc_df:     pd.DataFrame,
    db_path:   str = "greenops.db",
) -> sqlite3.Connection:
    """
    Store the processed DataFrames into SQLite tables:
      - state_carbon_intensity   : per-state yearly carbon data
      - datacenter_carbon        : per-DC mapped intensity
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    # ── state_carbon_intensity ────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS state_carbon_intensity (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            state                   TEXT NOT NULL,
            state_code              TEXT,
            year                    INTEGER NOT NULL,
            co2_intensity_gco2_kwh  REAL NOT NULL,
            co2_intensity_weighted  REAL,
            total_generation_gwh    REAL,
            coal_pct                REAL DEFAULT 0,
            gas_pct                 REAL DEFAULT 0,
            hydro_pct               REAL DEFAULT 0,
            solar_pct               REAL DEFAULT 0,
            wind_pct                REAL DEFAULT 0,
            nuclear_pct             REAL DEFAULT 0,
            bioenergy_pct           REAL DEFAULT 0,
            other_fossil_pct        REAL DEFAULT 0,
            other_renewables_pct    REAL DEFAULT 0,
            fossil_pct              REAL DEFAULT 0,
            clean_pct               REAL DEFAULT 0,
            data_source             TEXT,
            inserted_at             REAL DEFAULT (unixepoch()),
            UNIQUE(state, year)
        )
    """)

    # ── datacenter_carbon ─────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS datacenter_carbon (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            provider        TEXT NOT NULL,
            city            TEXT NOT NULL,
            state           TEXT NOT NULL,
            zone            TEXT,
            co2_intensity   REAL NOT NULL,
            is_provider_avg INTEGER DEFAULT 0,
            inserted_at     REAL DEFAULT (unixepoch()),
            UNIQUE(provider, city)
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_state_year ON state_carbon_intensity(state, year)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_dc_provider ON datacenter_carbon(provider)")
    conn.commit()

    # Insert state data
    inserted_states = 0
    for _, row in state_df.iterrows():
        cur.execute("""
            INSERT OR REPLACE INTO state_carbon_intensity
                (state, state_code, year, co2_intensity_gco2_kwh, co2_intensity_weighted,
                 total_generation_gwh, coal_pct, gas_pct, hydro_pct, solar_pct, wind_pct,
                 nuclear_pct, bioenergy_pct, other_fossil_pct, other_renewables_pct,
                 fossil_pct, clean_pct, data_source)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            row["state"], row["state_code"], row["year"],
            row["co2_intensity_gco2_kwh"], row["co2_intensity_weighted"],
            row["total_generation_gwh"],
            row["coal_pct"], row["gas_pct"], row["hydro_pct"],
            row["solar_pct"], row["wind_pct"], row["nuclear_pct"],
            row["bioenergy_pct"], row["other_fossil_pct"], row["other_renewables_pct"],
            row["fossil_pct"], row["clean_pct"], row["data_source"],
        ))
        inserted_states += 1

    # Insert datacenter data
    inserted_dcs = 0
    for _, row in dc_df.iterrows():
        cur.execute("""
            INSERT OR REPLACE INTO datacenter_carbon
                (provider, city, state, zone, co2_intensity, is_provider_avg)
            VALUES (?,?,?,?,?,?)
        """, (
            row["provider"], row["city"], row["state"],
            row["zone"], row["co2_intensity"], int(row["is_provider_avg"]),
        ))
        inserted_dcs += 1

    conn.commit()
    log.info("SQLite: inserted %d state rows, %d DC rows → %s", inserted_states, inserted_dcs, db_path)
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CARBON ESTIMATOR (reads from SQLite)
# ─────────────────────────────────────────────────────────────────────────────

# Move these to src/config.py
CPU_FREQUENCY_GHZ = config.CPU_FREQUENCY_GHZ
CPU_TDP_WATTS     = config.CPU_TDP_WATTS
JOULES_TO_KWH     = config.JOULES_TO_KWH

OPERATION_COSTS = {
    "function_call":     10,
    "loop_iteration":     5,
    "comprehension":      8,
    "conditional":        2,
    "arithmetic":         1,
    "assignment":         1,
    "import":            50,
    "class_definition":  20,
    "exception_handler": 15,
    "context_manager":    8,
    "yield":             12,
    "lambda":             5,
    "boolean_op":         2,
    "subscript":          3,
    "attribute_access":   4,
    "augmented_assign":   2,
}


@dataclass
class OperationProfile:
    module_path:        str
    function_calls:     int = 0
    loops:              int = 0
    comprehensions:     int = 0
    conditionals:       int = 0
    assignments:        int = 0
    augmented_assigns:  int = 0
    imports:            int = 0
    class_definitions:  int = 0
    exception_handlers: int = 0
    context_managers:   int = 0
    yields:             int = 0
    lambdas:            int = 0
    boolean_ops:        int = 0
    subscripts:         int = 0
    attribute_accesses: int = 0
    arithmetic_ops:     int = 0

    def total_weighted_cycles(self) -> int:
        return (
            self.function_calls     * OPERATION_COSTS["function_call"]    +
            self.loops              * OPERATION_COSTS["loop_iteration"]    +
            self.comprehensions     * OPERATION_COSTS["comprehension"]     +
            self.conditionals       * OPERATION_COSTS["conditional"]       +
            self.assignments        * OPERATION_COSTS["assignment"]        +
            self.augmented_assigns  * OPERATION_COSTS["augmented_assign"]  +
            self.imports            * OPERATION_COSTS["import"]            +
            self.class_definitions  * OPERATION_COSTS["class_definition"]  +
            self.exception_handlers * OPERATION_COSTS["exception_handler"] +
            self.context_managers   * OPERATION_COSTS["context_manager"]   +
            self.yields             * OPERATION_COSTS["yield"]             +
            self.lambdas            * OPERATION_COSTS["lambda"]            +
            self.boolean_ops        * OPERATION_COSTS["boolean_op"]        +
            self.subscripts         * OPERATION_COSTS["subscript"]         +
            self.attribute_accesses * OPERATION_COSTS["attribute_access"]  +
            self.arithmetic_ops     * OPERATION_COSTS["arithmetic"]
        )

    def total_operations(self) -> int:
        return (
            self.function_calls + self.loops + self.comprehensions +
            self.conditionals + self.assignments + self.augmented_assigns +
            self.imports + self.class_definitions + self.exception_handlers +
            self.context_managers + self.yields + self.lambdas +
            self.boolean_ops + self.subscripts + self.attribute_accesses +
            self.arithmetic_ops
        )


class ASTOperationCounter(pyast.NodeVisitor):
    """Counts every AST operation type for clock-cycle estimation."""

    def __init__(self):
        self.profile = None

    def count(self, source_code: str, module_path: str) -> OperationProfile:
        self.profile = OperationProfile(module_path=module_path)
        try:
            self.visit(pyast.parse(source_code))
        except SyntaxError as e:
            log.warning("Syntax error in %s: %s", module_path, e)
        return self.profile

    def visit_Call(self, n):          self.profile.function_calls += 1;      self.generic_visit(n)
    def visit_For(self, n):           self.profile.loops += 1;               self.generic_visit(n)
    def visit_While(self, n):         self.profile.loops += 1;               self.generic_visit(n)
    def visit_ListComp(self, n):      self.profile.comprehensions += 1;      self.generic_visit(n)
    def visit_DictComp(self, n):      self.profile.comprehensions += 1;      self.generic_visit(n)
    def visit_SetComp(self, n):       self.profile.comprehensions += 1;      self.generic_visit(n)
    def visit_GeneratorExp(self, n):  self.profile.comprehensions += 1;      self.generic_visit(n)
    def visit_If(self, n):            self.profile.conditionals += 1;        self.generic_visit(n)
    def visit_IfExp(self, n):         self.profile.conditionals += 1;        self.generic_visit(n)
    def visit_Assign(self, n):        self.profile.assignments += 1;         self.generic_visit(n)
    def visit_AnnAssign(self, n):     self.profile.assignments += 1;         self.generic_visit(n)
    def visit_AugAssign(self, n):     self.profile.augmented_assigns += 1;   self.generic_visit(n)
    def visit_ClassDef(self, n):      self.profile.class_definitions += 1;   self.generic_visit(n)
    def visit_ExceptHandler(self, n): self.profile.exception_handlers += 1;  self.generic_visit(n)
    def visit_With(self, n):          self.profile.context_managers += 1;    self.generic_visit(n)
    def visit_Yield(self, n):         self.profile.yields += 1;              self.generic_visit(n)
    def visit_YieldFrom(self, n):     self.profile.yields += 1;              self.generic_visit(n)
    def visit_Lambda(self, n):        self.profile.lambdas += 1;             self.generic_visit(n)
    def visit_BoolOp(self, n):        self.profile.boolean_ops += 1;         self.generic_visit(n)
    def visit_Subscript(self, n):     self.profile.subscripts += 1;          self.generic_visit(n)
    def visit_Attribute(self, n):     self.profile.attribute_accesses += 1;  self.generic_visit(n)
    def visit_BinOp(self, n):         self.profile.arithmetic_ops += 1;      self.generic_visit(n)
    def visit_UnaryOp(self, n):       self.profile.arithmetic_ops += 1;      self.generic_visit(n)
    def visit_Import(self, n):        self.profile.imports += len(n.names);  self.generic_visit(n)
    def visit_ImportFrom(self, n):    self.profile.imports += len(n.names);  self.generic_visit(n)
    def visit_Compare(self, n):       self.profile.arithmetic_ops += len(n.ops); self.generic_visit(n)


@dataclass
class CarbonEstimate:
    module_path:             str
    provider:                str
    region:                  str         # city or "ALL_REGIONS_AVG"
    state:                   str
    clock_cycles:            int
    num_operations:          int
    estimated_time_s:        float
    energy_joules:           float
    energy_kwh:              float
    carbon_intensity:        float       # gCO2/kWh from DB
    carbon_intensity_source: str
    carbon_gco2:             float
    fuel_mix_note:           str = ""
    operation_profile:       Optional[OperationProfile] = None

    def summary(self) -> str:
        lines = [
            f"\n  Module      : {self.module_path}",
            f"  Provider    : {self.provider.upper()}",
            f"  Region      : {self.region} ({self.state})",
            f"  Operations  : {self.num_operations:,}",
            f"  Clock cycles: {self.clock_cycles:,}",
            f"  Energy      : {self.energy_kwh:.8f} kWh",
            f"  CO2 intensity: {self.carbon_intensity:.1f} gCO2/kWh  [{self.carbon_intensity_source}]",
            f"  Carbon      : {self.carbon_gco2:.6f} gCO2",
        ]
        if self.fuel_mix_note:
            lines.append(f"  Fuel mix    : {self.fuel_mix_note}")
        lines.append(f"  Data source : {DATA_SOURCE_CITATION}")
        return "\n".join(lines)


class IndiaDatacenterCarbonEstimator:
    """
    Estimates the carbon cost of running module tests on Indian cloud datacenters.

    Reads carbon intensity FROM the SQLite database (populated by this script).
    Supports AWS, Azure, GCP with per-city or averaged intensity.

    Decision logic:
        - Default provider: AWS
        - If region not specified: use provider average (avg of all DC cities)
        - If region specified: use that city's state intensity
    """

    def __init__(
        self,
        db_path:       str = config.GREENOPS_DB,
        provider:      str = "aws",
        region:        Optional[str] = None,   # city name e.g. "Mumbai", "Chennai"
        cpu_tdp_watts: float = config.CPU_TDP_WATTS,
        cpu_ghz:       float = config.CPU_FREQUENCY_GHZ,
        year:          int = 2024,
    ):
        self.db_path       = db_path
        self.provider      = provider.lower()
        self.region        = region             # None = use provider average
        self.cpu_tdp_watts = cpu_tdp_watts
        self.cpu_ghz       = cpu_ghz
        self.year          = year
        self.joules_per_cycle = cpu_tdp_watts / (cpu_ghz * 1e9)
        self._counter      = ASTOperationCounter()
        self._conn         = None

        # Load intensity + fuel mix from DB on init
        self._intensity, self._intensity_source, self._state, self._fuel_note = \
            self._load_intensity_from_db()

        log.info(
            "IndiaDatacenterCarbonEstimator | Provider: %s | Region: %s | "
            "State: %s | Intensity: %.1f gCO2/kWh",
            self.provider.upper(),
            self.region or "ALL_REGIONS_AVG",
            self._state,
            self._intensity,
        )

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not Path(self.db_path).exists():
                raise FileNotFoundError(
                    f"Database not found: {self.db_path}\n"
                    "Run the pipeline first:  python india_carbon_pipeline.py --csv <file>"
                )
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _load_intensity_from_db(self) -> tuple:
        """
        Query SQLite for the right intensity value.

        Logic:
          1. If region (city) specified → look up that city in datacenter_carbon
          2. If no region → look up provider's ALL_REGIONS_AVG row
          3. Fall back to state_carbon_intensity directly if DC table missing
        Returns: (intensity, source_label, state_name, fuel_mix_note)
        """
        conn = self._get_conn()
        provider_upper = self.provider.upper()

        if self.region:
            # Specific city requested
            row = conn.execute("""
                SELECT dc.co2_intensity, dc.state, dc.zone,
                       s.coal_pct, s.gas_pct, s.hydro_pct, s.solar_pct,
                       s.wind_pct, s.nuclear_pct, s.fossil_pct, s.clean_pct
                FROM datacenter_carbon dc
                LEFT JOIN state_carbon_intensity s
                    ON dc.state = s.state AND s.year = ?
                WHERE dc.provider = ? AND dc.city = ?
            """, (self.year, provider_upper, self.region)).fetchone()

            if row:
                fuel_note = self._format_fuel_note(row)
                return (
                    row["co2_intensity"],
                    f"Ember {self.year} via DB — {provider_upper}/{self.region} ({row['state']})",
                    row["state"],
                    fuel_note,
                )
            else:
                log.warning("No DB entry for %s / %s — falling back to state lookup",
                            provider_upper, self.region)
                # Try direct state lookup using city→state map
                state = CITY_TO_STATE.get(self.region)
                if state:
                    return self._load_state_intensity(state)

        # No region specified → use provider average
        row = conn.execute("""
            SELECT dc.co2_intensity, dc.state
            FROM datacenter_carbon dc
            WHERE dc.provider = ? AND dc.city = 'ALL_REGIONS_AVG'
        """, (provider_upper,)).fetchone()

        if row:
            cities = [r["city"] for r in DATACENTER_REGIONS.get(self.provider, [])]
            source = (
                f"Ember {self.year} via DB — {provider_upper} avg of "
                f"{', '.join(cities)}"
            )
            # Get fuel mix note for each constituent state
            states = list({r["state"] for r in DATACENTER_REGIONS.get(self.provider, [])})
            fuel_notes = []
            for st in states:
                st_row = conn.execute("""
                    SELECT coal_pct, solar_pct, wind_pct, hydro_pct, nuclear_pct, clean_pct
                    FROM state_carbon_intensity WHERE state=? AND year=?
                """, (st, self.year)).fetchone()
                if st_row:
                    fuel_notes.append(
                        f"{st}: Coal {st_row['coal_pct']}% | "
                        f"Solar {st_row['solar_pct']}% | Wind {st_row['wind_pct']}% | "
                        f"Clean {st_row['clean_pct']}%"
                    )
            return (
                row["co2_intensity"],
                source,
                row["state"],    # multi-state string stored here
                " | ".join(fuel_notes),
            )

        # Last resort: query state table directly for first provider state
        first_state = DATACENTER_REGIONS.get(self.provider, [{}])[0].get("state", "Maharashtra")
        return self._load_state_intensity(first_state)

    def _load_state_intensity(self, state: str) -> tuple:
        conn = self._get_conn()
        row = conn.execute("""
            SELECT co2_intensity_gco2_kwh, coal_pct, gas_pct, hydro_pct,
                   solar_pct, wind_pct, nuclear_pct, fossil_pct, clean_pct
            FROM state_carbon_intensity
            WHERE state = ? AND year = ?
        """, (state, self.year)).fetchone()

        if row:
            fuel_note = self._format_fuel_note(row)
            return (
                row["co2_intensity_gco2_kwh"],
                f"Ember {self.year} via DB — {state}",
                state,
                fuel_note,
            )
        raise ValueError(f"No intensity data found for state={state}, year={self.year}")

    @staticmethod
    def _format_fuel_note(row) -> str:
        """Build a readable fuel mix note from a DB row."""
        try:
            parts = []
            for col, label in [
                ("coal_pct", "Coal"), ("solar_pct", "Solar"),
                ("wind_pct", "Wind"), ("hydro_pct", "Hydro"),
                ("nuclear_pct", "Nuclear"), ("clean_pct", "Clean total"),
            ]:
                val = row[col] if col in row.keys() else None
                if val is not None and val > 0:
                    parts.append(f"{label} {val:.1f}%")
            return " | ".join(parts) if parts else ""
        except Exception:
            return ""

    # ── Public API ──────────────────────────────────────────────────────────

    def get_carbon_intensity(self) -> tuple:
        """Returns (intensity_gco2_kwh, source_label, state, fuel_note)."""
        return self._intensity, self._intensity_source, self._state, self._fuel_note

    def estimate_module(
        self,
        module_path:      str,
        source_code:      str,
        execution_time_s: float = 0.0,
    ) -> CarbonEstimate:
        """
        Estimate carbon for one Python module.
        Uses actual execution time if provided, static AST counting otherwise.
        """
        profile  = self._counter.count(source_code, module_path)
        intensity, source, state, fuel_note = self.get_carbon_intensity()

        if execution_time_s > 0:
            clock_cycles   = int(execution_time_s * self.cpu_ghz * 1e9)
            estimated_time = execution_time_s
        else:
            clock_cycles   = profile.total_weighted_cycles()
            estimated_time = clock_cycles / (self.cpu_ghz * 1e9)

        energy_joules = clock_cycles * self.joules_per_cycle
        energy_kwh    = energy_joules * JOULES_TO_KWH
        carbon_gco2   = energy_kwh * intensity

        return CarbonEstimate(
            module_path          = module_path,
            provider             = self.provider,
            region               = self.region or "ALL_REGIONS_AVG",
            state                = state,
            clock_cycles         = clock_cycles,
            num_operations       = profile.total_operations(),
            estimated_time_s     = estimated_time,
            energy_joules        = energy_joules,
            energy_kwh           = energy_kwh,
            carbon_intensity     = intensity,
            carbon_intensity_source = source,
            carbon_gco2          = carbon_gco2,
            fuel_mix_note        = fuel_note,
            operation_profile    = profile,
        )

    def estimate_directory(self, directory: str) -> list:
        """Estimate carbon for all .py files in a directory."""
        results = []
        for p in Path(directory).rglob("*.py"):
            try:
                source = p.read_text(encoding="utf-8", errors="replace")
                results.append(self.estimate_module(str(p), source))
            except Exception as e:
                log.warning("Skipped %s: %s", p, e)
        log.info("Directory total: %.6f gCO2 | %d files",
                 sum(r.carbon_gco2 for r in results), len(results))
        return results

    def should_run_based_on_carbon(
        self,
        probability_of_failure: float,
        carbon_intensity:        Optional[float] = None,
    ) -> bool:
        """
        Green-Ops carbon gate.
        Run test only if  Pf > carbon_score
        carbon_score = intensity / 900   (900 = dirtiest grid ceiling)
        """
        if carbon_intensity is None:
            carbon_intensity = self._intensity
        carbon_score = min(carbon_intensity / 900.0, 1.0)
        decision     = probability_of_failure > carbon_score
        log.info(
            "Carbon gate | Pf=%.3f | %.0f gCO2/kWh | score=%.3f | → %s",
            probability_of_failure, carbon_intensity,
            carbon_score, "RUN ✓" if decision else "SKIP ✗"
        )
        return decision

    def compare_providers(self) -> pd.DataFrame:
        """
        Return a DataFrame comparing all provider intensities from the DB.
        Useful for deciding which cloud region is greenest to run tests in.
        """
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT provider, city, state, zone, co2_intensity, is_provider_avg
            FROM datacenter_carbon
            ORDER BY co2_intensity ASC
        """).fetchall()
        df = pd.DataFrame([dict(r) for r in rows])
        return df

    def get_state_breakdown(self, state: str) -> dict:
        """Return full fuel mix breakdown for a state from DB."""
        conn = self._get_conn()
        row  = conn.execute("""
            SELECT * FROM state_carbon_intensity
            WHERE state = ? AND year = ?
        """, (state, self.year)).fetchone()
        return dict(row) if row else {}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="India Carbon Intensity Pipeline — Ember Dataset → CSV → SQLite → Estimator"
    )
    parser.add_argument(
        "--csv",
        default="india_yearly_full_release_long_format.csv",
        help="Path to Ember India yearly CSV",
    )
    parser.add_argument(
        "--year",    type=int, default=2024,
        help="Year to process (default: 2024)",
    )
    parser.add_argument(
        "--outdir",  default="./greenops_output",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--db",      default="greenops.db",
        help="SQLite database path",
    )
    parser.add_argument(
        "--provider", default="aws",
        choices=["aws", "azure", "gcp"],
        help="Cloud provider for carbon estimation (default: aws)",
    )
    parser.add_argument(
        "--region",  default=None,
        help="Specific DC city (e.g. Mumbai, Chennai). Omit for provider average.",
    )
    parser.add_argument(
        "--srcdir",  default=None,
        help="Source directory to estimate carbon for (optional)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Green-Ops India Carbon Pipeline")
    print(f"Data source: {DATA_SOURCE_CITATION}")
    print(f"{'='*60}\n")

    # Step 1: Process Ember dataset
    state_df = process_ember_dataset(args.csv, year=args.year)

    # Step 2: Build datacenter table
    dc_df = build_datacenter_carbon_df(state_df)

    # Step 3: Export CSVs
    state_csv, dc_csv = export_csv(state_df, dc_df, args.outdir)

    # Step 4: Store in SQLite
    store_in_sqlite(state_df, dc_df, db_path=args.db)

    # Step 5: Demo the estimator
    print(f"\n{'='*60}")
    print(f"Carbon Estimator Demo | Provider: {args.provider.upper()} | "
          f"Region: {args.region or 'ALL_REGIONS_AVG'}")
    print(f"{'='*60}")

    estimator = IndiaDatacenterCarbonEstimator(
        db_path  = args.db,
        provider = args.provider,
        region   = args.region,
        year     = args.year,
    )

    intensity, source, state, fuel_note = estimator.get_carbon_intensity()
    print(f"\n  Resolved intensity : {intensity:.1f} gCO2/kWh")
    print(f"  Source             : {source}")
    print(f"  State(s)           : {state}")
    if fuel_note:
        print(f"  Fuel mix           : {fuel_note}")

    # Show provider comparison table
    print(f"\n{'='*60}")
    print("Provider Comparison (all DC regions)")
    print(f"{'='*60}")
    df_cmp = estimator.compare_providers()
    for _, row in df_cmp.iterrows():
        default_tag = " ← DEFAULT" if row["is_provider_avg"] else ""
        print(f"  {row['provider']:<6} {row['city']:<22} {row['state']:<20} "
              f"{row['co2_intensity']:>7.1f} gCO2/kWh{default_tag}")

    # Carbon gate demo
    print(f"\n{'='*60}")
    print(f"Carbon Gate Demo | {intensity:.0f} gCO2/kWh ({source.split('—')[0].strip()})")
    print(f"{'='*60}")
    for pf in [0.1, 0.3, 0.5, 0.65, 0.79, 0.85, 0.95]:
        d = estimator.should_run_based_on_carbon(pf)
        print(f"  Pf={pf:.2f} → {'RUN  ✓' if d else 'SKIP ✗'}")

    # Estimate a source directory if provided
    if args.srcdir:
        estimates = estimator.estimate_directory(args.srcdir)
        ranked    = sorted(estimates, key=lambda e: e.carbon_gco2, reverse=True)
        print(f"\n{'='*60}")
        print(f"Module Carbon Rankings | {args.provider.upper()} / {args.region or 'AVG'}")
        print(f"{'='*60}")
        for est in ranked:
            print(f"  {est.carbon_gco2:.6f} gCO2 | {est.num_operations:>5} ops | {est.module_path}")
        print(f"\n  Total: {sum(e.carbon_gco2 for e in estimates):.6f} gCO2")

    print(f"\n{'='*60}")
    print("Outputs:")
    print(f"  State CSV    → {state_csv}")
    print(f"  DC CSV       → {Path(args.outdir) / 'india_datacenter_carbon.csv'}")
    print(f"  SQLite DB    → {args.db}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
