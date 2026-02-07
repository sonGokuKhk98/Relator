"""
Build data_variable folder — curated ~1M row dataset from the full 17GB data.

Covers ALL 9 categories (DLD, DED, DEWA, DHA, DSC, KHDA, Municipality, RTA, Bayut)
with ALL areas, property types, transaction types represented.

Strategy:
  - Keep small reference/lookup tables COMPLETE
  - Sample large tables to target sizes (preserving diversity)
  - Total target: ~1,000,000 rows
"""

import os
import shutil
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "data_variable"

# =====================================================================
# Configuration: what to include and how much to sample
# =====================================================================

# Format: (source_path_relative_to_data, target_rows_or_None_for_full)
# None = keep full file, int = sample to that many rows

TABLES = [
    # ==============================
    # DLD — CORE REAL ESTATE (500K)
    # ==============================
    # Transactions — the heart of the data
    ("DLD/Transactions/Transactions.csv",                   200_000),
    ("DLD/Transactions/Lkp_Areas.csv",                      None),     # 302 — ALL areas
    ("DLD/Transactions/Lkp_Market_Types.csv",               None),     # 3
    ("DLD/Transactions/Lkp_Transaction_Groups.csv",         None),     # 4
    ("DLD/Transactions/Lkp_Transaction_Procedures.csv",     None),     # 65
    ("DLD/Transactions/Residential_Sale_Index.csv",         None),     # 160
    # Registrations
    ("DLD/Registrations/Rent_Contracts.csv",                150_000),
    ("DLD/Registrations/Units.csv",                         80_000),
    ("DLD/Registrations/Land_Registry.csv",                 50_000),
    ("DLD/Registrations/Buildings.csv",                     50_000),
    ("DLD/Registrations/Developers.csv",                    None),     # 2,105
    ("DLD/Registrations/Brokers.csv",                       None),     # 8,725
    ("DLD/Registrations/Projects.csv",                      None),     # 3,040
    ("DLD/Registrations/Offices.csv",                       None),     # 4,926
    ("DLD/Registrations/Accredited_Escrow_Agents.csv",      None),     # 26
    # Valuations
    ("DLD/Valuations/Valuation.csv",                        None),     # 87,945 — keep full
    ("DLD/Valuations/Valuator_Licensing.csv",               None),     # 124
    # Licenses & Permits
    ("DLD/Licenses/Real_Estate_Licenses.csv",               None),     # 2,766
    ("DLD/Licenses/Real_Estate_Permits.csv",                30_000),
    ("DLD/Licenses/Free_Zone_Companies_Licensing.csv",      None),     # 251
    ("DLD/Licenses/Licenced_Owner_Associations.csv",        None),     # 107
    # Land & Records
    ("DLD/Land/Map_Requests.csv",                           30_000),
    ("DLD/Records/oa_service_charges.csv",                  None),     # 91K — keep full

    # ==============================
    # BAYUT — MARKET DATA (32K)
    # ==============================
    ("Bayut/Transactions/bayut_transactions.csv",           None),     # 19,655
    ("Bayut/Transactions/bayut_commercial_transactions.csv",None),     # 10,021
    ("Bayut/Projects/bayut_new_projects.csv",               None),     # 1,993

    # ==============================
    # DED — BUSINESS & LICENSES (100K)
    # ==============================
    ("DED/Licenses/license_master.csv",                     50_000),
    ("DED/Licenses/business_activities.csv",                None),     # 3,894
    ("DED/Approvals/initial_approval.csv",                  30_000),
    ("DED/Inspections/inspection_report.csv",               30_000),
    ("DED/Permits/permits.csv",                             30_000),
    ("DED/Registrations/commerce_registry.csv",             30_000),
    ("DED/Registrations/trade_name.csv",                    30_000),

    # ==============================
    # DEWA — UTILITIES (60K)
    # ==============================
    ("DEWA/General/customers_master_data.csv",              50_000),
    ("DEWA/Registers/ev_green_charger.csv",                 None),     # 347
    ("DEWA/Consumption/Annual_Statistics_2024-12-31_00-00-00.csv",      None),     # 77
    ("DEWA/Consumption/gross_power_generation_2021-12-31_00-00-00.csv", None),    # 13
    ("DEWA/Consumption/water_production_2022-12-31_00-00-00.csv",      None),     # 13
    ("DEWA/Consumption/Water_Supply_MIG.csv",               None),     # 13
    ("DEWA/Procurement/open_tenders_floated_dewa_merged.csv", None),   # 95
    ("DEWA/Location/customer_happiness_center_information.csv", None), # 7
    ("DEWA/Location/water_supply_points.csv",               None),     # 6

    # ==============================
    # DHA — HEALTHCARE (45K)
    # ==============================
    ("DHA/Licenses/sheryan_professional_detail.csv",        30_000),
    ("DHA/Location/sheryan_facility_detail.csv",            None),     # 10,191
    ("DHA/Records/birth_notification.csv",                  None),     # 3,459
    ("DHA/Permits/permitted_health_insurance_tpas.csv",     None),     # 29
    ("DHA/Permits/permitted_insurers_hip.csv",              None),     # 60
    ("DHA/Permits/permitted_intermediaries_hiip.csv",       None),     # 184
    ("DHA/Permits/suspended_intermediaries.csv",            None),     # 63

    # ==============================
    # DSC — STATISTICS (35K)
    # ==============================
    ("DSC/Statistics/consumer_price_index.csv",             None),     # 1,710
    ("DSC/Statistics/gdp_quarterly.csv",                    None),     # 1,261
    ("DSC/Statistics/gross_domestic_product_at_current_prices.csv", None),  # 381
    ("DSC/Statistics/gross_domestic_product_at_constant_prices.csv", None), # 381
    ("DSC/Statistics/inflation_rate.csv",                   None),     # 15
    ("DSC/Statistics/commercial_property_index_2025-06-23_08-58-30.csv", None),     # 211
    ("DSC/Statistics/residential_property_index_2025-06-23_08-58-30.csv", None),  # 91
    ("DSC/Archive/buildings_2019-04-07_01-00-00.csv",       30_000),
    ("DSC/Archive/employment_2019-05-30_08-00-00.csv",      30_000),
    ("DSC/Archive/unemployment_2019-05-30_08-00-00.csv",    None),     # 20,516

    # ==============================
    # KHDA — EDUCATION (95K)
    # ==============================
    ("KHDA/Schools/school_search.csv",                      None),     # 242
    ("KHDA/Registers/dubai_private_schools.csv",            None),     # 221
    ("KHDA/Schools/hep_search.csv",                         None),     # 46
    ("KHDA/Schools/hep_programs.csv",                       None),     # 868
    ("KHDA/Schools/ti_search.csv",                          None),     # 1,767
    ("KHDA/Schools/ti_educationcentercourses.csv",          None),     # 92,653
    ("KHDA/Schools/inspection_grade_range.csv",             None),     # 5,686

    # ==============================
    # MUNICIPALITY — URBAN (120K)
    # ==============================
    ("Municipality/General/building_permits.csv",           30_000),
    ("Municipality/Registration/building_summary_information.csv", 30_000),
    ("Municipality/Registration/building_floor_level_information.csv", 20_000),
    ("Municipality/Contracts/consultant_projects.csv",      20_000),
    ("Municipality/Inspections/food_business_information.csv", 20_000),
    ("Municipality/General/food_item_catalogue.csv",        20_000),
    ("Municipality/Certification/food_health_certificate.csv", 20_000),
    ("Municipality/Transactions/consignments.csv",          20_000),
    ("Municipality/Applications/inprogress_applications_sla.csv", None),  # 36K
    ("Municipality/General/animal_hospitals_and_clinics.csv", None),      # 155
    ("Municipality/Location/dubai_parks_and_beaches_x_and_y_coordinates.csv", None), # 158
    ("Municipality/Location/heritage_places.csv",           None),     # 151
    ("Municipality/Archive/food_item_tests.csv",            30_000),
    ("Municipality/Registration/registered_cosmetics_and_personal_care_products.csv", 20_000),

    # ==============================
    # RTA — TRANSPORT (100K)
    # ==============================
    ("RTA/Rail/metro_stations.csv",                         None),     # 56
    ("RTA/Rail/metro_lines.csv",                            None),     # 3
    ("RTA/Bus/bus_routes.csv",                              None),     # 741
    ("RTA/Bus/bus_stop_details.csv",                        20_000),
    ("RTA/Bus/bus_network_coverage.csv",                    None),     # 228
    ("RTA/Tram/tram_stations.csv",                          None),     # 12
    ("RTA/Tram/tram_lines.csv",                             None),     # 2
    ("RTA/Marine/marine_stations.csv",                      None),     # 60
    ("RTA/Marine/marine_lines.csv",                         None),     # 29
    ("RTA/Rail/metro_ridership_2026-01-14_00-00-00.csv",     50_000),
    ("RTA/Bus/bus_ridership_2026-01-14_00-00-00.csv",       50_000),
    ("RTA/Tram/tram_ridership_2026-01-14_00-00-00.csv",     None),     # 48K
    ("RTA/Marine/marine_ridership_2026-01-17_00-00-00.csv",  None),     # 3,149
    ("RTA/Registers/public_transportation_routes_stops.csv", None),    # 18,449
    ("RTA/Registers/public_transportation_stations.csv",    None),     # 138
    ("RTA/Registers/taxi_stand_locations.csv",              None),     # 148
    ("RTA/Roads_And_Cars/drivers_population_census.csv",    None),     # 964
    ("RTA/Archive/number_of_parking_spaces_per_zone.csv",   None),     # 85

    # ==============================
    # ROOT
    # ==============================
    ("hex_master_list.csv",                                 None),     # 48,346
]


def sample_csv(src: Path, dst: Path, n_rows: int):
    """Read CSV, sample n_rows preserving diversity, write to dst."""
    try:
        # Read with low_memory=False for mixed types
        df = pd.read_csv(src, low_memory=False)
        total = len(df)

        if total <= n_rows:
            # File is smaller than target — keep full
            df.to_csv(dst, index=False)
            return total

        # Stratified sampling: try to sample evenly across areas/categories
        # to preserve diversity of the data
        strat_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(k in col_lower for k in ['area_name', 'area_en', 'property_type', 'trans_group', 'category', 'status', 'type', 'city', 'community', 'nationality', 'curriculum']):
                nunique = df[col].nunique()
                if 2 <= nunique <= 500:
                    strat_cols.append(col)
                    break

        if strat_cols:
            col = strat_cols[0]
            # Sample proportionally from each group
            sampled = df.groupby(col, group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), max(1, int(n_rows * len(x) / total))),
                                   random_state=42)
            )
            # If we didn't get enough, add more randomly
            if len(sampled) < n_rows:
                remaining = df.drop(sampled.index)
                extra = remaining.sample(n=min(len(remaining), n_rows - len(sampled)), random_state=42)
                sampled = pd.concat([sampled, extra])
            # If too many, trim
            if len(sampled) > n_rows:
                sampled = sampled.sample(n=n_rows, random_state=42)
            sampled.to_csv(dst, index=False)
            return len(sampled)
        else:
            # No good stratification column — just random sample
            sampled = df.sample(n=n_rows, random_state=42)
            sampled.to_csv(dst, index=False)
            return n_rows

    except Exception as e:
        print(f"  ERROR sampling {src.name}: {e}")
        # Fallback: just copy the file
        shutil.copy2(src, dst)
        return -1


def build():
    """Build the data_variable folder."""
    print("=" * 60)
    print("BUILDING data_variable FOLDER")
    print(f"Source: {DATA_DIR}")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

    if OUT_DIR.exists():
        print(f"Removing existing {OUT_DIR.name}/...")
        shutil.rmtree(OUT_DIR)

    total_rows = 0
    total_files = 0
    category_rows = {}

    for rel_path, target_rows in TABLES:
        src = DATA_DIR / rel_path
        dst = OUT_DIR / rel_path

        # Get category name
        parts = rel_path.split("/")
        category = parts[0] if len(parts) > 1 else "Root"

        if not src.exists():
            print(f"  SKIP (not found): {rel_path}")
            continue

        # Create output directory
        dst.parent.mkdir(parents=True, exist_ok=True)

        if target_rows is None:
            # Copy full file
            shutil.copy2(src, dst)
            # Count rows
            try:
                n = sum(1 for _ in open(src)) - 1  # subtract header
            except:
                n = 0
            print(f"  FULL  {rel_path:65s} → {n:>10,} rows")
        else:
            # Sample
            n = sample_csv(src, dst, target_rows)
            print(f"  SAMPLE {rel_path:64s} → {n:>10,} rows (target: {target_rows:,})")

        total_rows += max(n, 0)
        total_files += 1
        category_rows[category] = category_rows.get(category, 0) + max(n, 0)

    # Also copy metadata.json files
    for md in DATA_DIR.rglob("metadata.json"):
        rel = md.relative_to(DATA_DIR)
        dst = OUT_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md, dst)

    # Also copy community_to_hex_mapping.json
    mapping = DATA_DIR / "community_to_hex_mapping.json"
    if mapping.exists():
        shutil.copy2(mapping, OUT_DIR / "community_to_hex_mapping.json")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files: {total_files}")
    print(f"Total rows:  {total_rows:,}")
    print()
    print("By category:")
    for cat in sorted(category_rows.keys()):
        print(f"  {cat:20s} {category_rows[cat]:>10,} rows")
    print()
    print(f"Output: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    build()
