"""
Pre-build SQLite database from CSVs for production deployment.
Run once locally, commit the .db file, skip CSV loading on Render.

Usage:
    python build_db.py              # uses data_demo/
    python build_db.py variable     # uses data_variable/
"""

import sqlite3
import pandas as pd
import math
import sys
from pathlib import Path

DATA_MODES = {
    "demo": Path(__file__).parent / "data_demo",
    "variable": Path(__file__).parent / "data_variable",
}

DB_FILE = Path(__file__).parent / "production.db"


def haversine(lat1, lon1, lat2, lon2):
    if any(v is None for v in (lat1, lon1, lat2, lon2)):
        return None
    R = 6371.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def clean_table_name(csv_path, data_dir):
    rel = csv_path.relative_to(data_dir)
    parts = list(rel.parts)
    name = "_".join(parts).replace(".csv", "").replace("-", "_").replace(" ", "_").lower()
    # Remove leading digits/underscores
    while name and (name[0].isdigit() or name[0] == "_"):
        name = name[1:]
    return name or "unnamed_table"


def build():
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    data_dir = DATA_MODES.get(mode)
    if not data_dir or not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print(f"Available modes: {list(DATA_MODES.keys())}")
        sys.exit(1)

    print(f"Building production.db from {data_dir} ...")

    # Remove old DB
    if DB_FILE.exists():
        DB_FILE.unlink()

    conn = sqlite3.connect(str(DB_FILE))
    conn.create_function("haversine", 4, haversine)

    csv_files = sorted(data_dir.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files\n")

    loaded = {}
    total_rows = 0

    for csv_file in csv_files:
        table_name = clean_table_name(csv_file, data_dir)

        # Handle name collisions
        if table_name in loaded:
            category = csv_file.parent.name.lower().replace(" ", "_").replace("-", "_")
            table_name = f"{category}_{table_name}"

        try:
            df = pd.read_csv(csv_file, low_memory=False)
            # Clean column names
            df.columns = [
                c.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
                for c in df.columns
            ]
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            loaded[table_name] = len(df)
            total_rows += len(df)
            print(f"  {table_name}: {len(df):,} rows")
        except Exception as e:
            print(f"  SKIP {csv_file.name}: {e}")

    # Create performance indexes
    INDEX_TARGETS = [
        ("transactions", "area_name_en"),
        ("transactions", "property_type_en"),
        ("transactions", "trans_group_en"),
        ("transactions", "actual_worth"),
        ("transactions", "instance_date"),
        ("transactions", "rooms_en"),
        ("bayut_transactions", "area"),
        ("bayut_transactions", "property_type"),
        ("bayut_transactions", "price"),
        ("bayut_transactions", "latitude"),
        ("bayut_transactions", "longitude"),
        ("rent_contracts", "area_name_en"),
        ("rent_contracts", "annual_amount"),
        ("metro_stations", "station_location_latitude"),
        ("metro_stations", "station_location_longitude"),
    ]

    print("\nCreating indexes...")
    for table, col in INDEX_TARGETS:
        try:
            idx_name = f"idx_{table}_{col}"
            conn.execute(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table}" ("{col}")')
        except Exception:
            pass

    conn.execute("ANALYZE")
    conn.commit()

    # Vacuum to minimize file size
    conn.execute("VACUUM")
    conn.close()

    size_mb = DB_FILE.stat().st_size / (1024 * 1024)
    print(f"\nDone! {len(loaded)} tables, {total_rows:,} total rows")
    print(f"Output: {DB_FILE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    build()
