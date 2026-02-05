"""
Neon PostgreSQL Setup for Dubai Real Estate GNN
================================================
This script sets up the database schema and uploads data to Neon.

Prerequisites:
    pip install psycopg2-binary pandas h3 python-dotenv sqlalchemy

Setup:
    1. Create account at https://neon.tech (Free tier: 0.5GB)
    2. Create a new project
    3. Copy connection string to .env file
"""

import os
import pandas as pd
import h3
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from pathlib import Path

load_dotenv()

# =====================
# CONFIGURATION
# =====================

# Get from Neon dashboard: https://console.neon.tech
# Format: postgresql://user:password@ep-xxx.region.aws.neon.tech/dbname?sslmode=require
NEON_CONNECTION_STRING = os.getenv("NEON_DATABASE_URL")

DATA_DIR = Path(__file__).parent.parent / "data" / "dubai_pulse"

# H3 resolutions for different granularity
H3_RESOLUTIONS = {
    "district": 7,      # ~5.16 km²
    "neighborhood": 9,  # ~0.1 km²
    "building": 11      # ~0.01 km²
}


def get_engine():
    """Create SQLAlchemy engine for Neon."""
    if not NEON_CONNECTION_STRING:
        raise ValueError(
            "NEON_DATABASE_URL not set. Add to .env file:\n"
            "NEON_DATABASE_URL=postgresql://user:pass@ep-xxx.region.aws.neon.tech/dbname?sslmode=require"
        )
    return create_engine(NEON_CONNECTION_STRING)


def create_schema(engine):
    """Create database schema with extensions."""

    schema_sql = """
    -- Enable extensions (may need to be done via Neon console for some)
    CREATE EXTENSION IF NOT EXISTS postgis;
    CREATE EXTENSION IF NOT EXISTS vector;

    -- Note: H3 extension may need manual installation in Neon
    -- Alternative: We compute H3 in Python and store as TEXT

    -- Drop existing tables (for fresh setup)
    DROP TABLE IF EXISTS property_edges CASCADE;
    DROP TABLE IF EXISTS hex_edges CASCADE;
    DROP TABLE IF EXISTS hex_cells CASCADE;
    DROP TABLE IF EXISTS transactions CASCADE;
    DROP TABLE IF EXISTS properties CASCADE;
    DROP TABLE IF EXISTS projects CASCADE;
    DROP TABLE IF EXISTS brokers CASCADE;
    DROP TABLE IF EXISTS developers CASCADE;
    DROP TABLE IF EXISTS areas CASCADE;

    -- Areas table
    CREATE TABLE areas (
        area_id TEXT PRIMARY KEY,
        area_name TEXT NOT NULL,
        area_name_ar TEXT,
        emirate TEXT DEFAULT 'Dubai',
        district TEXT,
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        avg_price_sqft_aed NUMERIC,
        h3_res7 TEXT,
        h3_res9 TEXT,
        h3_res11 TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Properties table
    CREATE TABLE properties (
        property_id TEXT PRIMARY KEY,
        property_type TEXT,
        area_id TEXT REFERENCES areas(area_id),
        community TEXT,
        building_name TEXT,
        size_sqft INTEGER,
        bedrooms INTEGER,
        bathrooms INTEGER,
        parking_spaces INTEGER,
        floor_number INTEGER,
        view_type TEXT,
        is_furnished BOOLEAN,
        has_balcony BOOLEAN,
        year_built INTEGER,
        developer TEXT,
        h3_res9 TEXT,
        feature_vector VECTOR(128),
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Developers table
    CREATE TABLE developers (
        developer_id TEXT PRIMARY KEY,
        developer_number TEXT,
        developer_name_en TEXT,
        developer_name_ar TEXT,
        registration_date DATE,
        license_source TEXT,
        license_type TEXT,
        legal_status TEXT,
        webpage TEXT,
        phone TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Projects table
    CREATE TABLE projects (
        project_id TEXT PRIMARY KEY,
        project_number TEXT,
        project_name TEXT,
        developer_id TEXT,
        master_developer_id TEXT,
        area_id TEXT,
        area_name TEXT,
        project_status TEXT,
        percent_completed INTEGER,
        no_of_buildings INTEGER,
        no_of_villas INTEGER,
        no_of_units INTEGER,
        h3_res9 TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Brokers table
    CREATE TABLE brokers (
        broker_id TEXT PRIMARY KEY,
        broker_number TEXT,
        broker_name_en TEXT,
        broker_name_ar TEXT,
        license_start_date DATE,
        license_end_date DATE,
        phone TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Transactions table (large - will be uploaded in chunks)
    CREATE TABLE transactions (
        id SERIAL PRIMARY KEY,
        transaction_id TEXT UNIQUE,
        property_id TEXT,
        transaction_date DATE,
        transaction_type TEXT,
        amount NUMERIC,
        price_per_sqft NUMERIC,
        area_id TEXT,
        h3_res9 TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Graph tables for GNN
    CREATE TABLE hex_cells (
        h3_id TEXT PRIMARY KEY,
        resolution INTEGER,
        center_lat DOUBLE PRECISION,
        center_lng DOUBLE PRECISION,
        avg_price NUMERIC,
        property_count INTEGER DEFAULT 0,
        transaction_count INTEGER DEFAULT 0,
        feature_vector VECTOR(64),
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE hex_edges (
        id SERIAL PRIMARY KEY,
        source_h3 TEXT REFERENCES hex_cells(h3_id),
        target_h3 TEXT REFERENCES hex_cells(h3_id),
        distance INTEGER DEFAULT 1,
        edge_weight NUMERIC DEFAULT 1.0,
        UNIQUE(source_h3, target_h3)
    );

    CREATE TABLE property_edges (
        id SERIAL PRIMARY KEY,
        source_property TEXT REFERENCES properties(property_id),
        target_property TEXT REFERENCES properties(property_id),
        similarity_score NUMERIC,
        edge_type TEXT,
        UNIQUE(source_property, target_property, edge_type)
    );

    -- Indexes
    CREATE INDEX idx_properties_h3 ON properties(h3_res9);
    CREATE INDEX idx_properties_area ON properties(area_id);
    CREATE INDEX idx_areas_h3 ON areas(h3_res9);
    CREATE INDEX idx_hex_cells_res ON hex_cells(resolution);
    CREATE INDEX idx_transactions_date ON transactions(transaction_date);
    CREATE INDEX idx_transactions_h3 ON transactions(h3_res9);
    """

    with engine.connect() as conn:
        for statement in schema_sql.split(';'):
            if statement.strip():
                try:
                    conn.execute(text(statement))
                except Exception as e:
                    print(f"Warning: {e}")
        conn.commit()

    print("✅ Schema created successfully")


def compute_h3_ids(lat: float, lng: float) -> dict:
    """Compute H3 hex IDs at multiple resolutions."""
    if pd.isna(lat) or pd.isna(lng):
        return {"h3_res7": None, "h3_res9": None, "h3_res11": None}

    return {
        "h3_res7": h3.latlng_to_cell(lat, lng, 7),
        "h3_res9": h3.latlng_to_cell(lat, lng, 9),
        "h3_res11": h3.latlng_to_cell(lat, lng, 11)
    }


def upload_areas(engine):
    """Upload areas with H3 hex IDs."""
    print("📍 Uploading areas...")

    df = pd.read_csv(DATA_DIR / "areas.csv")

    # Compute H3 hex IDs
    h3_data = df.apply(
        lambda row: compute_h3_ids(row['latitude'], row['longitude']),
        axis=1
    )
    df['h3_res7'] = h3_data.apply(lambda x: x['h3_res7'])
    df['h3_res9'] = h3_data.apply(lambda x: x['h3_res9'])
    df['h3_res11'] = h3_data.apply(lambda x: x['h3_res11'])

    # Rename columns to match schema
    df = df.rename(columns={
        'area_name': 'area_name',
        'area_name_ar': 'area_name_ar'
    })

    df.to_sql('areas', engine, if_exists='append', index=False)
    print(f"   ✅ Uploaded {len(df)} areas")

    return df


def upload_properties(engine, areas_df):
    """Upload properties with H3 from their area."""
    print("🏠 Uploading properties...")

    df = pd.read_csv(DATA_DIR / "properties.csv")

    # Create area H3 lookup
    area_h3_lookup = areas_df.set_index('area_id')['h3_res9'].to_dict()

    # Map H3 from area
    df['h3_res9'] = df['area_id'].map(area_h3_lookup)

    # Convert boolean columns
    df['is_furnished'] = df['is_furnished'].map({'Yes': True, 'No': False})
    df['has_balcony'] = df['has_balcony'].map({'Yes': True, 'No': False})

    # Select columns matching schema
    cols = [
        'property_id', 'property_type', 'area_id', 'community', 'building_name',
        'size_sqft', 'bedrooms', 'bathrooms', 'parking_spaces', 'floor_number',
        'view_type', 'is_furnished', 'has_balcony', 'year_built', 'developer', 'h3_res9'
    ]
    df = df[cols]

    df.to_sql('properties', engine, if_exists='append', index=False)
    print(f"   ✅ Uploaded {len(df)} properties")


def parse_date(date_str):
    """Parse DD-MM-YYYY date format to YYYY-MM-DD."""
    if pd.isna(date_str) or date_str == '':
        return None
    try:
        return pd.to_datetime(date_str, format='%d-%m-%Y').strftime('%Y-%m-%d')
    except:
        return None


def upload_developers(engine):
    """Upload developers."""
    print("🏗️ Uploading developers...")

    df = pd.read_csv(DATA_DIR / "developers.csv")

    # Select specific columns to avoid duplicates
    df_clean = pd.DataFrame({
        'developer_id': df['participant_id'].astype(str),
        'developer_number': df['developer_number'].astype(str),
        'developer_name_en': df['developer_name_en'],
        'developer_name_ar': df['developer_name_ar'],
        'registration_date': df['registration_date'].apply(parse_date),
        'license_source': df['license_source_en'],
        'license_type': df['license_type_en'],
        'legal_status': df['legal_status_en'],
        'webpage': df['webpage'],
        'phone': df['phone']
    })

    df_clean = df_clean.drop_duplicates(subset=['developer_id'])

    df_clean.to_sql('developers', engine, if_exists='append', index=False)
    print(f"   ✅ Uploaded {len(df_clean)} developers")


def upload_projects(engine, areas_df):
    """Upload projects."""
    print("📋 Uploading projects...")

    df = pd.read_csv(DATA_DIR / "projects.csv")

    # Create area H3 lookup by area_id
    area_h3_lookup = areas_df.set_index('area_id')['h3_res9'].to_dict()

    # Select specific columns to avoid conflicts
    df_clean = pd.DataFrame({
        'project_id': df['project_id'].astype(str),
        'project_number': df['project_number'].astype(str),
        'project_name': df['project_name'],
        'developer_id': df['developer_id'].astype(str),
        'master_developer_id': df['master_developer_id'].astype(str),
        'area_id': df['area_id'].astype(str),
        'area_name': df['area_name_en'],
        'project_status': df['project_status'],
        'percent_completed': df['percent_completed'],
        'no_of_buildings': df['no_of_buildings'],
        'no_of_villas': df['no_of_villas'],
        'no_of_units': df['no_of_units']
    })

    # Map H3 from area
    df_clean['h3_res9'] = df_clean['area_id'].map(area_h3_lookup)

    df_clean = df_clean.drop_duplicates(subset=['project_id'])

    df_clean.to_sql('projects', engine, if_exists='append', index=False)
    print(f"   ✅ Uploaded {len(df_clean)} projects")


def upload_brokers(engine):
    """Upload brokers."""
    print("👔 Uploading brokers...")

    df = pd.read_csv(DATA_DIR / "brokers.csv")

    # Select specific columns
    df_clean = pd.DataFrame({
        'broker_id': df['real_estate_broker_id'].astype(str),
        'broker_number': df['broker_number'].astype(str),
        'broker_name_en': df['broker_name_en'],
        'broker_name_ar': df['broker_name_ar'],
        'license_start_date': df['license_start_date'].apply(parse_date),
        'license_end_date': df['license_end_date'].apply(parse_date),
        'phone': df['phone']
    })

    df_clean = df_clean.drop_duplicates(subset=['broker_id'])

    df_clean.to_sql('brokers', engine, if_exists='append', index=False)
    print(f"   ✅ Uploaded {len(df_clean)} brokers")


def create_hex_cells(engine, areas_df):
    """Create hex cell nodes for GNN spatial graph."""
    print("⬡ Creating hex cells for GNN...")

    # Get unique H3 cells from areas
    hex_cells = []

    for _, row in areas_df.iterrows():
        for res_name, h3_col in [("res7", "h3_res7"), ("res9", "h3_res9"), ("res11", "h3_res11")]:
            h3_id = row.get(h3_col)
            if h3_id:
                lat, lng = h3.cell_to_latlng(h3_id)
                res = int(res_name.replace("res", ""))
                hex_cells.append({
                    "h3_id": h3_id,
                    "resolution": res,
                    "center_lat": lat,
                    "center_lng": lng,
                    "avg_price": row.get('avg_price_sqft_aed'),
                    "property_count": 0
                })

    df = pd.DataFrame(hex_cells).drop_duplicates(subset=['h3_id'])
    df.to_sql('hex_cells', engine, if_exists='append', index=False)
    print(f"   ✅ Created {len(df)} hex cells")

    return df


def create_hex_edges(engine, hex_cells_df):
    """Create spatial adjacency edges between hex cells based on distance."""
    print("🔗 Creating hex edges (spatial graph)...")

    edges = []
    res9_cells = hex_cells_df[hex_cells_df['resolution'] == 9]

    # Create edges based on geographic distance (within 10km)
    MAX_DISTANCE_KM = 10

    for i, row1 in res9_cells.iterrows():
        for j, row2 in res9_cells.iterrows():
            if row1['h3_id'] >= row2['h3_id']:  # Avoid duplicates
                continue

            # Calculate distance using H3
            dist = h3.great_circle_distance(
                (row1['center_lat'], row1['center_lng']),
                (row2['center_lat'], row2['center_lng']),
                unit='km'
            )

            if dist <= MAX_DISTANCE_KM:
                # Edge weight inversely proportional to distance
                weight = 1.0 / (1.0 + dist)
                edges.append({
                    "source_h3": row1['h3_id'],
                    "target_h3": row2['h3_id'],
                    "distance": int(dist),
                    "edge_weight": round(weight, 4)
                })
                # Add reverse edge for undirected graph
                edges.append({
                    "source_h3": row2['h3_id'],
                    "target_h3": row1['h3_id'],
                    "distance": int(dist),
                    "edge_weight": round(weight, 4)
                })

    if edges:
        df = pd.DataFrame(edges).drop_duplicates(subset=['source_h3', 'target_h3'])
        df.to_sql('hex_edges', engine, if_exists='append', index=False)
        print(f"   ✅ Created {len(df)} hex edges (within {MAX_DISTANCE_KM}km)")
    else:
        print("   ⚠️ No hex edges created")


def export_for_gnn(engine):
    """Export graph data in format ready for PyTorch Geometric / DGL."""
    print("\n📦 Exporting GNN-ready data...")

    with engine.connect() as conn:
        # Node features
        nodes = pd.read_sql("""
            SELECT h3_id, resolution, center_lat, center_lng,
                   COALESCE(avg_price, 0) as avg_price,
                   property_count
            FROM hex_cells
            WHERE resolution = 9
        """, conn)

        # Edge list
        edges = pd.read_sql("""
            SELECT source_h3, target_h3, distance, edge_weight
            FROM hex_edges
        """, conn)

    # Save for GNN training
    output_dir = DATA_DIR.parent / "gnn_ready"
    output_dir.mkdir(exist_ok=True)

    nodes.to_csv(output_dir / "nodes.csv", index=False)
    edges.to_csv(output_dir / "edges.csv", index=False)

    print(f"   ✅ Saved to {output_dir}")
    print(f"   - nodes.csv: {len(nodes)} nodes")
    print(f"   - edges.csv: {len(edges)} edges")

    return output_dir


def main():
    """Main setup function."""
    print("=" * 50)
    print("🚀 Neon PostgreSQL Setup for Dubai GNN")
    print("=" * 50)

    # Check for connection string
    if not NEON_CONNECTION_STRING:
        print("""
❌ NEON_DATABASE_URL not set!

To set up:
1. Go to https://neon.tech and create free account
2. Create new project
3. Copy connection string from dashboard
4. Create .env file with:
   NEON_DATABASE_URL=postgresql://user:pass@ep-xxx.region.aws.neon.tech/dbname?sslmode=require

Then run this script again.
        """)
        return

    engine = get_engine()

    # Create schema
    create_schema(engine)

    # Upload data
    areas_df = upload_areas(engine)
    upload_properties(engine, areas_df)
    upload_developers(engine)
    upload_projects(engine, areas_df)
    upload_brokers(engine)

    # Create graph structure
    hex_cells_df = create_hex_cells(engine, areas_df)
    create_hex_edges(engine, hex_cells_df)

    # Export for GNN
    export_for_gnn(engine)

    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print("=" * 50)
    print("""
Next steps:
1. Connect Databricks to Neon:
   - Use JDBC connector: postgresql://...
   - Or use Spark PostgreSQL connector

2. For GNN training, use the exported files in data/gnn_ready/

3. Query example:
   SELECT p.*, a.h3_res9, a.latitude, a.longitude
   FROM properties p
   JOIN areas a ON p.area_id = a.area_id
   WHERE a.h3_res9 IS NOT NULL;
    """)


if __name__ == "__main__":
    main()
