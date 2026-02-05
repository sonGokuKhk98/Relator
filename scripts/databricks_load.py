"""
Load Dubai Real Estate data into Databricks Unity Catalog
"""

import os
import pandas as pd
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
import time

load_dotenv()

# Configuration
CATALOG = "workspace"
SCHEMA = "dubai_real_estate"
DATA_DIR = "/Users/hiteshkaushik/dynamic_filtering/data"


def get_client():
    return WorkspaceClient(
        host=os.getenv('DATABRICKS_HOST'),
        token=os.getenv('DATABRICKS_TOKEN')
    )


def execute_sql(w, warehouse_id, sql, description=""):
    """Execute SQL and wait for result."""
    print(f"  Executing: {description}...")
    result = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql,
        wait_timeout='5m'
    )

    if result.status.state == StatementState.FAILED:
        print(f"  ❌ Failed: {result.status.error}")
        return False
    elif result.status.state == StatementState.SUCCEEDED:
        print(f"  ✅ Success")
        return True
    else:
        print(f"  ⚠️ State: {result.status.state}")
        return False


def create_tables(w, warehouse_id):
    """Create all tables."""
    print("\n📋 Creating tables...")

    tables_sql = f"""
    -- Areas table with H3 hex IDs
    CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.areas (
        area_id STRING,
        area_name STRING,
        area_name_ar STRING,
        emirate STRING,
        district STRING,
        latitude DOUBLE,
        longitude DOUBLE,
        avg_price_sqft_aed DOUBLE,
        h3_res7 STRING,
        h3_res9 STRING,
        h3_res11 STRING
    ) USING DELTA;

    -- Properties table
    CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.properties (
        property_id STRING,
        property_type STRING,
        area_id STRING,
        community STRING,
        building_name STRING,
        size_sqft INT,
        bedrooms INT,
        bathrooms INT,
        parking_spaces INT,
        floor_number INT,
        view_type STRING,
        is_furnished BOOLEAN,
        has_balcony BOOLEAN,
        year_built INT,
        developer STRING,
        h3_res9 STRING
    ) USING DELTA;

    -- GNN Nodes (hex cells)
    CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.gnn_nodes (
        h3_id STRING,
        resolution INT,
        center_lat DOUBLE,
        center_lng DOUBLE,
        avg_price DOUBLE,
        property_count INT
    ) USING DELTA;

    -- GNN Edges (spatial graph)
    CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.gnn_edges (
        source_h3 STRING,
        target_h3 STRING,
        distance INT,
        edge_weight DOUBLE
    ) USING DELTA;
    """

    # Execute each statement separately
    for stmt in tables_sql.strip().split(';'):
        stmt = stmt.strip()
        if stmt and not stmt.startswith('--'):
            if 'CREATE TABLE' in stmt:
                table_name = stmt.split('EXISTS')[1].split('(')[0].strip()
                execute_sql(w, warehouse_id, stmt, f"Creating {table_name}")


def escape_sql_string(s):
    """Escape string for SQL."""
    if pd.isna(s) or s is None:
        return "NULL"
    s = str(s).replace("'", "''").replace("\\", "\\\\")
    return f"'{s}'"


def load_areas(w, warehouse_id):
    """Load areas data."""
    print("\n📍 Loading areas...")

    # First, clear existing data
    execute_sql(w, warehouse_id, f"DELETE FROM {CATALOG}.{SCHEMA}.areas", "Clearing areas")

    # Read and prepare data
    import h3
    df = pd.read_csv(f"{DATA_DIR}/dubai_pulse/areas.csv")

    # Compute H3 hex IDs
    def compute_h3(row):
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            return None, None, None
        lat, lng = row['latitude'], row['longitude']
        return (
            h3.latlng_to_cell(lat, lng, 7),
            h3.latlng_to_cell(lat, lng, 9),
            h3.latlng_to_cell(lat, lng, 11)
        )

    h3_data = df.apply(compute_h3, axis=1)
    df['h3_res7'] = [x[0] for x in h3_data]
    df['h3_res9'] = [x[1] for x in h3_data]
    df['h3_res11'] = [x[2] for x in h3_data]

    # Build INSERT statement
    values = []
    for _, row in df.iterrows():
        v = f"""({escape_sql_string(row['area_id'])}, {escape_sql_string(row['area_name'])},
                 {escape_sql_string(row.get('area_name_ar'))}, {escape_sql_string(row.get('emirate', 'Dubai'))},
                 {escape_sql_string(row.get('district'))}, {row['latitude']}, {row['longitude']},
                 {row.get('avg_price_sqft_aed', 'NULL')}, {escape_sql_string(row['h3_res7'])},
                 {escape_sql_string(row['h3_res9'])}, {escape_sql_string(row['h3_res11'])})"""
        values.append(v)

    insert_sql = f"""
    INSERT INTO {CATALOG}.{SCHEMA}.areas
    (area_id, area_name, area_name_ar, emirate, district, latitude, longitude,
     avg_price_sqft_aed, h3_res7, h3_res9, h3_res11)
    VALUES {', '.join(values)}
    """

    execute_sql(w, warehouse_id, insert_sql, f"Inserting {len(df)} areas")
    return df


def load_properties(w, warehouse_id, areas_df):
    """Load properties data."""
    print("\n🏠 Loading properties...")

    execute_sql(w, warehouse_id, f"DELETE FROM {CATALOG}.{SCHEMA}.properties", "Clearing properties")

    df = pd.read_csv(f"{DATA_DIR}/dubai_pulse/properties.csv")

    # Map H3 from areas
    area_h3_map = dict(zip(areas_df['area_id'], areas_df['h3_res9']))
    df['h3_res9'] = df['area_id'].map(area_h3_map)

    # Convert booleans
    df['is_furnished'] = df['is_furnished'].map({'Yes': True, 'No': False, True: True, False: False})
    df['has_balcony'] = df['has_balcony'].map({'Yes': True, 'No': False, True: True, False: False})

    values = []
    for _, row in df.iterrows():
        furnished = 'TRUE' if row.get('is_furnished') else 'FALSE'
        balcony = 'TRUE' if row.get('has_balcony') else 'FALSE'

        v = f"""({escape_sql_string(row['property_id'])}, {escape_sql_string(row['property_type'])},
                 {escape_sql_string(row['area_id'])}, {escape_sql_string(row.get('community'))},
                 {escape_sql_string(row.get('building_name'))}, {int(row.get('size_sqft', 0)) or 'NULL'},
                 {int(row.get('bedrooms', 0)) if pd.notna(row.get('bedrooms')) else 'NULL'},
                 {int(row.get('bathrooms', 0)) if pd.notna(row.get('bathrooms')) else 'NULL'},
                 {int(row.get('parking_spaces', 0)) if pd.notna(row.get('parking_spaces')) else 'NULL'},
                 {int(row.get('floor_number', 0)) if pd.notna(row.get('floor_number')) else 'NULL'},
                 {escape_sql_string(row.get('view_type'))}, {furnished}, {balcony},
                 {int(row.get('year_built')) if pd.notna(row.get('year_built')) else 'NULL'},
                 {escape_sql_string(row.get('developer'))}, {escape_sql_string(row.get('h3_res9'))})"""
        values.append(v)

    insert_sql = f"""
    INSERT INTO {CATALOG}.{SCHEMA}.properties
    (property_id, property_type, area_id, community, building_name, size_sqft,
     bedrooms, bathrooms, parking_spaces, floor_number, view_type, is_furnished,
     has_balcony, year_built, developer, h3_res9)
    VALUES {', '.join(values)}
    """

    execute_sql(w, warehouse_id, insert_sql, f"Inserting {len(df)} properties")


def load_gnn_data(w, warehouse_id):
    """Load GNN nodes and edges."""
    print("\n⬡ Loading GNN graph data...")

    # Load nodes
    execute_sql(w, warehouse_id, f"DELETE FROM {CATALOG}.{SCHEMA}.gnn_nodes", "Clearing nodes")

    nodes_df = pd.read_csv(f"{DATA_DIR}/gnn_ready/nodes.csv")

    values = []
    for _, row in nodes_df.iterrows():
        v = f"""({escape_sql_string(row['h3_id'])}, {int(row['resolution'])},
                 {row['center_lat']}, {row['center_lng']},
                 {row['avg_price'] if pd.notna(row['avg_price']) else 'NULL'},
                 {int(row['property_count']) if pd.notna(row['property_count']) else 0})"""
        values.append(v)

    insert_sql = f"""
    INSERT INTO {CATALOG}.{SCHEMA}.gnn_nodes
    (h3_id, resolution, center_lat, center_lng, avg_price, property_count)
    VALUES {', '.join(values)}
    """
    execute_sql(w, warehouse_id, insert_sql, f"Inserting {len(nodes_df)} nodes")

    # Load edges
    execute_sql(w, warehouse_id, f"DELETE FROM {CATALOG}.{SCHEMA}.gnn_edges", "Clearing edges")

    edges_df = pd.read_csv(f"{DATA_DIR}/gnn_ready/edges.csv")

    values = []
    for _, row in edges_df.iterrows():
        v = f"""({escape_sql_string(row['source_h3'])}, {escape_sql_string(row['target_h3'])},
                 {int(row['distance'])}, {row['edge_weight']})"""
        values.append(v)

    insert_sql = f"""
    INSERT INTO {CATALOG}.{SCHEMA}.gnn_edges
    (source_h3, target_h3, distance, edge_weight)
    VALUES {', '.join(values)}
    """
    execute_sql(w, warehouse_id, insert_sql, f"Inserting {len(edges_df)} edges")


def verify_data(w, warehouse_id):
    """Verify loaded data."""
    print("\n📊 Verifying data...")

    tables = ['areas', 'properties', 'gnn_nodes', 'gnn_edges']

    for table in tables:
        result = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=f"SELECT COUNT(*) as cnt FROM {CATALOG}.{SCHEMA}.{table}",
            wait_timeout='30s'
        )

        if result.status.state == StatementState.SUCCEEDED:
            count = result.result.data_array[0][0] if result.result.data_array else 0
            print(f"   {table}: {count} rows")


def main():
    print("=" * 50)
    print("🚀 Loading Dubai Real Estate Data to Databricks")
    print("=" * 50)

    w = get_client()

    # Get warehouse
    warehouses = list(w.warehouses.list())
    if not warehouses:
        print("❌ No SQL warehouses found!")
        return

    warehouse_id = warehouses[0].id
    print(f"Using warehouse: {warehouses[0].name}")

    # Create tables
    create_tables(w, warehouse_id)

    # Load data
    areas_df = load_areas(w, warehouse_id)
    load_properties(w, warehouse_id, areas_df)
    load_gnn_data(w, warehouse_id)

    # Verify
    verify_data(w, warehouse_id)

    print("\n" + "=" * 50)
    print("✅ Data loaded to Databricks Unity Catalog!")
    print("=" * 50)
    print(f"""
Tables created in: {CATALOG}.{SCHEMA}

Query examples:
  SELECT * FROM {CATALOG}.{SCHEMA}.areas;
  SELECT * FROM {CATALOG}.{SCHEMA}.properties;
  SELECT * FROM {CATALOG}.{SCHEMA}.gnn_nodes;
  SELECT * FROM {CATALOG}.{SCHEMA}.gnn_edges;

GNN Graph Query:
  SELECT
    a1.area_name as from_area,
    a2.area_name as to_area,
    e.distance,
    e.edge_weight
  FROM {CATALOG}.{SCHEMA}.gnn_edges e
  JOIN {CATALOG}.{SCHEMA}.areas a1 ON e.source_h3 = a1.h3_res9
  JOIN {CATALOG}.{SCHEMA}.areas a2 ON e.target_h3 = a2.h3_res9
  ORDER BY e.edge_weight DESC
  LIMIT 10;
    """)


if __name__ == "__main__":
    main()
