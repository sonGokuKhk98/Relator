# Neon + Databricks Setup for Dubai Real Estate GNN

## 1. Neon Setup (Free Tier)

### Create Account
1. Go to [neon.tech](https://neon.tech)
2. Sign up (GitHub/Google SSO available)
3. Free tier includes:
   - **0.5 GB storage**
   - **Unlimited projects** (1 active)
   - **Branching** (dev/staging/prod)
   - **Scale-to-zero** (no charges when idle)

### Create Project
1. Click "New Project"
2. Name: `dubai-real-estate-gnn`
3. Region: Choose closest to you
4. Copy connection string (format below)

```
postgresql://username:password@ep-xxx-xxx-123456.us-east-2.aws.neon.tech/neondb?sslmode=require
```

### Enable Extensions
In Neon SQL Editor, run:
```sql
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;
```

## 2. Upload Data

```bash
# Install dependencies
pip install psycopg2-binary pandas h3 python-dotenv sqlalchemy

# Create .env file
cp .env.example .env
# Edit .env with your Neon connection string

# Run setup script
python scripts/neon_setup.py
```

## 3. Databricks Integration

### Option A: JDBC Connector (Recommended)
In Databricks notebook:

```python
# Connection properties
jdbc_url = "jdbc:postgresql://ep-xxx.region.aws.neon.tech/neondb?sslmode=require"
connection_properties = {
    "user": "your_username",
    "password": "your_password",
    "driver": "org.postgresql.Driver"
}

# Read tables
properties_df = spark.read.jdbc(
    url=jdbc_url,
    table="properties",
    properties=connection_properties
)

areas_df = spark.read.jdbc(
    url=jdbc_url,
    table="areas",
    properties=connection_properties
)

# Query with H3 hex IDs
query = """
(SELECT p.property_id, p.property_type, p.size_sqft, p.bedrooms,
        a.h3_res9, a.latitude, a.longitude, a.avg_price_sqft_aed
 FROM properties p
 JOIN areas a ON p.area_id = a.area_id
 WHERE a.h3_res9 IS NOT NULL) AS dubai_properties
"""

gnn_data = spark.read.jdbc(
    url=jdbc_url,
    table=query,
    properties=connection_properties
)
```

### Option B: Neon Serverless Driver
```python
# Install in Databricks
%pip install neon-api psycopg2-binary

import psycopg2
from pyspark.sql import SparkSession

# Connect directly
conn = psycopg2.connect(
    "postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require"
)

# Use pandas then convert to Spark
import pandas as pd
df = pd.read_sql("SELECT * FROM properties", conn)
spark_df = spark.createDataFrame(df)
```

## 4. GNN Data Pipeline

### Export Graph Structure for PyTorch Geometric

```python
# In Databricks or local Python
import pandas as pd
from torch_geometric.data import Data
import torch

# Load from Neon
nodes_df = spark.read.jdbc(url, "hex_cells", connection_properties).toPandas()
edges_df = spark.read.jdbc(url, "hex_edges", connection_properties).toPandas()

# Create node index mapping
node_mapping = {h3_id: idx for idx, h3_id in enumerate(nodes_df['h3_id'])}

# Node features
x = torch.tensor(nodes_df[['center_lat', 'center_lng', 'avg_price', 'property_count']].values, dtype=torch.float)

# Edge index
edge_index = torch.tensor([
    [node_mapping[src] for src in edges_df['source_h3']],
    [node_mapping[tgt] for tgt in edges_df['target_h3']]
], dtype=torch.long)

# Create PyG Data object
data = Data(x=x, edge_index=edge_index)
print(data)
# Data(x=[N, 4], edge_index=[2, E])
```

## 5. Schema Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     NEON SCHEMA                             │
├─────────────────────────────────────────────────────────────┤
│  TABLES (Data)                                              │
│  ├── areas (21 rows) - with H3 hex IDs + lat/lng           │
│  ├── properties (31 rows) - with H3 from area              │
│  ├── transactions (large) - temporal data                   │
│  ├── developers (many) - builder info                       │
│  ├── projects (many) - project details                      │
│  └── brokers (many) - agent info                           │
│                                                             │
│  GRAPH TABLES (GNN)                                         │
│  ├── hex_cells - spatial nodes (H3 hexagons)               │
│  ├── hex_edges - spatial adjacency (k-ring neighbors)      │
│  └── property_edges - similarity edges                      │
└─────────────────────────────────────────────────────────────┘
```

## 6. Useful Queries

### Get properties with spatial context
```sql
SELECT
    p.property_id,
    p.property_type,
    p.size_sqft,
    p.bedrooms,
    a.area_name,
    a.h3_res9 as hex_id,
    a.latitude,
    a.longitude,
    a.avg_price_sqft_aed
FROM properties p
JOIN areas a ON p.area_id = a.area_id;
```

### Get hex cell neighbors (for GNN edge list)
```sql
SELECT
    e.source_h3,
    e.target_h3,
    s.avg_price as source_price,
    t.avg_price as target_price
FROM hex_edges e
JOIN hex_cells s ON e.source_h3 = s.h3_id
JOIN hex_cells t ON e.target_h3 = t.h3_id;
```

### Aggregate features per hex cell
```sql
UPDATE hex_cells h
SET
    property_count = (
        SELECT COUNT(*)
        FROM properties p
        WHERE p.h3_res9 = h.h3_id
    ),
    avg_price = (
        SELECT AVG(a.avg_price_sqft_aed)
        FROM areas a
        WHERE a.h3_res9 = h.h3_id
    );
```

## 7. Why Neon for GNN?

| Feature | Benefit for GNN |
|---------|-----------------|
| **Branching** | Test different graph structures without affecting prod |
| **Scale-to-zero** | No cost when not training |
| **pgvector** | Store node embeddings directly in DB |
| **JDBC** | Native Databricks/Spark integration |
| **Serverless** | No infrastructure management |

## 8. Next Steps

1. ✅ Set up Neon account
2. ✅ Upload Dubai data with H3 hex IDs
3. ⬜ Connect Databricks to Neon
4. ⬜ Build GNN model (GraphSAGE, GAT, etc.)
5. ⬜ Train on property price prediction
6. ⬜ Store embeddings back to Neon (pgvector)
