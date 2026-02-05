"""
FastAPI Backend for Schema Knowledge Graph UI
Provides API endpoints for the Dynamic SQL Explorer frontend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import re

from schema_graph import SchemaGraph, Table, Column, JoinEdge

app = FastAPI(title="SchemaLens API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the real estate schema graph
graph = SchemaGraph()

# Define tables matching the frontend schema
SCHEMA_DEFINITION = {
    "listings": {
        "description": "Property listings",
        "columns": ["id", "property_type_id", "neighborhood_id", "agent_id", "price", "bedrooms", "bathrooms", "square_feet", "lot_size", "year_built", "days_on_market", "status", "address", "description"],
        "primary_key": "id"
    },
    "property_types": {
        "description": "Types of properties (house, condo, etc.)",
        "columns": ["id", "name", "description"],
        "primary_key": "id"
    },
    "neighborhoods": {
        "description": "Neighborhood information",
        "columns": ["id", "name", "city", "state", "zip_code", "crime_index", "walk_score", "transit_score", "school_rating", "median_income", "population"],
        "primary_key": "id"
    },
    "agents": {
        "description": "Real estate agents",
        "columns": ["id", "brokerage_id", "name", "email", "phone", "license_number", "years_experience"],
        "primary_key": "id"
    },
    "brokerages": {
        "description": "Real estate brokerages",
        "columns": ["id", "name", "address", "phone"],
        "primary_key": "id"
    },
    "amenities": {
        "description": "Property amenities",
        "columns": ["id", "name", "category"],
        "primary_key": "id"
    },
    "listing_amenities": {
        "description": "Junction table for listings and amenities",
        "columns": ["id", "listing_id", "amenity_id"],
        "primary_key": "id"
    },
    "users": {
        "description": "Application users",
        "columns": ["id", "name", "email", "created_at"],
        "primary_key": "id"
    },
    "favorites": {
        "description": "User favorite listings",
        "columns": ["id", "user_id", "listing_id", "created_at"],
        "primary_key": "id"
    },
    "viewings": {
        "description": "Property viewing appointments",
        "columns": ["id", "user_id", "listing_id", "agent_id", "scheduled_at", "status"],
        "primary_key": "id"
    },
    "saved_searches": {
        "description": "User saved search criteria",
        "columns": ["id", "user_id", "name", "criteria", "created_at"],
        "primary_key": "id"
    },
}

# Foreign key relationships
RELATIONSHIPS = [
    ("listings", "property_type_id", "property_types", "id"),
    ("listings", "neighborhood_id", "neighborhoods", "id"),
    ("listings", "agent_id", "agents", "id"),
    ("agents", "brokerage_id", "brokerages", "id"),
    ("listing_amenities", "listing_id", "listings", "id"),
    ("listing_amenities", "amenity_id", "amenities", "id"),
    ("favorites", "user_id", "users", "id"),
    ("favorites", "listing_id", "listings", "id"),
    ("viewings", "user_id", "users", "id"),
    ("viewings", "listing_id", "listings", "id"),
    ("viewings", "agent_id", "agents", "id"),
    ("saved_searches", "user_id", "users", "id"),
]

# Build the graph
for table_name, info in SCHEMA_DEFINITION.items():
    columns = {}
    for col in info["columns"]:
        is_pk = col == info["primary_key"]
        is_fk = col.endswith("_id") and col != "id"
        columns[col] = Column(name=col, data_type="VARCHAR", is_primary_key=is_pk, is_foreign_key=is_fk)

    table = Table(name=table_name, description=info["description"], columns=columns)
    graph.add_table(table)

for from_table, from_col, to_table, to_col in RELATIONSHIPS:
    graph.add_relationship(JoinEdge(from_table, from_col, to_table, to_col))


# Pydantic models
class FilterItem(BaseModel):
    table: str
    column: str
    operator: str
    value: str

class SuggestFiltersRequest(BaseModel):
    query: str

class GenerateSQLRequest(BaseModel):
    tables: List[str]
    filters: List[FilterItem] = []
    order_by: Optional[Dict[str, str]] = None


# Keyword to table mapping for NLP-like parsing
KEYWORD_MAP = {
    "home": "listings", "homes": "listings", "house": "listings", "houses": "listings",
    "property": "listings", "properties": "listings", "listing": "listings",
    "condo": "listings", "apartment": "listings", "townhouse": "listings",
    "neighborhood": "neighborhoods", "area": "neighborhoods", "location": "neighborhoods",
    "agent": "agents", "realtor": "agents", "broker": "agents",
    "brokerage": "brokerages", "agency": "brokerages",
    "amenity": "amenities", "amenities": "amenities", "feature": "amenities",
    "user": "users", "buyer": "users", "customer": "users",
    "favorite": "favorites", "saved": "favorites",
    "viewing": "viewings", "appointment": "viewings", "tour": "viewings",
    "price": "listings", "bedroom": "listings", "bathroom": "listings",
    "crime": "neighborhoods", "walkable": "neighborhoods", "school": "neighborhoods",
}


def parse_query(query: str) -> Dict[str, Any]:
    """Parse natural language query to extract tables, filters, and order."""
    query_lower = query.lower()

    detected_tables = set()
    extracted_filters = []
    order_by = None

    # Detect tables from keywords
    for keyword, table in KEYWORD_MAP.items():
        if keyword in query_lower:
            detected_tables.add(table)

    # Extract price filters
    price_match = re.search(r'(?:under|below|less than|<)\s*\$?([\d,]+)k?', query_lower)
    if price_match:
        value = price_match.group(1).replace(',', '')
        if 'k' in query_lower[price_match.end()-2:price_match.end()+1]:
            value = str(int(value) * 1000)
        extracted_filters.append({
            "table": "listings",
            "column": "price",
            "operator": "<",
            "value": value
        })
        detected_tables.add("listings")

    price_match = re.search(r'(?:over|above|more than|>)\s*\$?([\d,]+)k?', query_lower)
    if price_match:
        value = price_match.group(1).replace(',', '')
        if 'k' in query_lower[price_match.end()-2:price_match.end()+1]:
            value = str(int(value) * 1000)
        extracted_filters.append({
            "table": "listings",
            "column": "price",
            "operator": ">",
            "value": value
        })
        detected_tables.add("listings")

    # Extract bedroom filters
    bed_match = re.search(r'(\d+)\+?\s*(?:bed|bedroom|br)', query_lower)
    if bed_match:
        extracted_filters.append({
            "table": "listings",
            "column": "bedrooms",
            "operator": ">=" if '+' in query_lower[bed_match.start():bed_match.end()+1] else "=",
            "value": bed_match.group(1)
        })
        detected_tables.add("listings")

    # Extract bathroom filters
    bath_match = re.search(r'(\d+)\+?\s*(?:bath|bathroom|ba)', query_lower)
    if bath_match:
        extracted_filters.append({
            "table": "listings",
            "column": "bathrooms",
            "operator": ">=" if '+' in query_lower[bath_match.start():bath_match.end()+1] else "=",
            "value": bath_match.group(1)
        })
        detected_tables.add("listings")

    # Detect amenity keywords
    amenity_keywords = ["pool", "garage", "garden", "gym", "parking", "balcony", "fireplace"]
    for amenity in amenity_keywords:
        if amenity in query_lower:
            extracted_filters.append({
                "table": "amenities",
                "column": "name",
                "operator": "LIKE",
                "value": f"%{amenity}%"
            })
            detected_tables.add("amenities")
            detected_tables.add("listing_amenities")
            detected_tables.add("listings")

    # Detect neighborhood keywords
    if "downtown" in query_lower:
        extracted_filters.append({
            "table": "neighborhoods",
            "column": "name",
            "operator": "LIKE",
            "value": "%downtown%"
        })
        detected_tables.add("neighborhoods")
        detected_tables.add("listings")

    # Detect safety/crime filters
    if any(word in query_lower for word in ["safe", "low crime", "safety"]):
        extracted_filters.append({
            "table": "neighborhoods",
            "column": "crime_index",
            "operator": "<",
            "value": "30"
        })
        detected_tables.add("neighborhoods")
        detected_tables.add("listings")

    # Detect walkability
    if "walkable" in query_lower or "walk score" in query_lower:
        extracted_filters.append({
            "table": "neighborhoods",
            "column": "walk_score",
            "operator": ">",
            "value": "70"
        })
        detected_tables.add("neighborhoods")
        detected_tables.add("listings")

    # Detect order by
    if "cheapest" in query_lower or "lowest price" in query_lower:
        order_by = {"column": "listings.price", "direction": "ASC"}
    elif "most expensive" in query_lower or "highest price" in query_lower:
        order_by = {"column": "listings.price", "direction": "DESC"}
    elif "newest" in query_lower:
        order_by = {"column": "listings.days_on_market", "direction": "ASC"}
    elif "largest" in query_lower or "biggest" in query_lower:
        order_by = {"column": "listings.square_feet", "direction": "DESC"}

    return {
        "detected_tables": list(detected_tables),
        "extracted_filters": extracted_filters,
        "order_by": order_by
    }


@app.get("/api/init")
async def init():
    """Return schema information for the frontend."""
    tables = []
    for table_name, info in SCHEMA_DEFINITION.items():
        tables.append({
            "name": table_name,
            "description": info["description"],
            "columns": [{"name": col, "type": "VARCHAR"} for col in info["columns"]]
        })
    return {"tables": tables}


@app.post("/api/suggest_filters")
async def suggest_filters(request: SuggestFiltersRequest):
    """Parse natural language query and suggest filters."""
    result = parse_query(request.query)
    return result


@app.post("/api/generate_sql")
async def generate_sql(request: GenerateSQLRequest):
    """Generate SQL from selected tables and filters."""
    if not request.tables:
        return {"sql": ""}

    # Build SELECT clause
    select_cols = []
    for table in request.tables:
        if table in SCHEMA_DEFINITION:
            cols = SCHEMA_DEFINITION[table]["columns"][:5]
            select_cols.extend([f"{table}.{col}" for col in cols])

    sql = f"SELECT\n  " + ",\n  ".join(select_cols[:10])

    # Build FROM/JOIN clause using schema graph
    try:
        join_sql = graph.generate_join_sql(request.tables)
        sql += f"\nFROM {join_sql}"
    except ValueError:
        sql += f"\nFROM {request.tables[0]}"

    # Build WHERE clause
    if request.filters:
        conditions = []
        for f in request.filters:
            if f.value:
                if f.operator == "LIKE":
                    conditions.append(f"{f.table}.{f.column} {f.operator} '{f.value}'")
                elif f.value.replace('.','').replace('-','').isdigit():
                    conditions.append(f"{f.table}.{f.column} {f.operator} {f.value}")
                else:
                    conditions.append(f"{f.table}.{f.column} {f.operator} '{f.value}'")

        if conditions:
            sql += "\nWHERE " + "\n  AND ".join(conditions)

    # Build ORDER BY clause
    if request.order_by:
        sql += f"\nORDER BY {request.order_by['column']} {request.order_by['direction']}"

    sql += "\nLIMIT 100"

    return {"sql": sql}


# Serve frontend
@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
