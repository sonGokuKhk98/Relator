"""
FastAPI Backend for Dynamic Filtering UI
Netflix-Inspired LLM Layer Implementation

Features:
- AST-based Filter DSL with validation
- Multi-strategy extraction (LLM, Hybrid, Regex)
- Entity resolution with @mentions
- UI Chip generation
- Caching and metrics
"""

import os
import time
import json
import sqlite3
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Any
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import json as pyjson

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Google Gemini for LLM-based query extraction
import google.generativeai as genai

# Import the Netflix-inspired LLM layer
from llm_layer import (
    SmartExtractor,
    ExtractionCache,
    MetricsCollector,
    ChipGenerator,
    EntityResolver,
    FilterValidator,
    FilterAST,
    FilterNode,
    FilterGroup,
    FilterOperator,
    LogicalOperator,
)
from llm_layer.validator import SchemaRegistry

# Configure Gemini API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")
else:
    gemini_model = None
    print("Warning: GOOGLE_API_KEY not set. LLM extraction will be disabled.")

# Global components
extraction_cache = ExtractionCache(max_size=1000, ttl_seconds=3600)
metrics_collector = MetricsCollector(max_log_size=10000)
smart_extractor = None  # Initialized after DB is ready
entity_resolver = None
chip_generator = ChipGenerator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and components on startup."""
    global smart_extractor, entity_resolver

    init_database()

    # Initialize entity resolver with DB connection
    entity_resolver = EntityResolver(get_db())

    # Initialize smart extractor
    smart_extractor = SmartExtractor(
        llm_model=gemini_model,
        schema_definition=SCHEMA,
        db_connection=get_db(),
        cache=extraction_cache,
    )

    print("Netflix-inspired LLM Layer initialized!")
    yield


app = FastAPI(
    title="Dynamic Filtering API",
    description="Netflix-inspired intelligent query extraction with AST-based filtering",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directory - Dubai Proptech Data
# Use DATA_MODE=demo to use smaller dataset (12MB vs 17GB)
DATA_MODE = os.environ.get("DATA_MODE", "demo")  # "full" or "demo"
if DATA_MODE == "demo":
    DATA_DIR = Path(__file__).parent / "data_demo"
    print("Using DEMO data (12MB) - set DATA_MODE=full for complete dataset")
else:
    DATA_DIR = Path(__file__).parent / "data"
    print("Using FULL data (17GB)")
ANALYSIS_LOG_PATH = Path(__file__).parent / "feedback" / "analysis_log.jsonl"

# In-memory SQLite database
DB_CONNECTION = None

# Schema definition for the UI - Dubai Proptech Data Model
SCHEMA = [
    # ==========================================
    # DLD - TRANSACTIONS & REGISTRATIONS
    # ==========================================
    {
        "name": "transactions",
        "description": "DLD property transactions (sales, mortgages, gifts)",
        "columns": [
            {"name": "transaction_id", "type": "TEXT", "is_pk": True},
            {"name": "procedure_id", "type": "INTEGER"},
            {"name": "trans_group_id", "type": "INTEGER"},
            {"name": "trans_group_en", "type": "TEXT"},
            {"name": "procedure_name_en", "type": "TEXT"},
            {"name": "instance_date", "type": "TEXT"},
            {"name": "property_type_id", "type": "INTEGER"},
            {"name": "property_type_en", "type": "TEXT"},
            {"name": "property_sub_type_en", "type": "TEXT"},
            {"name": "property_usage_en", "type": "TEXT"},
            {"name": "reg_type_en", "type": "TEXT"},
            {"name": "area_id", "type": "INTEGER"},
            {"name": "area_name_en", "type": "TEXT"},
            {"name": "building_name_en", "type": "TEXT"},
            {"name": "project_name_en", "type": "TEXT"},
            {"name": "master_project_en", "type": "TEXT"},
            {"name": "nearest_metro_en", "type": "TEXT"},
            {"name": "nearest_mall_en", "type": "TEXT"},
            {"name": "rooms_en", "type": "TEXT"},
            {"name": "has_parking", "type": "INTEGER"},
            {"name": "procedure_area", "type": "REAL"},
            {"name": "actual_worth", "type": "REAL"},
            {"name": "meter_sale_price", "type": "REAL"},
            {"name": "rent_value", "type": "REAL"},
        ]
    },
    {
        "name": "units",
        "description": "DLD registered property units (apartments, villas)",
        "columns": [
            {"name": "property_id", "type": "INTEGER", "is_pk": True},
            {"name": "area_id", "type": "INTEGER"},
            {"name": "area_name_en", "type": "TEXT"},
            {"name": "unit_number", "type": "TEXT"},
            {"name": "floor", "type": "TEXT"},
            {"name": "rooms", "type": "INTEGER"},
            {"name": "rooms_en", "type": "TEXT"},
            {"name": "actual_area", "type": "REAL"},
            {"name": "property_type_en", "type": "TEXT"},
            {"name": "property_sub_type_en", "type": "TEXT"},
            {"name": "project_name_en", "type": "TEXT"},
            {"name": "master_project_en", "type": "TEXT"},
            {"name": "is_free_hold", "type": "INTEGER"},
            {"name": "is_registered", "type": "INTEGER"},
        ]
    },
    {
        "name": "land_registry",
        "description": "DLD land parcels registry",
        "columns": [
            {"name": "property_id", "type": "INTEGER", "is_pk": True},
            {"name": "area_id", "type": "INTEGER"},
            {"name": "area_name_en", "type": "TEXT"},
            {"name": "land_number", "type": "TEXT"},
            {"name": "actual_area", "type": "REAL"},
            {"name": "property_type_en", "type": "TEXT"},
            {"name": "property_sub_type_en", "type": "TEXT"},
            {"name": "parcel_id", "type": "TEXT"},
            {"name": "is_free_hold", "type": "INTEGER"},
            {"name": "project_name_en", "type": "TEXT"},
            {"name": "master_project_en", "type": "TEXT"},
            {"name": "land_type_en", "type": "TEXT"},
        ]
    },
    {
        "name": "rent_contracts",
        "description": "DLD Ejari rental contracts",
        "columns": [
            {"name": "contract_id", "type": "TEXT", "is_pk": True},
            {"name": "contract_reg_type_en", "type": "TEXT"},
            {"name": "contract_start_date", "type": "TEXT"},
            {"name": "contract_end_date", "type": "TEXT"},
            {"name": "contract_amount", "type": "REAL"},
            {"name": "annual_amount", "type": "REAL"},
            {"name": "ejari_property_type_en", "type": "TEXT"},
            {"name": "ejari_property_sub_type_en", "type": "TEXT"},
            {"name": "property_usage_en", "type": "TEXT"},
            {"name": "project_name_en", "type": "TEXT"},
            {"name": "master_project_en", "type": "TEXT"},
            {"name": "area_id", "type": "INTEGER"},
            {"name": "area_name_en", "type": "TEXT"},
            {"name": "actual_area", "type": "REAL"},
            {"name": "tenant_type_en", "type": "TEXT"},
        ]
    },
    # ==========================================
    # DLD - STAKEHOLDERS
    # ==========================================
    {
        "name": "developers",
        "description": "DLD registered real estate developers",
        "columns": [
            {"name": "participant_id", "type": "INTEGER", "is_pk": True},
            {"name": "developer_id", "type": "INTEGER"},
            {"name": "developer_number", "type": "TEXT"},
            {"name": "developer_name_en", "type": "TEXT"},
            {"name": "registration_date", "type": "TEXT"},
            {"name": "license_source_en", "type": "TEXT"},
            {"name": "license_number", "type": "TEXT"},
            {"name": "license_issue_date", "type": "TEXT"},
            {"name": "license_expiry_date", "type": "TEXT"},
            {"name": "legal_status_en", "type": "TEXT"},
            {"name": "webpage", "type": "TEXT"},
            {"name": "phone", "type": "TEXT"},
        ]
    },
    {
        "name": "brokers",
        "description": "DLD licensed real estate brokers",
        "columns": [
            {"name": "participant_id", "type": "INTEGER", "is_pk": True},
            {"name": "real_estate_broker_id", "type": "INTEGER"},
            {"name": "broker_number", "type": "TEXT"},
            {"name": "broker_name_en", "type": "TEXT"},
            {"name": "license_start_date", "type": "TEXT"},
            {"name": "license_end_date", "type": "TEXT"},
            {"name": "real_estate_id", "type": "INTEGER"},
        ]
    },
    # ==========================================
    # DLD - LOOKUPS
    # ==========================================
    {
        "name": "lkp_areas",
        "description": "Dubai areas/districts lookup",
        "columns": [
            {"name": "area_id", "type": "INTEGER", "is_pk": True},
            {"name": "name_en", "type": "TEXT"},
            {"name": "name_ar", "type": "TEXT"},
            {"name": "municipality_number", "type": "TEXT"},
        ]
    },
    {
        "name": "lkp_transaction_groups",
        "description": "Transaction type groups (Sales, Mortgages, Gifts)",
        "columns": [
            {"name": "group_id", "type": "INTEGER", "is_pk": True},
            {"name": "name_en", "type": "TEXT"},
            {"name": "name_ar", "type": "TEXT"},
        ]
    },
    {
        "name": "lkp_transaction_procedures",
        "description": "Transaction procedures lookup",
        "columns": [
            {"name": "procedure_id", "type": "INTEGER", "is_pk": True},
            {"name": "group_id", "type": "INTEGER"},
            {"name": "name_en", "type": "TEXT"},
            {"name": "is_pre_registration", "type": "INTEGER"},
        ]
    },
    # ==========================================
    # BAYUT - MARKET DATA
    # ==========================================
    {
        "name": "bayut_transactions",
        "description": "Bayut real estate transactions with market data",
        "columns": [
            {"name": "transaction_hash_id", "type": "TEXT", "is_pk": True},
            {"name": "date_transaction", "type": "TEXT"},
            {"name": "transaction_category", "type": "TEXT"},
            {"name": "completion_status", "type": "TEXT"},
            {"name": "location_area", "type": "TEXT"},
            {"name": "location_sub_area", "type": "TEXT"},
            {"name": "location_building", "type": "TEXT"},
            {"name": "property_type_id", "type": "INTEGER"},
            {"name": "beds", "type": "INTEGER"},
            {"name": "floor", "type": "INTEGER"},
            {"name": "builtup_area_sqm", "type": "REAL"},
            {"name": "transaction_amount", "type": "REAL"},
            {"name": "transaction_per_sqm_amount", "type": "REAL"},
            {"name": "rental_yield", "type": "REAL"},
            {"name": "sale_market", "type": "TEXT"},
            {"name": "seller_type", "type": "TEXT"},
            {"name": "payment_type", "type": "TEXT"},
            {"name": "developer_name", "type": "TEXT"},
            {"name": "latitude", "type": "REAL"},
            {"name": "longitude", "type": "REAL"},
        ]
    },
    # ==========================================
    # RTA - TRANSPORT
    # ==========================================
    {
        "name": "metro_stations",
        "description": "Dubai Metro stations",
        "columns": [
            {"name": "station_id", "type": "INTEGER", "is_pk": True},
            {"name": "station_name", "type": "TEXT"},
            {"name": "line_name", "type": "TEXT"},
            {"name": "latitude", "type": "REAL"},
            {"name": "longitude", "type": "REAL"},
        ]
    },
    {
        "name": "bus_routes",
        "description": "Dubai bus routes",
        "columns": [
            {"name": "route_id", "type": "INTEGER", "is_pk": True},
            {"name": "route_name", "type": "TEXT"},
            {"name": "route_type", "type": "TEXT"},
        ]
    },
    # ==========================================
    # KHDA - EDUCATION
    # ==========================================
    {
        "name": "school_search",
        "description": "KHDA Dubai schools directory",
        "columns": [
            {"name": "school_id", "type": "INTEGER", "is_pk": True},
            {"name": "school_name", "type": "TEXT"},
            {"name": "curriculum", "type": "TEXT"},
            {"name": "area", "type": "TEXT"},
            {"name": "rating", "type": "TEXT"},
        ]
    },
    # ==========================================
    # DHA - HEALTH
    # ==========================================
    {
        "name": "sheryan_facility_detail",
        "description": "DHA healthcare facilities",
        "columns": [
            {"name": "facility_id", "type": "INTEGER", "is_pk": True},
            {"name": "facility_name", "type": "TEXT"},
            {"name": "facility_type", "type": "TEXT"},
            {"name": "area", "type": "TEXT"},
        ]
    },
]

# Schema relationships for graph visualization - Dubai Proptech Model
SCHEMA_EDGES = [
    # ==========================================
    # DLD Transactions relationships
    # ==========================================
    {"from": "transactions", "to": "lkp_areas"},
    {"from": "transactions", "to": "lkp_transaction_groups"},
    {"from": "transactions", "to": "lkp_transaction_procedures"},
    {"from": "transactions", "to": "developers"},
    {"from": "transactions", "to": "units"},

    # ==========================================
    # DLD Units relationships
    # ==========================================
    {"from": "units", "to": "lkp_areas"},
    {"from": "units", "to": "developers"},
    {"from": "units", "to": "land_registry"},

    # ==========================================
    # DLD Land Registry relationships
    # ==========================================
    {"from": "land_registry", "to": "lkp_areas"},
    {"from": "land_registry", "to": "developers"},

    # ==========================================
    # DLD Rent Contracts relationships
    # ==========================================
    {"from": "rent_contracts", "to": "lkp_areas"},
    {"from": "rent_contracts", "to": "units"},
    {"from": "rent_contracts", "to": "developers"},

    # ==========================================
    # Lookup table relationships
    # ==========================================
    {"from": "lkp_transaction_procedures", "to": "lkp_transaction_groups"},

    # ==========================================
    # DLD Stakeholders
    # ==========================================
    {"from": "brokers", "to": "developers"},

    # ==========================================
    # Bayut Market Data
    # ==========================================
    {"from": "bayut_transactions", "to": "lkp_areas"},
    {"from": "bayut_transactions", "to": "developers"},

    # ==========================================
    # Valuations
    # ==========================================
    {"from": "valuation", "to": "lkp_areas"},
    {"from": "valuation", "to": "transactions"},

    # ==========================================
    # Real Estate Licenses
    # ==========================================
    {"from": "real_estate_licenses", "to": "developers"},

    # ==========================================
    # Cross-domain: Transport proximity
    # ==========================================
    {"from": "transactions", "to": "metro_stations"},
    {"from": "bayut_transactions", "to": "metro_stations"},

    # ==========================================
    # Cross-domain: Education proximity
    # ==========================================
    {"from": "units", "to": "school_search"},
    {"from": "bayut_transactions", "to": "school_search"},
    {"from": "school_search", "to": "lkp_areas"},

    # ==========================================
    # Cross-domain: Healthcare proximity
    # ==========================================
    {"from": "units", "to": "sheryan_facility_detail"},
    {"from": "sheryan_facility_detail", "to": "lkp_areas"},
]


def init_database():
    """Load CSV files into SQLite in-memory database - Dubai Proptech Data."""
    global DB_CONNECTION

    DB_CONNECTION = sqlite3.connect(":memory:", check_same_thread=False)

    # Map of table names to their CSV file paths (relative to DATA_DIR)
    csv_mapping = {
        # DLD - Transactions & Registrations
        "transactions": "DLD/Transactions/Transactions.csv",
        "units": "DLD/Registrations/Units.csv",
        "land_registry": "DLD/Registrations/Land_Registry.csv",
        "rent_contracts": "DLD/Registrations/Rent_Contracts.csv",
        # DLD - Stakeholders
        "developers": "DLD/Registrations/Developers.csv",
        "brokers": "DLD/Registrations/Brokers.csv",
        # DLD - Lookups
        "lkp_areas": "DLD/Transactions/Lkp_Areas.csv",
        "lkp_transaction_groups": "DLD/Transactions/Lkp_Transaction_Groups.csv",
        "lkp_transaction_procedures": "DLD/Transactions/Lkp_Transaction_Procedures.csv",
        "lkp_market_types": "DLD/Transactions/Lkp_Market_Types.csv",
        # Bayut - Market Data
        "bayut_transactions": "Bayut/Transactions/bayut_transactions.csv",
        "bayut_commercial_transactions": "Bayut/Transactions/bayut_commercial_transactions.csv",
        # RTA - Transport
        "metro_stations": "RTA/Rail/metro_stations.csv",
        "metro_lines": "RTA/Rail/metro_lines.csv",
        "bus_routes": "RTA/Bus/bus_routes.csv",
        "bus_stop_details": "RTA/Bus/bus_stop_details.csv",
        "tram_stations": "RTA/Tram/tram_stations.csv",
        # KHDA - Education
        "school_search": "KHDA/Schools/school_search.csv",
        "dubai_private_schools": "KHDA/Registers/dubai_private_schools.csv",
        # DHA - Health
        "sheryan_facility_detail": "DHA/Location/sheryan_facility_detail.csv",
        "sheryan_professional_detail": "DHA/Licenses/sheryan_professional_detail.csv",
        # DLD - Valuations
        "valuation": "DLD/Valuations/Valuation.csv",
        "real_estate_licenses": "DLD/Licenses/Real_Estate_Licenses.csv",
    }

    for table_name, csv_path in csv_mapping.items():
        full_path = DATA_DIR / csv_path
        if full_path.exists():
            try:
                df = pd.read_csv(full_path)
                df.to_sql(table_name, DB_CONNECTION, if_exists="replace", index=False)
                print(f"Loaded {table_name}: {len(df)} rows")
            except Exception as e:
                print(f"Warning: Could not load {table_name}: {e}")
        else:
            print(f"Warning: File not found - {csv_path}")


def get_db():
    """Get database connection."""
    if DB_CONNECTION is None:
        init_database()
    return DB_CONNECTION


# ============================================================================
# Pydantic Models
# ============================================================================

class FilterItem(BaseModel):
    table: str
    column: str
    operator: str
    value: Any
    confidence: float = 1.0


class SuggestFiltersRequest(BaseModel):
    query: str


class SuggestFiltersResponse(BaseModel):
    """Enhanced response with Netflix-style features."""
    detected_tables: List[str]
    extracted_filters: List[Dict]
    order_by: Optional[Dict] = None
    extraction_method: str
    confidence: float
    interpretation: Optional[str] = None
    chips: List[Dict] = []
    entities: List[Dict] = []
    validation: Optional[Dict] = None
    warnings: List[str] = []


class GenerateSQLRequest(BaseModel):
    tables: List[str]
    filters: List[FilterItem] = []
    order_by: Optional[Dict[str, str]] = None


class ExecuteQueryRequest(BaseModel):
    sql: str
    limit: int = 100


class UpdateChipRequest(BaseModel):
    chip_id: str
    new_value: Optional[Any] = None
    new_operator: Optional[str] = None
    remove: bool = False


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/init")
async def init():
    """Return schema information for the UI."""
    return {
        "tables": SCHEMA,
        "edges": SCHEMA_EDGES,
        "features": {
            "llm_enabled": gemini_model is not None,
            "entity_resolution": True,
            "chip_ui": True,
            "validation": True,
            "caching": True,
        }
    }


@app.post("/api/suggest_filters", response_model=SuggestFiltersResponse)
async def suggest_filters(request: SuggestFiltersRequest):
    """
    Netflix-inspired intelligent filter extraction.

    Features:
    - Multi-strategy extraction (LLM, Hybrid, Regex)
    - Entity resolution with @mentions support
    - AST-based filter validation
    - UI chip generation
    """
    query = request.query
    start_time = time.time()

    if not smart_extractor:
        raise HTTPException(status_code=500, detail="Extractor not initialized")

    # Bust cache for this query to avoid stale table mappings
    if extraction_cache:
        extraction_cache.invalidate(query)

    # Extract using Netflix-inspired smart extractor
    result = smart_extractor.extract_sync(query)

    # Generate UI chips
    chips = chip_generator.generate(result)

    # Record metrics
    latency_ms = (time.time() - start_time) * 1000
    metrics_collector.record_extraction(query, result, latency_ms)

    # Build response
    response = SuggestFiltersResponse(
        detected_tables=result.ast.tables,
        extracted_filters=[node.to_dict() for node in result.ast.filters.flatten()],
        order_by=result.ast.order_by.to_dict() if result.ast.order_by else None,
        extraction_method=result.strategy_used.value,
        confidence=result.confidence,
        interpretation=result.llm_interpretation,
        chips=[chip.to_dict() for chip in chips],
        entities=[e.to_dict() for e in result.entities_resolved],
        validation=result.validation_result.to_dict() if result.validation_result else None,
        warnings=result.warnings,
    )

    _append_analysis_log(query, response)

    return response


@app.get("/api/entity/suggest")
async def suggest_entities(
    q: str = Query(..., min_length=1, description="Partial entity text to search"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    limit: int = Query(10, ge=1, le=50),
):
    """
    Netflix-style @mention autocomplete.

    Returns suggestions for entity completion in the UI.
    """
    if not entity_resolver:
        raise HTTPException(status_code=500, detail="Entity resolver not initialized")

    from llm_layer.entity_resolver import EntityType

    # Parse entity type if provided
    etype = None
    if entity_type:
        try:
            etype = EntityType(entity_type)
        except ValueError:
            pass

    suggestions = entity_resolver.suggest_completions(q, entity_type=etype, limit=limit)

    return {
        "query": q,
        "suggestions": suggestions,
    }


@app.get("/api/entity/vocabulary/{entity_type}")
async def get_vocabulary(entity_type: str):
    """Get full vocabulary for an entity type."""
    if not entity_resolver:
        raise HTTPException(status_code=500, detail="Entity resolver not initialized")

    from llm_layer.entity_resolver import EntityType

    try:
        etype = EntityType(entity_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown entity type: {entity_type}")

    vocab = entity_resolver.get_vocabulary(etype)
    return {
        "entity_type": entity_type,
        "values": vocab,
        "count": len(vocab),
    }


@app.post("/api/generate_sql")
async def generate_sql(request: GenerateSQLRequest):
    """Generate SQL query from tables and filters with validation."""
    if not request.tables and not request.filters:
        return {"sql": "", "validation": None}

    remapped_filters = _remap_fact_filters(request.filters)
    normalized_filters = _normalize_filter_items(remapped_filters)
    tables = _merge_tables_and_filters(request.tables, normalized_filters)

    # Build FilterAST from request
    filter_group = FilterGroup(LogicalOperator.AND)
    for f in normalized_filters:
        node = FilterNode(
            table=f.table,
            column=f.column,
            operator=FilterOperator.from_string(f.operator),
            value=f.value,
        )
        filter_group.add(node)

    from llm_layer.filter_dsl import OrderByClause
    order_by = None
    if request.order_by:
        order_by = OrderByClause(
            column=request.order_by.get("column", ""),
            direction=request.order_by.get("direction", "ASC"),
        )

    ast = FilterAST(
        tables=tables,
        filters=filter_group,
        order_by=order_by,
    )

    # Validate
    registry = SchemaRegistry.from_schema_definition(SCHEMA)
    validator = FilterValidator(registry)
    validation = validator.validate(ast)

    # Generate SQL
    sql = _generate_sql_from_ast(ast)

    return {
        "sql": sql,
        "validation": validation.to_dict(),
    }


def _generate_sql_from_ast(ast: FilterAST) -> str:
    """Generate SQL from FilterAST."""
    if not ast.tables:
        return ""

    # Build SELECT clause
    select_cols = []
    for table in ast.tables:
        table_schema = next((t for t in SCHEMA if t["name"] == table), None)
        if table_schema:
            cols = [c["name"] for c in table_schema["columns"][:6]]
            select_cols.extend([f"{table}.{col}" for col in cols])

    # Ensure WHERE columns are included in SELECT (and prioritized)
    where_cols = []
    for node in ast.filters.flatten():
        if node.table and node.column:
            where_cols.append(f"{node.table}.{node.column}")

    if not select_cols:
        select_cols = ["*"]

    final_select = _prioritized_select_list(where_cols, select_cols, limit=12)
    sql = f"SELECT {', '.join(_unique_select_list(final_select))}\n"

    # Build FROM clause with JOINs
    base_table = ast.tables[0]
    sql += f"FROM {base_table}\n"

    joined = {base_table}
    join_lines = _build_join_clauses(base_table, ast.tables[1:])
    if join_lines:
        sql += "\n".join(join_lines) + "\n"
        joined.update(_tables_from_join_lines(join_lines))

    # Build WHERE clause from AST
    if not ast.filters.is_empty():
        sql += f"WHERE {ast.filters.to_sql()}\n"

    # Add ORDER BY
    if ast.order_by:
        sql += f"ORDER BY {ast.order_by.to_sql()}\n"

    sql += "LIMIT 100"

    return sql


def _find_join_condition(from_table: str, to_table: str) -> Optional[str]:
    """Find join condition between two tables."""
    join_map = _join_condition_map()
    return join_map.get((from_table, to_table)) or join_map.get((to_table, from_table))


def _append_analysis_log(query: str, response: SuggestFiltersResponse) -> None:
    """Append per-query analysis to local JSONL file."""
    try:
        analysis = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "query": query,
            "detected_tables": response.detected_tables,
            "filters": response.extracted_filters,
            "order_by": response.order_by,
            "strategy": response.extraction_method,
            "confidence": response.confidence,
            "warnings": response.warnings,
            "validation": response.validation,
            "is_true": bool(response.validation and response.validation.get("valid")),
        }
        ANALYSIS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ANALYSIS_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(pyjson.dumps(analysis, ensure_ascii=True) + "\n")
    except Exception:
        # Avoid breaking the request if logging fails
        pass


def _unique_select_list(select_cols: List[str]) -> List[str]:
    """Alias select columns to avoid duplicate names."""
    seen = set()
    aliased = []
    for col in select_cols:
        if col == "*":
            aliased.append(col)
            continue
        base = col.split(".")[-1]
        if base in seen:
            alias = col.replace(".", "_")
            aliased.append(f"{col} AS {alias}")
        else:
            aliased.append(col)
        seen.add(base)
    return aliased


def _prioritized_select_list(where_cols: List[str], select_cols: List[str], limit: int = 12) -> List[str]:
    """Build a select list that always includes WHERE columns."""
    if "*" in select_cols:
        return select_cols

    seen = set()
    ordered: List[str] = []

    for col in where_cols + select_cols:
        if col in seen:
            continue
        ordered.append(col)
        seen.add(col)

    if limit and len(ordered) > limit:
        # Keep all WHERE columns, then truncate the rest
        where_set = {c for c in where_cols}
        must_keep = [c for c in ordered if c in where_set]
        remaining = [c for c in ordered if c not in where_set]
        ordered = must_keep + remaining[: max(0, limit - len(must_keep))]

    return ordered


def _merge_tables_and_filters(tables: List[str], filters: List[FilterItem]) -> List[str]:
    """Ensure all filter tables are included and deduped."""
    table_list = [t for t in tables if t]
    table_set = {t for t in table_list}
    for f in filters:
        if f.table and f.table not in table_set:
            table_set.add(f.table)
            table_list.append(f.table)
    if "fact_listings" in table_set:
        table_list = ["fact_listings"] + [t for t in table_list if t != "fact_listings"]
    return table_list


def _normalize_filter_items(filters: List[FilterItem]) -> List[FilterItem]:
    """Deduplicate and merge compatible filters (e.g., EQ -> IN)."""
    if not filters:
        return []

    seen = set()
    eq_groups: Dict[tuple, Dict[str, Any]] = {}
    passthrough: List[FilterItem] = []

    def _normalize_value(v):
        if isinstance(v, str):
            return v.strip().lower()
        return v

    for f in filters:
        op = (f.operator or "=").upper()
        table = (f.table or "").strip()
        column = (f.column or "").strip()
        if not table or not column:
            continue

        if op in {"=", "IN"}:
            key = (table.lower(), column.lower())
            group = eq_groups.setdefault(
                key,
                {
                    "table": table,
                    "column": column,
                    "values": [],
                    "norms": set(),
                    "confidence": f.confidence,
                    "has_in": op == "IN",
                },
            )
            values = f.value if isinstance(f.value, list) else [f.value]
            for v in values:
                norm = _normalize_value(v)
                if norm not in group["norms"]:
                    group["values"].append(v)
                    group["norms"].add(norm)
            if op == "IN":
                group["has_in"] = True
            continue

        key = (table.lower(), column.lower(), op, _normalize_value(f.value))
        if key in seen:
            continue
        seen.add(key)
        passthrough.append(f)

    normalized: List[FilterItem] = []
    for group in eq_groups.values():
        values = group["values"]
        if len(values) == 1 and not group["has_in"]:
            normalized.append(FilterItem(
                table=group["table"],
                column=group["column"],
                operator="=",
                value=values[0],
                confidence=group["confidence"],
            ))
        else:
            normalized.append(FilterItem(
                table=group["table"],
                column=group["column"],
                operator="IN",
                value=values,
                confidence=group["confidence"],
            ))

    normalized.extend(passthrough)
    return normalized


def _remap_fact_filters(filters: List[FilterItem]) -> List[FilterItem]:
    """Remap transaction entity filters to lookup tables for joins."""
    if not filters:
        return []

    mapping = {
        # Remap area filters to lkp_areas
        "area_name_en": ("lkp_areas", "name_en"),
        "area_id": ("lkp_areas", "area_id"),
        # Remap transaction group filters
        "trans_group_en": ("lkp_transaction_groups", "name_en"),
        "trans_group_id": ("lkp_transaction_groups", "group_id"),
    }

    remapped: List[FilterItem] = []
    for f in filters:
        table = (f.table or "").strip()
        column = (f.column or "").strip()
        if table == "transactions" and column in mapping:
            new_table, new_column = mapping[column]
            remapped.append(FilterItem(
                table=new_table,
                column=new_column,
                operator=f.operator,
                value=f.value,
                confidence=f.confidence,
            ))
        else:
            remapped.append(f)
    return remapped


def _join_condition_map() -> Dict[tuple, str]:
    return {
        # ==========================================
        # DLD Transactions joins
        # ==========================================
        ("transactions", "lkp_areas"): "transactions.area_id = lkp_areas.area_id",
        ("transactions", "lkp_transaction_groups"): "transactions.trans_group_id = lkp_transaction_groups.group_id",
        ("transactions", "lkp_transaction_procedures"): "transactions.procedure_id = lkp_transaction_procedures.procedure_id",
        ("transactions", "developers"): "transactions.project_name_en = developers.developer_name_en",
        ("transactions", "units"): "transactions.area_id = units.area_id AND transactions.building_name_en = units.project_name_en",

        # ==========================================
        # DLD Units joins
        # ==========================================
        ("units", "lkp_areas"): "units.area_id = lkp_areas.area_id",
        ("units", "developers"): "units.project_name_en = developers.developer_name_en",
        ("units", "land_registry"): "units.area_id = land_registry.area_id",

        # ==========================================
        # DLD Land Registry joins
        # ==========================================
        ("land_registry", "lkp_areas"): "land_registry.area_id = lkp_areas.area_id",
        ("land_registry", "developers"): "land_registry.project_name_en = developers.developer_name_en",

        # ==========================================
        # DLD Rent Contracts joins
        # ==========================================
        ("rent_contracts", "lkp_areas"): "rent_contracts.area_id = lkp_areas.area_id",
        ("rent_contracts", "units"): "rent_contracts.area_id = units.area_id",
        ("rent_contracts", "developers"): "rent_contracts.project_name_en = developers.developer_name_en",

        # ==========================================
        # Lookup table relationships
        # ==========================================
        ("lkp_transaction_procedures", "lkp_transaction_groups"): "lkp_transaction_procedures.group_id = lkp_transaction_groups.group_id",
        ("lkp_transaction_groups", "lkp_transaction_procedures"): "lkp_transaction_groups.group_id = lkp_transaction_procedures.group_id",

        # ==========================================
        # Bayut joins
        # ==========================================
        ("bayut_transactions", "lkp_areas"): "bayut_transactions.location_area = lkp_areas.name_en",
        ("bayut_transactions", "developers"): "bayut_transactions.developer_name = developers.developer_name_en",

        # ==========================================
        # Developers & Brokers joins
        # ==========================================
        ("developers", "lkp_areas"): "1=1",  # Developers don't have direct area link
        ("brokers", "developers"): "brokers.real_estate_id = developers.developer_id",

        # ==========================================
        # Valuation joins
        # ==========================================
        ("valuation", "lkp_areas"): "valuation.area_id = lkp_areas.area_id",
        ("valuation", "transactions"): "valuation.area_id = transactions.area_id",

        # ==========================================
        # Real Estate Licenses joins
        # ==========================================
        ("real_estate_licenses", "developers"): "real_estate_licenses.license_number = developers.license_number",
        ("real_estate_licenses", "lkp_areas"): "1=1",  # No direct link

        # ==========================================
        # Transport (RTA) joins
        # ==========================================
        ("metro_stations", "transactions"): "transactions.nearest_metro_en = metro_stations.station_name",
        ("metro_stations", "bayut_transactions"): "1=1",  # Would need geo-spatial join

        # ==========================================
        # Education (KHDA) joins
        # ==========================================
        ("school_search", "lkp_areas"): "school_search.area = lkp_areas.name_en",
        ("school_search", "units"): "school_search.area = units.area_name_en",

        # ==========================================
        # Health (DHA) joins
        # ==========================================
        ("sheryan_facility_detail", "lkp_areas"): "sheryan_facility_detail.area = lkp_areas.name_en",
    }


def _build_join_clauses(base_table: str, target_tables: List[str]) -> List[str]:
    """Build JOIN clauses, including multi-hop paths."""
    if not target_tables:
        return []

    graph = _schema_graph()
    join_map = _join_condition_map()
    join_lines: List[str] = []
    joined = {base_table}

    for target in target_tables:
        if target in joined:
            continue
        path = _find_join_path(graph, base_table, target)
        if not path:
            continue
        for i in range(len(path) - 1):
            left = path[i]
            right = path[i + 1]
            if right in joined:
                continue
            join_condition = join_map.get((left, right)) or join_map.get((right, left))
            if not join_condition:
                break
            join_lines.append(f"INNER JOIN {right} ON {join_condition}")
            joined.add(right)

    return join_lines


def _schema_graph() -> Dict[str, List[str]]:
    graph: Dict[str, List[str]] = {}
    for edge in SCHEMA_EDGES:
        a = edge["from"]
        b = edge["to"]
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
    return graph


def _find_join_path(graph: Dict[str, List[str]], start: str, goal: str) -> Optional[List[str]]:
    if start == goal:
        return [start]
    queue = [(start, [start])]
    visited = {start}
    while queue:
        current, path = queue.pop(0)
        for neighbor in graph.get(current, []):
            if neighbor in visited:
                continue
            new_path = path + [neighbor]
            if neighbor == goal:
                return new_path
            visited.add(neighbor)
            queue.append((neighbor, new_path))
    return None


def _tables_from_join_lines(join_lines: List[str]) -> List[str]:
    tables = []
    for line in join_lines:
        parts = line.split()
        if len(parts) >= 3:
            tables.append(parts[2])
    return tables


@app.post("/api/query")
async def execute_query(request: ExecuteQueryRequest):
    """Execute SQL query and return results."""
    db = get_db()

    try:
        sql = request.sql
        if "LIMIT" not in sql.upper():
            sql += f" LIMIT {request.limit}"

        df = pd.read_sql_query(sql, db)
        data_json = json.loads(df.to_json(orient="records"))

        return {
            "columns": list(df.columns),
            "data": data_json,
            "row_count": len(df),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get database statistics - Dubai Proptech Data."""
    db = get_db()
    cursor = db.cursor()

    stats = {}

    for table_info in SCHEMA:
        table = table_info["name"]
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]
        except:
            stats[table] = 0

    # DLD Transactions stats
    try:
        cursor.execute("SELECT AVG(actual_worth) FROM transactions WHERE actual_worth > 0")
        avg_transaction_value = cursor.fetchone()[0]
    except:
        avg_transaction_value = 0

    try:
        cursor.execute("SELECT COUNT(DISTINCT area_name_en) FROM transactions")
        area_count = cursor.fetchone()[0]
    except:
        area_count = 0

    try:
        cursor.execute("SELECT COUNT(DISTINCT trans_group_en) FROM transactions")
        transaction_types = cursor.fetchone()[0]
    except:
        transaction_types = 0

    # Bayut stats
    try:
        cursor.execute("SELECT AVG(transaction_amount) FROM bayut_transactions WHERE transaction_amount > 0")
        avg_bayut_price = cursor.fetchone()[0]
    except:
        avg_bayut_price = 0

    return {
        "table_counts": stats,
        "avg_transaction_value": round(avg_transaction_value, 2) if avg_transaction_value else 0,
        "avg_bayut_price": round(avg_bayut_price, 2) if avg_bayut_price else 0,
        "unique_areas": area_count,
        "transaction_types": transaction_types,
    }


@app.get("/api/sample/{table_name}")
async def get_sample(table_name: str, limit: int = 10):
    """Get sample data from a table."""
    db = get_db()

    valid_tables = [t["name"] for t in SCHEMA]
    if table_name not in valid_tables:
        raise HTTPException(status_code=404, detail=f"Table {table_name} not found")

    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", db)
        return {
            "table": table_name,
            "columns": list(df.columns),
            "data": df.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/column_values/{table_name}/{column_name}")
async def get_column_values(table_name: str, column_name: str, limit: int = 100):
    """Get distinct values for a categorical column (for dropdown filters)."""
    db = get_db()

    valid_tables = [t["name"] for t in SCHEMA]
    if table_name not in valid_tables:
        raise HTTPException(status_code=404, detail=f"Table {table_name} not found")

    try:
        # Get distinct values with counts
        query = f"""
            SELECT {column_name} as value, COUNT(*) as count
            FROM {table_name}
            WHERE {column_name} IS NOT NULL AND {column_name} != ''
            GROUP BY {column_name}
            ORDER BY count DESC
            LIMIT {limit}
        """
        df = pd.read_sql_query(query, db)
        return {
            "table": table_name,
            "column": column_name,
            "values": df.to_dict(orient="records"),
            "total_distinct": len(df),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/categorical_columns")
async def get_categorical_columns():
    """Get all categorical columns with their distinct values for filter dropdowns."""
    db = get_db()

    # Define which columns are categorical (good for dropdowns)
    categorical_columns = {
        "transactions": ["property_type_en", "trans_group_en", "area_name_en", "property_usage_en", "rooms_en"],
        "rent_contracts": ["ejari_property_type_en", "property_usage_en", "area_name_en", "tenant_type_en"],
        "units": ["property_type_en", "property_sub_type_en", "area_name_en", "rooms_en"],
        "bayut_transactions": ["completion_status", "property_type_id", "location_area", "seller_type", "payment_type"],
        "lkp_areas": ["name_en"],
        "lkp_transaction_groups": ["name_en"],
        "lkp_transaction_procedures": ["name_en"],
        "developers": ["developer_name_en", "license_source_en"],
    }

    result = {}
    cursor = db.cursor()

    for table, columns in categorical_columns.items():
        result[table] = {}
        for column in columns:
            try:
                cursor.execute(f"""
                    SELECT DISTINCT {column}
                    FROM {table}
                    WHERE {column} IS NOT NULL AND {column} != ''
                    ORDER BY {column}
                    LIMIT 100
                """)
                values = [row[0] for row in cursor.fetchall()]
                if values:
                    result[table][column] = values
            except:
                pass

    return result


# ============================================================================
# Metrics & Monitoring Endpoints
# ============================================================================

@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """Get extraction metrics summary."""
    return metrics_collector.get_summary()


@app.get("/api/metrics/dashboard")
async def get_metrics_dashboard():
    """Get full dashboard data for monitoring."""
    return metrics_collector.get_dashboard_data()


@app.get("/api/metrics/strategies")
async def get_strategy_analysis():
    """Analyze extraction strategy effectiveness."""
    return metrics_collector.get_strategy_analysis()


@app.get("/api/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    return extraction_cache.get_stats()


@app.post("/api/cache/clear")
async def clear_cache():
    """Clear the extraction cache."""
    extraction_cache.clear()
    return {"status": "cleared"}


# ============================================================================
# Health & Static Files
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "llm_available": gemini_model is not None,
        "db_connected": DB_CONNECTION is not None,
        "cache_size": len(extraction_cache.cache),
    }


@app.get("/")
async def read_root():
    """Serve the main HTML file."""
    return FileResponse(Path(__file__).parent / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
