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
import re
import time
import json
import sqlite3
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Any
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timedelta
import json as pyjson
import statistics

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

# Import Intelligence System
from intelligence import (
    MetricTracker,
    CausalExplainer,
    BriefingGenerator,
    create_intelligence_system,
)

# Configure Gemini API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Use the first available model — skip the slow test call at startup
    _model_name = "gemini-2.0-flash"
    gemini_model = genai.GenerativeModel(_model_name)
    print(f"Configured Gemini model: {_model_name} (will validate on first use)")
else:
    gemini_model = None
    print("Warning: GOOGLE_API_KEY not set. LLM extraction will be disabled.")

# Global components
extraction_cache = ExtractionCache(max_size=1000, ttl_seconds=3600)
metrics_collector = MetricsCollector(max_log_size=10000)
smart_extractor = None  # Initialized after DB is ready
entity_resolver = None
chip_generator = ChipGenerator()

# Intelligence system components
intelligence_tracker = None
intelligence_explainer = None
intelligence_generator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and components on startup."""
    global smart_extractor, entity_resolver
    global intelligence_tracker, intelligence_explainer, intelligence_generator

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

    # Initialize Intelligence System
    intelligence_tracker = MetricTracker(get_db())
    intelligence_explainer = CausalExplainer(gemini_model, get_db())
    intelligence_generator = BriefingGenerator(
        gemini_model, intelligence_tracker, intelligence_explainer
    )

    print("Netflix-inspired LLM Layer initialized!")
    print("Intelligence System initialized!")
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
# DATA_MODE: "variable" (curated 1.8M rows), "demo" (12MB), "full" (17GB)
DATA_MODE = os.environ.get("DATA_MODE", "variable")
if DATA_MODE == "variable":
    DATA_DIR = Path(__file__).parent / "data_variable"
    print("Using VARIABLE data (1.8M rows, all categories) - set DATA_MODE=full for complete dataset")
elif DATA_MODE == "demo":
    DATA_DIR = Path(__file__).parent / "data_demo"
    print("Using DEMO data (12MB) - set DATA_MODE=full for complete dataset")
else:
    DATA_DIR = Path(__file__).parent / "data"
    print("Using FULL data (17GB)")
ANALYSIS_LOG_PATH = Path(__file__).parent / "feedback" / "analysis_log.jsonl"

# Pre-built SQLite DB file (built by build_db.py) — skips CSV loading on deploy
PREBUILT_DB = Path(__file__).parent / "production.db"

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
            {"name": "location_id", "type": "INTEGER", "is_pk": True},
            {"name": "zone_id", "type": "INTEGER"},
            {"name": "location_name_english", "type": "TEXT"},
            {"name": "location_name_arabic", "type": "TEXT"},
            {"name": "line_name", "type": "TEXT"},
            {"name": "station_location_longitude", "type": "REAL"},
            {"name": "station_location_latitude", "type": "REAL"},
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
        "description": "KHDA Dubai schools directory with ratings and locations",
        "columns": [
            {"name": "education_center_id", "type": "INTEGER", "is_pk": True},
            {"name": "name_eng", "type": "TEXT", "description": "School name in English"},
            {"name": "lat", "type": "REAL", "description": "Latitude"},
            {"name": "long", "type": "REAL", "description": "Longitude"},
            {"name": "overall_performance_en", "type": "TEXT", "description": "KHDA rating: Outstanding, Very good, Good, Acceptable, Weak"},
            {"name": "curriculum_en", "type": "TEXT", "description": "Curriculum e.g. British, American, IB, Indian, etc."},
            {"name": "area_en", "type": "TEXT", "description": "Area name e.g. AL BARSHA FIRST, JUMEIRA FIRST"},
            {"name": "inspection_year", "type": "TEXT"},
            {"name": "student_count", "type": "INTEGER"},
            {"name": "established_on", "type": "TEXT"},
        ]
    },
    # ==========================================
    # DHA - HEALTH
    # ==========================================
    {
        "name": "sheryan_facility_detail",
        "description": "DHA licensed healthcare facilities (hospitals, clinics, pharmacies)",
        "columns": [
            {"name": "unique_id", "type": "TEXT", "is_pk": True},
            {"name": "f_name_english", "type": "TEXT", "description": "Facility name"},
            {"name": "facility_category_name_english", "type": "TEXT", "description": "Category e.g. General Hospital, Clinic, Pharmacy, Laboratory"},
            {"name": "facilitysubcategorynameenglish", "type": "TEXT", "description": "Subcategory e.g. General Hospital (>100), Specialist Clinic"},
            {"name": "area_english", "type": "TEXT", "description": "Area name e.g. AL BARSHA FIRST"},
            {"name": "x_coordinate", "type": "REAL", "description": "Latitude"},
            {"name": "y_coordinate", "type": "REAL", "description": "Longitude"},
            {"name": "status", "type": "TEXT"},
            {"name": "telephone_1", "type": "TEXT"},
            {"name": "website", "type": "TEXT"},
        ]
    },

    # ==========================================
    # RTA - TRANSPORT
    # ==========================================
    {
        "name": "bus_stop_details",
        "description": "RTA bus stop locations, routes, and facilities across Dubai",
        "columns": [
            {"name": "stop_name", "type": "TEXT", "description": "Bus stop name"},
            {"name": "stop_id", "type": "REAL", "is_pk": True},
            {"name": "street_name", "type": "TEXT"},
            {"name": "route_name", "type": "TEXT", "description": "Bus route number"},
            {"name": "stop_location_latitude", "type": "REAL"},
            {"name": "stop_location_longitude", "type": "REAL"},
            {"name": "bus_stop_type", "type": "TEXT"},
        ]
    },
    {
        "name": "tram_stations",
        "description": "Dubai Tram station locations and details",
        "columns": [
            {"name": "location_name_english", "type": "TEXT", "description": "Station name"},
            {"name": "station_location_latitude", "type": "REAL"},
            {"name": "station_location_longitude", "type": "REAL"},
            {"name": "line_name", "type": "TEXT"},
        ]
    },
]

# Schema relationships for graph visualization - Dubai Proptech Model
SCHEMA_EDGES = [
    # ==========================================
    # DLD Core: Transactions & Registrations
    # ==========================================
    {"from": "transactions", "to": "lkp_areas"},
    {"from": "transactions", "to": "lkp_transaction_groups"},
    {"from": "transactions", "to": "lkp_transaction_procedures"},
    {"from": "transactions", "to": "developers"},
    {"from": "transactions", "to": "units"},
    {"from": "transactions", "to": "buildings"},
    {"from": "units", "to": "lkp_areas"},
    {"from": "units", "to": "developers"},
    {"from": "units", "to": "land_registry"},
    {"from": "land_registry", "to": "lkp_areas"},
    {"from": "land_registry", "to": "developers"},
    {"from": "rent_contracts", "to": "lkp_areas"},
    {"from": "rent_contracts", "to": "units"},
    {"from": "buildings", "to": "lkp_areas"},
    {"from": "buildings", "to": "projects"},

    # DLD Lookups
    {"from": "lkp_transaction_procedures", "to": "lkp_transaction_groups"},

    # DLD Stakeholders & Valuations
    {"from": "brokers", "to": "developers"},
    {"from": "valuation", "to": "lkp_areas"},
    {"from": "valuation", "to": "transactions"},
    {"from": "real_estate_licenses", "to": "developers"},
    {"from": "real_estate_permits", "to": "real_estate_licenses"},
    {"from": "oa_service_charges", "to": "projects"},
    {"from": "map_requests", "to": "lkp_areas"},

    # ==========================================
    # Bayut Market Data
    # ==========================================
    {"from": "bayut_transactions", "to": "lkp_areas"},
    {"from": "bayut_transactions", "to": "developers"},
    {"from": "bayut_new_projects", "to": "developers"},
    {"from": "bayut_commercial_transactions", "to": "lkp_areas"},

    # ==========================================
    # DED — Business Licenses & Commerce
    # ==========================================
    {"from": "license_master", "to": "business_activities"},
    {"from": "initial_approval", "to": "license_master"},
    {"from": "inspection_report", "to": "license_master"},
    {"from": "permits", "to": "license_master"},
    {"from": "commerce_registry", "to": "license_master"},
    {"from": "trade_name", "to": "license_master"},

    # ==========================================
    # DEWA — Utilities
    # ==========================================
    {"from": "customers_master_data", "to": "lkp_areas"},
    {"from": "ev_green_charger", "to": "lkp_areas"},

    # ==========================================
    # DHA — Healthcare
    # ==========================================
    {"from": "sheryan_professional_detail", "to": "sheryan_facility_detail"},
    {"from": "sheryan_facility_detail", "to": "lkp_areas"},

    # ==========================================
    # KHDA — Education
    # ==========================================
    {"from": "school_search", "to": "lkp_areas"},
    {"from": "hep_programs", "to": "hep_search"},
    {"from": "ti_educationcentercourses", "to": "ti_search"},
    {"from": "inspection_grade_range", "to": "school_search"},

    # ==========================================
    # Municipality — Urban
    # ==========================================
    {"from": "building_permits", "to": "lkp_areas"},
    {"from": "building_summary_information", "to": "lkp_areas"},
    {"from": "building_permits", "to": "consultant_projects"},
    {"from": "food_business_information", "to": "license_master"},
    {"from": "food_health_certificate", "to": "consignments"},
    {"from": "inprogress_applications_sla", "to": "building_permits"},

    # ==========================================
    # RTA — Transport
    # ==========================================
    {"from": "metro_ridership", "to": "metro_stations"},
    {"from": "bus_ridership", "to": "bus_routes"},
    {"from": "bus_stop_details", "to": "bus_routes"},
    {"from": "tram_ridership", "to": "tram_stations"},
    {"from": "marine_ridership", "to": "marine_stations"},
    {"from": "public_transportation_routes_stops", "to": "metro_stations"},

    # ==========================================
    # Cross-domain relationships
    # ==========================================
    {"from": "transactions", "to": "metro_stations"},
    {"from": "bayut_transactions", "to": "metro_stations"},
    {"from": "transactions", "to": "school_search"},
    {"from": "units", "to": "school_search"},
    {"from": "bayut_transactions", "to": "school_search"},
    {"from": "transactions", "to": "sheryan_facility_detail"},
    {"from": "units", "to": "sheryan_facility_detail"},
    {"from": "bayut_transactions", "to": "sheryan_facility_detail"},
    {"from": "transactions", "to": "tram_stations"},
    {"from": "bayut_transactions", "to": "tram_stations"},
    {"from": "building_permits", "to": "buildings"},

    # DSC — Statistics (standalone reference)
    {"from": "residential_property_index", "to": "residential_sale_index"},
]


def _csv_to_table_name(csv_path: str) -> str:
    """Convert CSV file path to a clean SQL table name.

    Examples:
      DLD/Transactions/Transactions.csv -> transactions
      DLD/Registrations/Rent_Contracts.csv -> rent_contracts
      RTA/Rail/metro_ridership_2026-01-14_00-00-00.csv -> metro_ridership
      DSC/Statistics/commercial_property_index_2025-06-23_08-58-30.csv -> commercial_property_index
      DEWA/Consumption/Annual_Statistics_2024-12-31_00-00-00.csv -> annual_statistics_2024
    """
    import re
    name = Path(csv_path).stem  # Remove .csv
    # Remove timestamp suffixes like _2026-01-14_00-00-00 or _2025-06-23_08-58-30
    name = re.sub(r'_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$', '', name)
    # But keep year suffixes like _2024 (useful for multi-year data)
    # Clean up special chars
    name = name.lower().replace('-', '_').replace(' ', '_')
    # Remove double underscores
    name = re.sub(r'_+', '_', name).strip('_')
    return name


def _infer_sql_type(series: pd.Series) -> str:
    """Infer SQL column type from a pandas Series."""
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(series):
        return "REAL"
    else:
        return "TEXT"


# Category descriptions for auto-generated schema
CATEGORY_DESCRIPTIONS = {
    "DLD": "Dubai Land Department",
    "DED": "Department of Economic Development",
    "DEWA": "Dubai Electricity & Water Authority",
    "DHA": "Dubai Health Authority",
    "DSC": "Dubai Statistics Center",
    "KHDA": "Knowledge & Human Development Authority",
    "Municipality": "Dubai Municipality",
    "RTA": "Roads & Transport Authority",
    "Bayut": "Bayut real estate market data",
}

# Friendly table descriptions (for well-known tables)
TABLE_DESCRIPTIONS = {
    "transactions": "DLD property transactions (sales, mortgages, gifts) - 200K records across all Dubai areas",
    "rent_contracts": "Ejari rental contracts with amounts, property types, areas - 150K records",
    "units": "Registered property units (apartments, villas, offices) - 80K records",
    "land_registry": "Dubai land parcels with ownership and area info - 50K records",
    "buildings": "DLD registered buildings with type, floors, car parks - 50K records",
    "valuation": "Property valuations with appraised values - 88K records",
    "map_requests": "DLD map/service requests - 30K records",
    "developers": "All registered real estate developers in Dubai",
    "brokers": "Licensed real estate brokers",
    "projects": "Development projects (completion %, units, buildings)",
    "offices": "Real estate company offices",
    "lkp_areas": "All 300+ Dubai areas/districts lookup",
    "lkp_transaction_groups": "Transaction groups (Sales, Mortgages, Gifts)",
    "lkp_transaction_procedures": "Transaction procedure types",
    "lkp_market_types": "Market type lookup",
    "residential_sale_index": "Dubai residential property price index (monthly)",
    "real_estate_licenses": "Real estate company licenses",
    "real_estate_permits": "Real estate activity permits - 30K records",
    "oa_service_charges": "Owner association service charges by community - 91K records",
    "bayut_transactions": "Bayut transactions with prices, yields, lat/lon - 20K records",
    "bayut_commercial_transactions": "Bayut commercial property transactions - 10K records",
    "bayut_new_projects": "New development projects from Bayut - 2K records",
    "license_master": "DED business licenses in Dubai - 50K records",
    "business_activities": "DED business activity types catalog",
    "initial_approval": "DED business initial approvals - 30K records",
    "inspection_report": "DED business inspection reports - 30K records",
    "permits": "DED business permits - 30K records",
    "commerce_registry": "Dubai commerce registry entries - 30K records",
    "trade_name": "Registered trade names in Dubai - 30K records",
    "customers_master_data": "DEWA utility customers by area, nationality - 50K records",
    "ev_green_charger": "EV charging station locations in Dubai",
    "annual_statistics_2024": "DEWA annual electricity/water statistics 2024",
    "gross_power_generation_2021": "Monthly power generation by station",
    "water_production_2022": "Monthly water production by station",
    "water_supply_mig": "Monthly water supply data",
    "sheryan_professional_detail": "Healthcare professionals (doctors, nurses) - 30K records",
    "sheryan_facility_detail": "Healthcare facilities (hospitals, clinics) - 10K records",
    "birth_notification": "Birth records by facility and demographics",
    "consumer_price_index": "Dubai CPI by expenditure groups (monthly)",
    "gdp_quarterly": "Dubai GDP by economic sector (quarterly)",
    "gross_domestic_product_at_current_prices": "Annual GDP at current prices by sector",
    "gross_domestic_product_at_constant_prices": "Annual GDP at constant prices by sector",
    "inflation_rate": "Dubai monthly/annual inflation rate",
    "commercial_property_index": "Commercial property price index (quarterly)",
    "residential_property_index": "Residential property price index (quarterly)",
    "buildings_2019": "Dubai building census by type (2019)",
    "employment_2019": "Employment data by nationality, gender, age",
    "unemployment_2019": "Unemployment data by nationality, gender, age",
    "school_search": "KHDA school directory with ratings, curriculum, area",
    "dubai_private_schools": "Dubai private schools list",
    "hep_search": "Higher education institutions",
    "hep_programs": "University programs with fees and degrees",
    "ti_search": "Training institutes directory",
    "ti_educationcentercourses": "Training courses catalog - 93K courses",
    "inspection_grade_range": "School inspection grade ranges",
    "building_permits": "Municipality building permits - 30K records",
    "building_summary_information": "Building details (area, type, status) - 30K records",
    "building_floor_level_information": "Building floor/unit details - 20K records",
    "consultant_projects": "Construction projects with consultants - 20K records",
    "food_business_information": "Food business licenses & activities - 20K records",
    "food_item_catalogue": "Registered food products catalog - 20K records",
    "food_health_certificate": "Food import health certificates - 20K records",
    "consignments": "Import consignments by port, country, type - 20K records",
    "inprogress_applications_sla": "Municipality applications SLA tracking - 36K records",
    "food_item_tests": "Food safety test results - 30K records",
    "registered_cosmetics": "Registered cosmetics & personal care products - 20K records",
    "metro_stations": "Dubai Metro station locations",
    "metro_lines": "Metro line definitions",
    "metro_ridership": "Metro ridership data (Jan 2026) - 50K records",
    "bus_routes": "Dubai bus route definitions",
    "bus_stop_details": "Bus stop locations and facilities - 20K records",
    "bus_ridership": "Bus ridership data (Jan 2026) - 50K records",
    "bus_network_coverage": "Bus network geographic coverage by community",
    "tram_stations": "Dubai Tram station locations",
    "tram_lines": "Tram line definitions",
    "tram_ridership": "Tram ridership data (Jan 2026) - 48K records",
    "marine_stations": "Marine transport station locations",
    "marine_lines": "Marine transport route definitions",
    "marine_ridership": "Marine ridership data (Jan 2026)",
    "public_transportation_routes_stops": "All public transport routes/stops - 18K records",
    "public_transportation_stations": "All public transport stations",
    "taxi_stand_locations": "Taxi stand locations in Dubai",
    "drivers_population_census": "Driver license census by type, nationality, gender",
    "number_of_parking_spaces_per_zone": "Parking spaces by zone/community",
    "hex_master_list": "Dubai hex grid with community mapping and coordinates",
}


def _create_performance_indexes(conn, loaded_tables: dict):
    """Create indexes on commonly filtered/joined columns for query speed."""
    # Column names that are frequently used in WHERE / JOIN clauses
    INDEX_COLUMNS = [
        "area_name_en", "area_name", "location_area", "community",
        "property_type_en", "trans_group_en", "rooms_en",
        "developer_name", "building_name_en",
        "latitude", "longitude", "lat", "lng",
        "station_location_latitude", "station_location_longitude",
        "completion_status", "usage_en", "project_name_en",
        "nearest_metro_en", "nearest_mall_en", "nearest_landmark_en",
        "actual_worth", "meter_sale_price",
        "line_name", "type", "status", "category",
    ]
    cursor = conn.cursor()
    idx_count = 0
    for table_name in loaded_tables:
        # Get actual columns in this table
        try:
            cursor.execute(f"PRAGMA table_info({table_name})")
            actual_cols = {row[1] for row in cursor.fetchall()}
        except Exception:
            continue
        for col in INDEX_COLUMNS:
            if col in actual_cols:
                idx_name = f"idx_{table_name}_{col}"
                try:
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name}({col})")
                    idx_count += 1
                except Exception:
                    pass
    conn.commit()
    print(f"Created {idx_count} performance indexes")



def _create_short_name_views(conn, existing_tables):
    """Create short-name views for long-prefixed tables in production.db.

    E.g. 'dld_transactions_transactions' gets a view named 'transactions',
    'rta_rail_metro_stations' gets 'metro_stations', etc.
    This lets the manual SCHEMA entries and SQL generation work unchanged.
    """
    # Explicit mapping for ambiguous short names (multiple tables end with same suffix)
    _EXPLICIT_MAP = {
        "transactions": "dld_transactions_transactions",
        "bayut_commercial_transactions": "bayut_transactions_bayut_commercial_transactions",
    }

    manual_short_names = {s["name"] for s in SCHEMA}
    existing_set = set(existing_tables)
    created = 0

    for short_name in manual_short_names:
        # Skip if the short name already exists as a real table or view
        if short_name in existing_set:
            continue

        long_name = None

        # Check explicit mapping first
        if short_name in _EXPLICIT_MAP and _EXPLICIT_MAP[short_name] in existing_set:
            long_name = _EXPLICIT_MAP[short_name]
        else:
            # Find matching long-prefixed tables that end with _short_name
            candidates = [t for t in existing_tables if t.endswith("_" + short_name)]
            if len(candidates) == 1:
                long_name = candidates[0]
            elif len(candidates) > 1:
                # Prefer shortest name (least nesting)
                long_name = min(candidates, key=len)

        if long_name:
            try:
                conn.execute(f'CREATE VIEW IF NOT EXISTS "{short_name}" AS SELECT * FROM "{long_name}"')
                created += 1
                existing_set.add(short_name)
            except Exception as e:
                print(f"  WARN: Could not create view {short_name} -> {long_name}: {e}")

    if created:
        conn.commit()
        print(f"Created {created} short-name views for pre-built DB tables")


def init_database():
    """Load database — from pre-built production.db if available, else from CSVs.

    Auto-discovers all CSVs, creates clean table names, loads data,
    and generates SCHEMA entries for the dynamic filtering UI.
    """
    global DB_CONNECTION, SCHEMA

    # --- FAST PATH: Load pre-built DB file (built by build_db.py) ---
    if PREBUILT_DB.exists():
        size_mb = PREBUILT_DB.stat().st_size / (1024 * 1024)
        print(f"\nLoading pre-built database: {PREBUILT_DB.name} ({size_mb:.1f} MB)")
        DB_CONNECTION = sqlite3.connect(str(PREBUILT_DB), check_same_thread=False, timeout=30)
        DB_CONNECTION.execute("PRAGMA busy_timeout = 5000")
        DB_CONNECTION.execute("PRAGMA journal_mode = WAL")
        _install_query_timeout(DB_CONNECTION)

        # Discover tables and auto-generate schema entries
        cursor = DB_CONNECTION.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        manual_schema_names = {s["name"] for s in SCHEMA}
        auto_schema_entries = []
        total_rows = 0

        for table_name in tables:
            try:
                row_count = DB_CONNECTION.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
                total_rows += row_count
            except Exception:
                row_count = 0

            if table_name not in manual_schema_names:
                # Get column info
                col_info = DB_CONNECTION.execute(f'PRAGMA table_info("{table_name}")').fetchall()
                columns = [{"name": c[1], "type": c[2] or "TEXT"} for c in col_info]
                auto_schema_entries.append({
                    "name": table_name,
                    "description": f"{table_name.replace('_', ' ').title()} ({row_count:,} rows)",
                    "columns": columns,
                })

        SCHEMA.extend(auto_schema_entries)
        print(f"Database loaded: {len(tables)} tables, {total_rows:,} total rows")
        print(f"Schema: {len(SCHEMA)} table definitions")

        # Create short-name views for long-prefixed tables so manual SCHEMA
        # entries (e.g. "transactions") resolve against the actual DB tables
        # (e.g. "dld_transactions_transactions").
        _create_short_name_views(DB_CONNECTION, tables)

        # Build area coordinates if not already in DB
        cursor = DB_CONNECTION.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='area_coordinates'"
        )
        if not cursor.fetchone():
            _load_area_coordinates_table(DB_CONNECTION)
        else:
            print("area_coordinates table already present in pre-built DB")

        return

    # --- SLOW PATH: Build from CSVs (local development) ---
    print("\nNo pre-built DB found, loading from CSVs...")
    DB_CONNECTION = sqlite3.connect(":memory:", check_same_thread=False, timeout=30)
    DB_CONNECTION.execute("PRAGMA busy_timeout = 5000")
    _install_query_timeout(DB_CONNECTION)

    # Auto-discover all CSV files in DATA_DIR
    csv_files = sorted(DATA_DIR.rglob("*.csv"))
    print(f"\nDiscovered {len(csv_files)} CSV files in {DATA_DIR.name}/")

    # Track loaded tables to avoid name collisions
    loaded_tables = {}
    total_rows = 0

    # Keep the manual SCHEMA entries for core tables (they have curated columns)
    manual_schema_names = {s["name"] for s in SCHEMA}
    auto_schema_entries = []

    for csv_file in csv_files:
        rel_path = csv_file.relative_to(DATA_DIR)
        table_name = _csv_to_table_name(str(rel_path))

        # Handle name collisions by appending category prefix
        if table_name in loaded_tables:
            category = str(rel_path).split("/")[0].lower()
            table_name = f"{category}_{table_name}"

        if table_name in loaded_tables:
            print(f"  SKIP (duplicate name): {rel_path} -> {table_name}")
            continue

        try:
            df = pd.read_csv(csv_file, low_memory=False)
            if len(df) == 0:
                print(f"  SKIP (empty): {rel_path}")
                continue

            # Clean column names: lowercase, replace spaces with underscores
            df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_').replace('.', '_') for c in df.columns]

            df.to_sql(table_name, DB_CONNECTION, if_exists="replace", index=False)
            loaded_tables[table_name] = str(rel_path)
            total_rows += len(df)
            print(f"  Loaded {table_name:50s} ({len(df):>8,} rows) <- {rel_path}")

            # Auto-generate SCHEMA entry if not already manually defined
            if table_name not in manual_schema_names:
                # Get category from path
                parts = str(rel_path).split("/")
                category = parts[0] if len(parts) > 1 else "General"

                # Build description
                desc = TABLE_DESCRIPTIONS.get(table_name, "")
                if not desc:
                    cat_desc = CATEGORY_DESCRIPTIONS.get(category, category)
                    desc = f"{cat_desc} - {table_name.replace('_', ' ').title()} ({len(df):,} rows)"

                # Build column definitions
                columns = []
                for col in df.columns:
                    col_type = _infer_sql_type(df[col])
                    col_entry = {"name": col, "type": col_type}
                    # Mark likely PKs
                    if col in ("id", f"{table_name}_id") or col.endswith("_id") and df[col].nunique() == len(df):
                        col_entry["is_pk"] = True
                    columns.append(col_entry)

                auto_schema_entries.append({
                    "name": table_name,
                    "description": desc,
                    "category": category,
                    "columns": columns,
                })

        except Exception as e:
            print(f"  ERROR loading {rel_path}: {e}")

    # Merge auto-generated schema into SCHEMA
    SCHEMA.extend(auto_schema_entries)

    print(f"\n{'='*60}")
    print(f"Database loaded: {len(loaded_tables)} tables, {total_rows:,} total rows")
    print(f"Schema: {len(SCHEMA)} table definitions")
    print(f"{'='*60}\n")

    # Create indexes on commonly queried columns for performance
    _create_performance_indexes(DB_CONNECTION, loaded_tables)

    # Load area coordinates into SQLite as a geo-lookup table
    _load_area_coordinates_table(DB_CONNECTION)


def _load_area_coordinates_table(conn):
    """
    Build area_coordinates lookup table from REAL database data.

    Strategy:
    1. Compute centroids from bayut_transactions (has native lat/lon)
    2. Map DLD area names to bayut areas via known aliases
    3. Insert ALL names (both DLD and bayut) so JOINs work for any table
    """
    rows = []
    seen = set()

    def _add(name, lat, lng, src="db"):
        key = name.strip().lower()
        if key in seen or not name or not lat or not lng:
            return
        rows.append({"area_name": name.strip(), "lat": float(lat), "lng": float(lng), "source": src})
        seen.add(key)

    # --- Source 1: Bayut centroids (ground-truth coordinates) ---
    bayut_centroids = {}
    try:
        df = pd.read_sql_query("""
            SELECT location_area,
                   AVG(latitude) AS lat, AVG(longitude) AS lng
            FROM bayut_transactions
            WHERE latitude IS NOT NULL AND latitude != 0
              AND longitude IS NOT NULL AND longitude != 0
            GROUP BY location_area
        """, conn)
        for _, r in df.iterrows():
            name = r["location_area"]
            if name:
                bayut_centroids[name] = (r["lat"], r["lng"])
                _add(name, r["lat"], r["lng"], "bayut")
    except Exception:
        pass

    # --- Source 2: DLD area names mapped to coordinates ---
    # Known mapping: DLD official area name → bayut/common area name
    DLD_TO_BAYUT = {
        "Marsa Dubai": "Dubai Marina",
        "Al Thanyah Fifth": "Jumeirah Village Circle (JVC)",
        "Al Thanyah Third": "Jumeirah Lake Towers (JLT)",
        "Burj Khalifa": "Downtown Dubai",
        "Al Barsha South Fourth": "Arjan",
        "Al Hebiah Fourth": "Motor City",
        "Al Hebiah Fifth": "Dubai Sports City",
        "Al Hebiah First": "Dubai Sports City",
        "Al Hebiah Third": "Dubai Studio City",
        "Al Hebiah Second": "Dubai Sports City",
        "Al Hebiah Sixth": "Dubai Sports City",
        "Hadaeq Sheikh Mohammed Bin Rashid": "Mohammed Bin Rashid City",
        "Wadi Al Safa 5": "Dubailand",
        "Wadi Al Safa 3": "Dubailand",
        "Wadi Al Safa 7": "Dubailand",
        "Al Warsan First": "International City",
        "Al Warsan Second": "International City",
        "Al Merkadh": "Dubai Creek Harbour",
        "Madinat Al Mataar": "Dubai South",
        "Al Thanayah Fourth": "Barsha Heights (Tecom)",
        "Me'Aisem First": "Dubai Production City (IMPZ)",
        "Me'Aisem Second": "Dubai Production City (IMPZ)",
        "Nadd Hessa": "Dubai Silicon Oasis (DSO)",
        "Al Khairan First": "Dubai Hills Estate",
        "Al Yelayiss 2": "Dubailand",
        "Al Yelayiss 1": "Dubailand",
        "Jabal Ali First": "Jebel Ali",
        "Jabal Ali Second": "Jebel Ali",
        "Al Jadaf": "Al Jaddaf",
        "Al Karama": "Al Satwa",
        "Al Goze First": "Al Quoz",
        "Al Goze Fourth": "Al Quoz",
        "Al Goze Third": "Al Quoz",
        "Al Goze Industrial First": "Al Quoz",
        "Al Goze Industrial Second": "Al Quoz",
        "Al Goze Industrial Third": "Al Quoz",
        "Al Goze Industrial Fourth": "Al Quoz",
        "Al Barshaa South First": "Al Barsha",
        "Al Barshaa South Second": "Al Barsha",
        "Al Barshaa South Third": "Al Barsha",
        "Al Barsha South Fifth": "Al Barsha",
        "Al Barsha First": "Al Barsha",
        "Al Barsha Second": "Al Barsha",
        "Al Barsha Third": "Al Barsha",
        "Al Safouh First": "Al Sufouh",
        "Al Safouh Second": "Al Sufouh",
        "Al Saffa First": "Jumeirah",
        "Al Saffa Second": "Jumeirah",
        "Al Safaa": "Jumeirah",
        "Al Rowaiyah First": "Meydan City",
        "Al Rowaiyah Third": "Meydan City",
        "Al Ruwayyah": "Meydan City",
        "Al Nahda First": "Al Qusais",
        "Al Nahda Second": "Al Qusais",
        "Umm Al Sheif": "Umm Suqeim",
        "Al Manara": "Umm Suqeim",
        "World Islands": "The World Islands",
        "Al Qoaz": "Al Quoz",
        "Al Garhoud": "Deira",
        "Oud Al Muteena First": "Muhaisnah",
        "Oud Al Muteena Second": "Muhaisnah",
        "Al Qusais Industrial First": "Al Qusais",
        "Al Qusais Industrial Second": "Al Qusais",
        "Al Qusais Industrial Third": "Al Qusais",
        "Al Qusais Industrial Fourth": "Al Qusais",
        "Al Qusais Industrial Fifth": "Al Qusais",
        "Al Khawaneej First": "Mirdif",
        "Al Khawaneej Second": "Mirdif",
        "Al Khawaneej": "Mirdif",
        "Al Mizhar First": "Al Mizhar",
        "Al Mizhar Second": "Al Mizhar",
        "Al Mizhar Third": "Al Mizhar",
    }

    # Get all distinct DLD area names and map them
    try:
        dld_areas = pd.read_sql_query(
            "SELECT DISTINCT area_name_en FROM transactions WHERE area_name_en IS NOT NULL",
            conn,
        )
        for _, r in dld_areas.iterrows():
            dld_name = r["area_name_en"]
            # Try known mapping first
            bayut_name = DLD_TO_BAYUT.get(dld_name)
            if bayut_name and bayut_name in bayut_centroids:
                lat, lng = bayut_centroids[bayut_name]
                _add(dld_name, lat, lng, "dld_mapped")
            elif dld_name in bayut_centroids:
                # Exact match (e.g. "Business Bay", "Palm Jumeirah")
                lat, lng = bayut_centroids[dld_name]
                _add(dld_name, lat, lng, "dld_exact")
            else:
                # Try case-insensitive partial match
                dld_lower = dld_name.lower()
                for bname, (blat, blng) in bayut_centroids.items():
                    if dld_lower in bname.lower() or bname.lower() in dld_lower:
                        _add(dld_name, blat, blng, "dld_fuzzy")
                        break
    except Exception:
        pass

    # Also add lkp_areas names (they match DLD area_name_en)
    try:
        lkp = pd.read_sql_query(
            "SELECT DISTINCT name_en FROM lkp_areas WHERE name_en IS NOT NULL", conn
        )
        for _, r in lkp.iterrows():
            name = r["name_en"]
            bayut_name = DLD_TO_BAYUT.get(name)
            if bayut_name and bayut_name in bayut_centroids:
                lat, lng = bayut_centroids[bayut_name]
                _add(name, lat, lng, "lkp_mapped")
            # If already added via DLD path, skip (seen check handles it)
    except Exception:
        pass

    # --- Source 3: JSON files as fallback for remaining areas ---
    for fpath in ["dubai_area_polygons.json", "dubai_areas_coordinates.json"]:
        full = Path(__file__).parent / fpath
        if full.exists():
            try:
                data = json.load(open(full))
                for item in data:
                    lat = item.get("lat") or (item.get("center") or [None])[0]
                    lng = item.get("lng") or (item.get("center") or [None, None])[1]
                    if lat and lng:
                        _add(item["name"], lat, lng, "json")
            except Exception:
                pass

    if rows:
        df = pd.DataFrame(rows)
        df.to_sql("area_coordinates", conn, if_exists="replace", index=False)
        mapped_count = len([r for r in rows if r["source"].startswith("dld")])
        print(f"Loaded area_coordinates: {len(df)} entries ({mapped_count} DLD areas mapped to coordinates)")


def get_db():
    """Get database connection."""
    if DB_CONNECTION is None:
        init_database()
    return DB_CONNECTION


# Max query execution time (seconds) — prevents runaway queries from locking the server
_QUERY_TIMEOUT_SECONDS = int(os.environ.get("QUERY_TIMEOUT", "30"))
_query_start_time = 0.0


def _progress_handler():
    """SQLite progress callback — raises interrupt if query exceeds timeout."""
    if time.time() - _query_start_time > _QUERY_TIMEOUT_SECONDS:
        return 1  # non-zero = interrupt the query
    return 0


def _install_query_timeout(conn):
    """Install a progress handler that aborts queries after QUERY_TIMEOUT seconds."""
    conn.set_progress_handler(_progress_handler, 10000)  # check every 10K VM ops


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
    limit: int = 500


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


_llm_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm")

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

    # Run the synchronous LLM call in a thread pool so it doesn't block the event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_llm_pool, smart_extractor.extract_sync, query)

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
        ("metro_stations", "transactions"): "transactions.nearest_metro_en = metro_stations.location_name_english",
        ("metro_stations", "bayut_transactions"): "1=1",  # Would need geo-spatial join

        # ==========================================
        # Education (KHDA) joins
        # ==========================================
        ("school_search", "lkp_areas"): "school_search.area_en = lkp_areas.name_en",
        ("school_search", "units"): "school_search.area_en = units.area_name_en",

        # ==========================================
        # Health (DHA) joins
        # ==========================================
        ("sheryan_facility_detail", "lkp_areas"): "sheryan_facility_detail.area_english = lkp_areas.name_en",
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
    global _query_start_time
    db = get_db()

    try:
        sql = request.sql
        if "LIMIT" not in sql.upper():
            sql += f" LIMIT {request.limit}"

        _query_start_time = time.time()
        df = pd.read_sql_query(sql, db)
        data_json = json.loads(df.to_json(orient="records"))

        return {
            "columns": list(df.columns),
            "data": data_json,
            "row_count": len(df),
        }
    except sqlite3.OperationalError as e:
        if "interrupt" in str(e).lower():
            raise HTTPException(status_code=408, detail=f"Query timed out after {_QUERY_TIMEOUT_SECONDS}s. Try a simpler query or add filters to reduce data.")
        raise HTTPException(status_code=400, detail=str(e))
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


_categorical_cache: Dict[str, Any] = {"data": None}


def _compute_categorical_columns_sync() -> dict:
    """Compute categorical columns once — uses sampling for speed."""
    db = get_db()
    result = {}
    cursor = db.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        cursor.execute(f'PRAGMA table_info([{table}])')
        columns_info = cursor.fetchall()

        table_categoricals = {}
        for col_info in columns_info:
            col_name = col_info[1]
            col_type = (col_info[2] or "").upper()

            if not any(t in col_type for t in ["TEXT", "VARCHAR", "CHAR"]):
                continue

            try:
                # Fast: get distinct values with a hard LIMIT — skip full COUNT(DISTINCT)
                cursor.execute(f"""
                    SELECT DISTINCT [{col_name}]
                    FROM [{table}]
                    WHERE [{col_name}] IS NOT NULL AND [{col_name}] != ''
                    ORDER BY [{col_name}]
                    LIMIT 501
                """)
                values = [row[0] for row in cursor.fetchall()]

                # If more than 500, it's not categorical — skip
                if 0 < len(values) <= 500:
                    table_categoricals[col_name] = values
            except Exception:
                pass

        if table_categoricals:
            result[table] = table_categoricals

    return result


@app.get("/api/categorical_columns")
async def get_categorical_columns():
    """Auto-detect categorical columns from ALL tables and return distinct values for filter dropdowns."""
    if _categorical_cache["data"] is None:
        _categorical_cache["data"] = _compute_categorical_columns_sync()
    return _categorical_cache["data"]


# ============================================================================
# Intelligence System Endpoints
# ============================================================================

_briefing_cache = {"data": None, "generated_at": None}

@app.get("/api/intelligence/briefing")
async def get_intelligence_briefing():
    """
    Get the latest intelligence briefing with insights, risks, and opportunities.
    Cached for 30 minutes to avoid repeated LLM calls.
    """
    if not intelligence_generator:
        raise HTTPException(status_code=503, detail="Intelligence system not initialized")

    # Return cached briefing if less than 30 min old
    import time as _time
    if (_briefing_cache["data"] and _briefing_cache["generated_at"]
            and _time.time() - _briefing_cache["generated_at"] < 1800):
        return _briefing_cache["data"]

    try:
        loop = asyncio.get_event_loop()
        briefing = await loop.run_in_executor(
            _llm_pool, intelligence_generator.generate_briefing
        )
        result = briefing.to_dict()
        _briefing_cache["data"] = result
        _briefing_cache["generated_at"] = _time.time()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate briefing: {str(e)}")


@app.get("/api/intelligence/metrics")
async def get_intelligence_metrics():
    """Get current snapshot of all key market metrics."""
    if not intelligence_tracker:
        raise HTTPException(status_code=503, detail="Intelligence system not initialized")

    try:
        return intelligence_tracker.get_metrics_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.get("/api/intelligence/anomalies")
async def get_anomalies():
    """Get detected anomalies in the current data."""
    if not intelligence_tracker:
        raise HTTPException(status_code=503, detail="Intelligence system not initialized")

    try:
        metrics = intelligence_tracker.get_metrics_snapshot()
        anomalies = intelligence_tracker.detect_anomalies(metrics)
        return [a.to_dict() for a in anomalies]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect anomalies: {str(e)}")


@app.get("/api/intelligence/explain")
async def explain_metric(
    metric_type: str = Query(..., description="Type of metric (price_spike, volume_spike, yield_decline, etc.)"),
    entity: str = Query(..., description="Entity name (area, property type, etc.)"),
    domain: str = Query("area", description="Domain (area, property_type, transaction_type)")
):
    """
    Get causal explanation for a specific metric/entity combination.

    Example: /api/intelligence/explain?metric_type=price_spike&entity=Palm Jumeirah&domain=area
    """
    if not intelligence_explainer:
        raise HTTPException(status_code=503, detail="Intelligence system not initialized")

    try:
        from intelligence import Anomaly, Severity

        # Create a synthetic anomaly for explanation
        anomaly = Anomaly(
            anomaly_id=f"EXPLAIN-{datetime.now().strftime('%H%M%S')}",
            metric_type=metric_type,
            domain=domain,
            entity=entity,
            current_value=0,  # Will be looked up
            baseline_value=0,
            change_percent=0,
            severity=Severity.MEDIUM
        )

        insight = intelligence_explainer.explain(anomaly)
        return insight.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {str(e)}")


@app.post("/api/intelligence/ask")
async def ask_intelligence(query: str = Query(..., description="Natural language question about the market")):
    """
    Ask a natural language question about market conditions.

    Examples:
    - "Why are prices rising in Palm Jumeirah?"
    - "What are the risks in Dubai Marina?"
    - "Where are the best investment opportunities?"
    """
    if not gemini_model:
        raise HTTPException(status_code=503, detail="LLM not available")

    try:
        # Get current metrics for context
        metrics = intelligence_tracker.get_metrics_snapshot() if intelligence_tracker else {}

        # Build context
        context = {
            "total_transactions": metrics.get("total_transactions", "N/A"),
            "total_sales_value": metrics.get("total_sales_value", "N/A"),
            "top_areas": metrics.get("sales_by_area", [])[:5],
            "rental_yields": metrics.get("rental_yields", [])[:5],
        }

        prompt = f"""You are a Dubai real estate market intelligence agent. Answer this question based on the market data.

QUESTION: {query}

CURRENT MARKET DATA:
- Total Transactions: {context['total_transactions']:,} if isinstance(context['total_transactions'], (int, float)) else context['total_transactions']
- Top Areas by Volume: {json.dumps(context['top_areas'][:3], default=str)}
- Rental Yields: {json.dumps(context['rental_yields'][:3], default=str)}

Provide a concise, data-driven answer (2-4 sentences). Include specific numbers where available.
If the data doesn't directly answer the question, explain what the data does show and suggest what additional analysis might help."""

        response = gemini_model.generate_content(prompt)
        return {
            "question": query,
            "answer": response.text.strip(),
            "context_used": list(context.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")


# ============================================================================
# Local Intelligence — Insights on queried data subset
# ============================================================================

class LocalInsightRequest(BaseModel):
    query: str  # original natural language question
    sql: str    # the SQL that was executed
    columns: List[str]
    data: List[Dict[str, Any]]
    row_count: int = 0


@app.post("/api/intelligence/local")
async def get_local_insights(request: LocalInsightRequest):
    """
    Analyze a query result subset and generate insights a human would miss.

    Returns:
    - Statistical patterns (outliers, distributions, concentrations)
    - Comparisons to market baselines
    - Hidden correlations
    - Actionable observations
    """
    if not request.data or len(request.data) == 0:
        return {"insights": [], "summary": "No data to analyze."}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _llm_pool,
        _generate_local_insights,
        request.query,
        request.sql,
        request.columns,
        request.data,
        request.row_count,
    )
    return result


# ============================================================================
# Temporal Market Trends — period-over-period analysis
# ============================================================================
_trends_cache: Dict = {"data": None, "generated_at": None}


@app.get("/api/intelligence/trends")
async def get_market_trends():
    """
    Time-based market trends: recent 3 months vs prior 3 months,
    recent 12 months vs prior 12 months, and monthly series.
    """
    # Cache for 30 minutes
    if _trends_cache["data"] and _trends_cache["generated_at"]:
        age = (datetime.now() - _trends_cache["generated_at"]).total_seconds()
        if age < 1800:
            return _trends_cache["data"]

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_llm_pool, _compute_temporal_trends)
    _trends_cache["data"] = result
    _trends_cache["generated_at"] = datetime.now()
    return result


def _compute_temporal_trends() -> Dict:
    """Compute temporal market trends (runs in thread pool)."""
    db = get_db()
    cursor = db.cursor()

    trends: Dict = {"periods": [], "area_momentum": [], "monthly_series": [], "generated_at": datetime.now().isoformat()}

    # ---------- Detect which date column & table we have ----------
    # Also detect the date format so we can build proper SQL expressions
    date_configs = [
        ("transactions", "instance_date", "actual_worth", "area_name_en", "trans_group_en = 'Sales' AND actual_worth > 0"),
        ("bayut_transactions", "date_transaction", "transaction_amount", "location_area", "transaction_amount > 0"),
    ]

    date_col = None
    price_col = None
    table_name = None
    area_col = None
    extra_where = None
    date_format = None  # 'iso' (yyyy-mm-dd) or 'dmy' (dd-mm-yyyy)

    for tbl, dcol, pcol, acol, where in date_configs:
        try:
            cursor.execute(f"SELECT {dcol} FROM {tbl} WHERE {dcol} IS NOT NULL AND {dcol} != '' LIMIT 1")
            sample = cursor.fetchone()
            if sample:
                table_name, date_col, price_col, area_col, extra_where = tbl, dcol, pcol, acol, where
                # Detect format from sample value
                val = str(sample[0]).strip()[:10]
                if len(val) >= 10 and val[4] == '-':
                    date_format = 'iso'  # yyyy-mm-dd
                elif len(val) >= 10 and val[2] == '-':
                    date_format = 'dmy'  # dd-mm-yyyy
                elif len(val) >= 10 and val[2] == '/':
                    date_format = 'dmy_slash'  # dd/mm/yyyy
                else:
                    date_format = 'iso'  # assume iso
                break
        except Exception:
            continue

    if not date_col:
        trends["error"] = "No date column found"
        return trends

    # ---------- Build a SQL expression that normalises dates to yyyy-mm-dd ----------
    if date_format == 'dmy':
        # dd-mm-yyyy → yyyy-mm-dd via SQLite string ops
        norm_date = f"(SUBSTR({date_col},7,4)||'-'||SUBSTR({date_col},4,2)||'-'||SUBSTR({date_col},1,2))"
        norm_month = f"(SUBSTR({date_col},7,4)||'-'||SUBSTR({date_col},4,2))"
    elif date_format == 'dmy_slash':
        norm_date = f"(SUBSTR({date_col},7,4)||'-'||SUBSTR({date_col},4,2)||'-'||SUBSTR({date_col},1,2))"
        norm_month = f"(SUBSTR({date_col},7,4)||'-'||SUBSTR({date_col},4,2))"
    else:
        norm_date = date_col
        norm_month = f"SUBSTR({date_col},1,7)"

    # ---------- Find date range ----------
    try:
        cursor.execute(f"SELECT MIN({norm_date}), MAX({norm_date}) FROM {table_name} WHERE {date_col} IS NOT NULL AND {date_col} != ''")
        row = cursor.fetchone()
        date_min_str, date_max_str = row[0], row[1]
        trends["date_range"] = {"min": date_min_str, "max": date_max_str}
    except Exception as e:
        trends["error"] = f"Cannot determine date range: {e}"
        return trends

    # ---------- Helper: make date filter using normalised expression ----------
    def _make_date_filter(start_str, end_str):
        return f"{norm_date} >= '{start_str}' AND {norm_date} < '{end_str}'"

    # ---------- Period comparisons ----------
    from dateutil.relativedelta import relativedelta
    try:
        max_date = datetime.strptime(date_max_str.strip()[:10], '%Y-%m-%d')
        min_date = datetime.strptime(date_min_str.strip()[:10], '%Y-%m-%d')
        trends["parsed_range"] = {"min": min_date.strftime('%Y-%m-%d'), "max": max_date.strftime('%Y-%m-%d')}
    except Exception as e:
        trends["error"] = f"Date parsing error: {e}"
        return trends

    # Define comparison windows
    period_defs = []
    # 3-month window
    r3_start = max_date - relativedelta(months=3)
    p3_start = r3_start - relativedelta(months=3)
    if p3_start >= min_date:
        period_defs.append({
            "label": "Last 3 Months",
            "recent_start": r3_start.strftime('%Y-%m-%d'),
            "recent_end": (max_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            "prior_start": p3_start.strftime('%Y-%m-%d'),
            "prior_end": r3_start.strftime('%Y-%m-%d'),
        })

    # 12-month window
    r12_start = max_date - relativedelta(months=12)
    p12_start = r12_start - relativedelta(months=12)
    if p12_start >= min_date:
        period_defs.append({
            "label": "Last 12 Months",
            "recent_start": r12_start.strftime('%Y-%m-%d'),
            "recent_end": (max_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            "prior_start": p12_start.strftime('%Y-%m-%d'),
            "prior_end": r12_start.strftime('%Y-%m-%d'),
        })

    # Full-range halves (split data in two)
    mid_date = min_date + (max_date - min_date) / 2
    period_defs.append({
        "label": "Recent Half vs Earlier Half",
        "recent_start": mid_date.strftime('%Y-%m-%d'),
        "recent_end": (max_date + timedelta(days=1)).strftime('%Y-%m-%d'),
        "prior_start": min_date.strftime('%Y-%m-%d'),
        "prior_end": mid_date.strftime('%Y-%m-%d'),
    })

    for pdef in period_defs:
        try:
            recent_filter = _make_date_filter(pdef["recent_start"], pdef["recent_end"])
            prior_filter = _make_date_filter(pdef["prior_start"], pdef["prior_end"])

            # Recent period stats
            cursor.execute(f"""
                SELECT COUNT(*) as cnt, AVG({price_col}) as avg_price,
                       SUM({price_col}) as total_val, MAX({price_col}) as max_price
                FROM {table_name}
                WHERE {extra_where} AND {recent_filter}
            """)
            r = cursor.fetchone()
            recent = {"count": r[0] or 0, "avg_price": r[1] or 0, "total_value": r[2] or 0, "max_price": r[3] or 0}

            # Prior period stats
            cursor.execute(f"""
                SELECT COUNT(*) as cnt, AVG({price_col}) as avg_price,
                       SUM({price_col}) as total_val, MAX({price_col}) as max_price
                FROM {table_name}
                WHERE {extra_where} AND {prior_filter}
            """)
            p = cursor.fetchone()
            prior = {"count": p[0] or 0, "avg_price": p[1] or 0, "total_value": p[2] or 0, "max_price": p[3] or 0}

            vol_change = ((recent["count"] - prior["count"]) / prior["count"] * 100) if prior["count"] > 0 else 0
            price_change = ((recent["avg_price"] - prior["avg_price"]) / prior["avg_price"] * 100) if prior["avg_price"] > 0 else 0
            val_change = ((recent["total_value"] - prior["total_value"]) / prior["total_value"] * 100) if prior["total_value"] > 0 else 0

            trends["periods"].append({
                "label": pdef["label"],
                "recent_range": f"{pdef['recent_start']} to {pdef['recent_end'][:10]}",
                "prior_range": f"{pdef['prior_start']} to {pdef['prior_end'][:10]}",
                "metrics": {
                    "transactions": {"recent": recent["count"], "prior": prior["count"], "change_pct": round(vol_change, 1)},
                    "avg_price": {"recent": round(recent["avg_price"]), "prior": round(prior["avg_price"]), "change_pct": round(price_change, 1)},
                    "total_value": {"recent": round(recent["total_value"]), "prior": round(prior["total_value"]), "change_pct": round(val_change, 1)},
                },
            })
        except Exception as e:
            print(f"Period {pdef['label']} error: {e}")

    # ---------- Area momentum: which areas gained/lost the most ----------
    try:
        mid_str = mid_date.strftime('%Y-%m-%d')
        end_str = (max_date + timedelta(days=1)).strftime('%Y-%m-%d')
        start_str = min_date.strftime('%Y-%m-%d')

        cursor.execute(f"""
            SELECT {area_col},
                   COUNT(*) as cnt,
                   AVG({price_col}) as avg_p
            FROM {table_name}
            WHERE {extra_where} AND {norm_date} >= '{mid_str}' AND {norm_date} < '{end_str}'
              AND {area_col} IS NOT NULL
            GROUP BY {area_col}
            HAVING cnt >= 3
        """)
        recent_areas = {row[0]: {"count": row[1], "avg_price": row[2]} for row in cursor.fetchall()}

        cursor.execute(f"""
            SELECT {area_col},
                   COUNT(*) as cnt,
                   AVG({price_col}) as avg_p
            FROM {table_name}
            WHERE {extra_where} AND {norm_date} >= '{start_str}' AND {norm_date} < '{mid_str}'
              AND {area_col} IS NOT NULL
            GROUP BY {area_col}
            HAVING cnt >= 3
        """)
        prior_areas = {row[0]: {"count": row[1], "avg_price": row[2]} for row in cursor.fetchall()}

        momentum = []
        for area in set(recent_areas) & set(prior_areas):
            rc = recent_areas[area]
            pc = prior_areas[area]
            price_chg = ((rc["avg_price"] - pc["avg_price"]) / pc["avg_price"] * 100) if pc["avg_price"] > 0 else 0
            vol_chg = ((rc["count"] - pc["count"]) / pc["count"] * 100) if pc["count"] > 0 else 0
            momentum.append({
                "area": area,
                "price_change_pct": round(price_chg, 1),
                "volume_change_pct": round(vol_chg, 1),
                "recent_avg_price": round(rc["avg_price"]),
                "recent_volume": rc["count"],
                "prior_avg_price": round(pc["avg_price"]),
                "prior_volume": pc["count"],
            })

        # Sort by absolute price change to show biggest movers
        momentum.sort(key=lambda x: abs(x["price_change_pct"]), reverse=True)
        trends["area_momentum"] = momentum[:10]

    except Exception as e:
        print(f"Area momentum error: {e}")

    # ---------- Monthly transaction series ----------
    try:
        cursor.execute(f"""
            SELECT
                {norm_month} as month,
                COUNT(*) as cnt,
                AVG({price_col}) as avg_p,
                SUM({price_col}) as total_v
            FROM {table_name}
            WHERE {extra_where} AND {date_col} IS NOT NULL AND {date_col} != ''
            GROUP BY month
            ORDER BY month
        """)
        monthly = []
        for row in cursor.fetchall():
            if row[0] and len(row[0]) >= 7:
                monthly.append({
                    "month": row[0],
                    "transactions": row[1] or 0,
                    "avg_price": round(row[2] or 0),
                    "total_value": round(row[3] or 0),
                })
        trends["monthly_series"] = monthly
    except Exception as e:
        print(f"Monthly series error: {e}")

    # ---------- Rental trends (if available) ----------
    try:
        cursor.execute("SELECT contract_start_date FROM rent_contracts WHERE contract_start_date IS NOT NULL AND contract_start_date != '' LIMIT 1")
        rent_sample = cursor.fetchone()
        if rent_sample:
            # Detect rental date format
            rv = str(rent_sample[0]).strip()[:10]
            if len(rv) >= 10 and rv[2] == '-':
                rent_norm_month = "(SUBSTR(contract_start_date,7,4)||'-'||SUBSTR(contract_start_date,4,2))"
            elif len(rv) >= 10 and rv[2] == '/':
                rent_norm_month = "(SUBSTR(contract_start_date,7,4)||'-'||SUBSTR(contract_start_date,4,2))"
            else:
                rent_norm_month = "SUBSTR(contract_start_date,1,7)"

            cursor.execute(f"""
                SELECT
                    {rent_norm_month} as month,
                    COUNT(*) as cnt,
                    AVG(annual_amount) as avg_rent
                FROM rent_contracts
                WHERE annual_amount > 0 AND contract_start_date IS NOT NULL AND contract_start_date != ''
                GROUP BY month
                ORDER BY month
            """)
            rental_series = []
            for row in cursor.fetchall():
                if row[0] and len(row[0]) >= 7:
                    rental_series.append({
                        "month": row[0],
                        "contracts": row[1] or 0,
                        "avg_rent": round(row[2] or 0),
                    })
            trends["rental_series"] = rental_series
    except Exception:
        pass

    return trends


def _generate_local_insights(
    query: str, sql: str, columns: List[str], data: List[Dict], row_count: int
) -> Dict:
    """Synchronous local insight generation (runs in thread pool)."""
    from collections import Counter, defaultdict

    insights = []
    stats = {}
    row_count = row_count or len(data)

    # ---- Columns to SKIP (IDs, coordinates, technical fields) ----
    _SKIP_PATTERNS = {
        'id', 'idx', 'index', 'pk', 'key', 'hash', 'uuid',
        'latitude', 'longitude', 'lat', 'lng', 'lon',
        'created_at', 'updated_at', 'timestamp', 'date_insert',
        'objectid', 'fid', 'gml_id', 'ogc_fid',
    }
    _SKIP_SUFFIXES = ('_id', '_pk', '_key', '_hash', '_uuid', '_idx')

    def _is_junk_col(col_name: str, vals) -> bool:
        """Return True if column is an ID, coordinate, or technical field."""
        cl = col_name.lower().strip()
        if cl in _SKIP_PATTERNS:
            return True
        if cl.endswith(_SKIP_SUFFIXES):
            return True
        if cl.startswith(('id_', 'pk_', 'fk_')):
            return True
        # If nearly all values are unique, it's likely an ID
        unique_ratio = len(set(str(v) for v in vals if v is not None)) / max(len(vals), 1)
        if unique_ratio > 0.95 and len(vals) > 10:
            return True
        return False

    # Column display names (human-readable)
    def _nice_name(col):
        return col.replace('_', ' ').replace('  ', ' ').strip().title()

    # Format large numbers nicely
    def _fmt(v):
        if abs(v) >= 1e6: return f"AED {v/1e6:.1f}M"
        if abs(v) >= 1e3: return f"AED {v/1e3:,.0f}K" if v > 100 else f"{v:,.0f}"
        return f"{v:,.0f}"

    # ---- Classify columns ----
    # Known business-meaningful columns by category
    _PRICE_COLS = {'actual_worth', 'transaction_amount', 'meter_sale_price', 'annual_amount',
                   'listprice', 'price', 'soldprice', 'meter_rent_price', 'worth'}
    _AREA_COLS = {'area_name_en', 'area_name', 'location_area', 'community', 'nearest_metro_en'}
    _TYPE_COLS = {'property_type_en', 'property_sub_type_en', 'usage_en', 'rooms_en',
                  'completion_status', 'trans_group_en', 'type', 'category', 'status'}

    price_col = None
    area_col = None
    type_cols = []
    numeric_cols = []

    for c in columns:
        cl = c.lower()
        if cl in _PRICE_COLS and not price_col:
            price_col = c
        if cl in _AREA_COLS and not area_col:
            area_col = c
        if cl in _TYPE_COLS:
            type_cols.append(c)

    # ---- 1. Numeric analysis (prices, sizes — NOT IDs) ----
    for col in columns:
        vals = [row[col] for row in data if row.get(col) is not None]
        if not vals or _is_junk_col(col, vals):
            continue
        numeric_vals = []
        for v in vals:
            try:
                numeric_vals.append(float(v))
            except (ValueError, TypeError):
                continue
        if len(numeric_vals) < 5:
            continue

        numeric_cols.append(col)
        mean_val = statistics.mean(numeric_vals)
        median_val = statistics.median(numeric_vals)
        std_val = statistics.stdev(numeric_vals) if len(numeric_vals) > 1 else 0
        min_val = min(numeric_vals)
        max_val = max(numeric_vals)
        name = _nice_name(col)

        stats[col] = {"mean": round(mean_val, 2), "median": round(median_val, 2),
                       "std": round(std_val, 2), "min": round(min_val, 2), "max": round(max_val, 2)}

        # Price-specific insights
        is_price = col.lower() in _PRICE_COLS

        # Skew: mean far from median
        if mean_val > 0 and abs(mean_val - median_val) / mean_val > 0.3:
            if is_price:
                insights.append({
                    "type": "pattern", "icon": "📊", "severity": "medium",
                    "title": f"Prices are skewed — average misleading",
                    "detail": f"Average price ({_fmt(mean_val)}) is much {'higher' if mean_val > median_val else 'lower'} than the typical price ({_fmt(median_val)}). A few {'premium' if mean_val > median_val else 'bargain'} listings pull the average. Use median for a fairer picture.",
                })
            elif mean_val > 100:  # Only for meaningful numbers
                insights.append({
                    "type": "pattern", "icon": "📊", "severity": "low",
                    "title": f"{name} distribution is uneven",
                    "detail": f"Average ({mean_val:,.0f}) differs from median ({median_val:,.0f}). Some extreme values are skewing the numbers.",
                })

        # Outliers in prices
        if std_val > 0 and is_price:
            outliers = [v for v in numeric_vals if abs(v - mean_val) > 2 * std_val]
            if outliers and 0 < len(outliers) <= len(numeric_vals) * 0.1:
                top_outlier = max(outliers)
                insights.append({
                    "type": "anomaly", "icon": "⚠️", "severity": "high",
                    "title": f"{len(outliers)} unusually priced transaction{'s' if len(outliers)>1 else ''}",
                    "detail": f"Most prices cluster around {_fmt(median_val)}, but {len(outliers)} are significantly above at up to {_fmt(top_outlier)}. Worth investigating — could be bulk deals or premium units.",
                })

        # Price range insight
        if is_price and max_val > min_val and min_val > 0:
            ratio = max_val / min_val
            if ratio > 5:
                insights.append({
                    "type": "comparison", "icon": "💰", "severity": "high",
                    "title": f"{ratio:.0f}x price range in results",
                    "detail": f"From {_fmt(min_val)} to {_fmt(max_val)}. This wide spread suggests very different property segments are mixed together.",
                })

    # ---- 2. Area-price comparison ----
    if price_col and area_col:
        area_prices = defaultdict(list)
        for row in data:
            a, p = row.get(area_col), row.get(price_col)
            if a and p:
                try: area_prices[str(a)].append(float(p))
                except: pass
        if len(area_prices) > 1:
            area_avgs = {a: statistics.mean(ps) for a, ps in area_prices.items() if len(ps) >= 2}
            if len(area_avgs) > 1:
                sorted_areas = sorted(area_avgs.items(), key=lambda x: x[1], reverse=True)
                top_area, top_avg = sorted_areas[0]
                bot_area, bot_avg = sorted_areas[-1]
                if bot_avg > 0:
                    ratio = top_avg / bot_avg
                    if ratio > 1.5:
                        insights.append({
                            "type": "comparison", "icon": "📍", "severity": "high",
                            "title": f"{top_area} is {ratio:.1f}x pricier than {bot_area}",
                            "detail": f"Avg {_fmt(top_avg)} vs {_fmt(bot_avg)}. {'Consider whether you want to compare like-for-like.' if ratio > 3 else 'Significant pricing difference between these areas.'}",
                        })
        # Single area — show price summary
        elif len(area_prices) == 1:
            area_name = list(area_prices.keys())[0]
            prices = list(area_prices.values())[0]
            if len(prices) >= 3:
                med = statistics.median(prices)
                insights.append({
                    "type": "info", "icon": "🏘️", "severity": "low",
                    "title": f"All results in {area_name}",
                    "detail": f"Typical price: {_fmt(med)} (from {len(prices)} transactions). {'Narrow geographic focus.' if len(prices) > 10 else ''}",
                })

    # ---- 3. Property type breakdown ----
    for tc in type_cols[:2]:
        vals = [str(row[tc]) for row in data if row.get(tc) and str(row[tc]).strip()]
        if len(vals) < 3:
            continue
        counts = Counter(vals)
        total = len(vals)
        top_val, top_count = counts.most_common(1)[0]
        top_pct = top_count / total * 100

        if top_pct > 70 and len(counts) > 1:
            name = _nice_name(tc)
            insights.append({
                "type": "concentration", "icon": "📌", "severity": "medium",
                "title": f"{top_pct:.0f}% are {top_val}",
                "detail": f"Results heavily concentrated in one {name.lower()} category. Other types: {', '.join(v for v,_ in counts.most_common(4)[1:])}.",
            })
        elif len(counts) >= 3 and top_pct < 40:
            top3 = counts.most_common(3)
            breakdown = ', '.join(f"{v} ({c})" for v, c in top3)
            insights.append({
                "type": "diversity", "icon": "📋", "severity": "low",
                "title": f"Mix of {_nice_name(tc).lower()} types",
                "detail": f"Top categories: {breakdown}. Results span {len(counts)} different types.",
            })

    # ---- 4. LLM deep insight (only if we have a model and meaningful data) ----
    if gemini_model and len(data) >= 5:
        # Build compact summary — only business-relevant columns
        relevant_cols = [c for c in columns if not _is_junk_col(c, [row.get(c) for row in data[:10] if row.get(c)])]
        sample = [{k: row.get(k) for k in relevant_cols[:8]} for row in data[:8]]

        prompt = f"""Analyze this Dubai real estate query result. Be a sharp analyst.

QUESTION: "{query}"
RESULT: {row_count} rows with columns: {', '.join(relevant_cols[:10])}
STATS: {json.dumps(stats, default=str)[:800]}
SAMPLE: {json.dumps(sample, default=str)[:1200]}

Give exactly 2 insights a human would MISS. Each must be specific, use numbers from the data, and be actionable.
Focus on: pricing anomalies, area comparisons, investment signals, risk flags, or market timing.
Do NOT mention technical things like column names or IDs.

Return JSON: [{{"title":"short headline","detail":"1-2 specific sentences with numbers"}}]
JSON only, no markdown."""

        try:
            response = gemini_model.generate_content(prompt)
            text = response.text.strip()
            if '[' in text:
                text = text[text.index('['):text.rindex(']') + 1]
            for li in json.loads(text)[:2]:
                insights.append({
                    "type": "insight", "icon": "🧠", "severity": "high",
                    "title": li.get("title", ""), "detail": li.get("detail", ""),
                })
        except Exception as e:
            print(f"LLM local insight failed: {e}")

    # ---- 5. Deduplicate and rank ----
    # Remove low-value insights if we have enough good ones
    high_med = [i for i in insights if i["severity"] in ("high", "medium")]
    if len(high_med) >= 3:
        insights = high_med  # Drop low-severity noise

    # Cap at 5 insights max
    severity_order = {"high": 0, "medium": 1, "low": 2}
    insights.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 2))
    insights = insights[:5]

    n = len(insights)
    summary = f"{n} insight{'s' if n != 1 else ''}" if n > 0 else "No notable patterns."

    return {"insights": insights, "summary": summary, "stats": stats, "row_count": row_count}


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
# AI Smart Scraper Endpoints
# ============================================================================

# Import Smart Scraper (optional — requires selenium, not available on Render)
try:
    from scraper_layer import SmartScraper, ScrapeStrategy, SchemaGenerator
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False
    print("Scraper layer not available (selenium not installed) — scraper endpoints disabled")

# Global scraper instance
smart_scraper = None


def _get_scraper():
    """Get or create the smart scraper instance."""
    global smart_scraper
    if not SCRAPER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Scraper not available — selenium not installed on this deployment")
    if smart_scraper is None:
        smart_scraper = SmartScraper(
            llm_model=gemini_model,
            db_connection=get_db(),
            headless=True,
        )
    return smart_scraper


class ScrapeRequest(BaseModel):
    url: str
    description: str = ""
    strategy: Optional[str] = None  # auto, api_sniff, dom_parse, ai_analyzed, hybrid
    max_pages: int = 3
    load_to_db: bool = True


class ScrapeAnalyzeRequest(BaseModel):
    url: str


@app.post("/api/scraper/scrape")
async def scrape_url(request: ScrapeRequest):
    """
    AI-Powered Smart Scraper.

    Give it ANY URL → AI figures out how to extract data:
    1. Discovers hidden APIs via network interception
    2. Falls back to intelligent DOM parsing
    3. Uses AI to structure raw data into clean schemas
    4. Auto-loads into database for querying

    Example:
        POST /api/scraper/scrape
        {"url": "https://example.com/products", "description": "product listings"}
    """
    scraper = _get_scraper()

    # Map strategy string to enum
    strategy = None
    if request.strategy and request.strategy != "auto":
        strategy_map = {
            "api_sniff": ScrapeStrategy.API_SNIFF,
            "dom_parse": ScrapeStrategy.DOM_PARSE,
            "ai_analyzed": ScrapeStrategy.AI_ANALYZED,
            "hybrid": ScrapeStrategy.HYBRID,
            "direct_api": ScrapeStrategy.DIRECT_API,
        }
        strategy = strategy_map.get(request.strategy)

    try:
        result = scraper.scrape(
            url=request.url,
            description=request.description,
            strategy=strategy,
            max_pages=request.max_pages,
            load_to_db=request.load_to_db,
        )

        # If loaded to DB, update the schema registry
        if result.success and result.table_name and request.load_to_db:
            # Add to SCHEMA so it's available in the filter UI
            new_schema = {
                "name": result.table_name,
                "description": result.schema.get("description", f"Scraped from {request.url}"),
                "columns": [
                    {"name": col, "type": "TEXT"}
                    for col in result.columns
                ],
                "source": "scraper",
                "source_url": request.url,
            }
            # Check if table already exists in schema
            existing = [t for t in SCHEMA if t["name"] == result.table_name]
            if existing:
                SCHEMA.remove(existing[0])
            SCHEMA.append(new_schema)

        return result.to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


@app.post("/api/scraper/analyze")
async def analyze_url(request: ScrapeAnalyzeRequest):
    """
    Pre-analyze a URL before scraping.

    Returns:
    - Predicted site type
    - Expected data fields
    - Recommended strategy
    - Anti-scraping assessment
    """
    scraper = _get_scraper()

    try:
        analysis = scraper.analyze_url(request.url)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/scraper/tables")
async def list_scraped_tables():
    """List all tables that were created by the scraper."""
    scraped = [t for t in SCHEMA if t.get("source") == "scraper"]
    db = get_db()

    tables_info = []
    for table in scraped:
        try:
            cursor = db.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table['name']}")
            count = cursor.fetchone()[0]
            tables_info.append({
                "table_name": table["name"],
                "description": table.get("description", ""),
                "source_url": table.get("source_url", ""),
                "row_count": count,
                "columns": [c["name"] for c in table.get("columns", [])],
            })
        except Exception:
            pass

    return {"scraped_tables": tables_info, "total": len(tables_info)}


@app.get("/api/scraper/preview/{table_name}")
async def preview_scraped_data(table_name: str, limit: int = 20):
    """Preview data from a scraped table."""
    db = get_db()

    # Verify table exists and was scraped
    scraped_tables = [t["name"] for t in SCHEMA if t.get("source") == "scraper"]
    if table_name not in scraped_tables:
        raise HTTPException(status_code=404, detail=f"Scraped table '{table_name}' not found")

    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", db)
        return {
            "table_name": table_name,
            "columns": list(df.columns),
            "data": df.to_dict(orient="records"),
            "row_count": len(df),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/scraper/table/{table_name}")
async def delete_scraped_table(table_name: str):
    """Delete a scraped table from the database."""
    db = get_db()

    scraped_tables = [t for t in SCHEMA if t.get("source") == "scraper" and t["name"] == table_name]
    if not scraped_tables:
        raise HTTPException(status_code=404, detail=f"Scraped table '{table_name}' not found")

    try:
        db.execute(f"DROP TABLE IF EXISTS {table_name}")
        SCHEMA.remove(scraped_tables[0])
        return {"status": "deleted", "table_name": table_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Map Data Endpoints
# ============================================================================

# Load pre-scraped Dubai area coordinates (from Google Maps Selenium scraper)
AREA_COORDINATES_FILE = Path(__file__).parent / "dubai_areas_coordinates.json"
AREA_COORDINATES = []
if AREA_COORDINATES_FILE.exists():
    try:
        with open(AREA_COORDINATES_FILE, "r") as f:
            AREA_COORDINATES = json.load(f)
        print(f"Loaded {len(AREA_COORDINATES)} area coordinates from {AREA_COORDINATES_FILE.name}")
    except Exception as e:
        print(f"Warning: Could not load area coordinates: {e}")

# Load Dubai area polygon boundaries
AREA_POLYGONS_FILE = Path(__file__).parent / "dubai_area_polygons.json"
AREA_POLYGONS = []
if AREA_POLYGONS_FILE.exists():
    try:
        with open(AREA_POLYGONS_FILE, "r") as f:
            AREA_POLYGONS = json.load(f)
        print(f"Loaded {len(AREA_POLYGONS)} area polygons from {AREA_POLYGONS_FILE.name}")
    except Exception as e:
        print(f"Warning: Could not load area polygons: {e}")

# Load real community polygon boundaries from GeoJSON (parsed from KML)
COMMUNITY_GEOJSON_FILE = Path(__file__).parent / "data_variable" / "community_polygons.geojson"
COMMUNITY_GEOJSON = None
if COMMUNITY_GEOJSON_FILE.exists():
    try:
        with open(COMMUNITY_GEOJSON_FILE, "r") as f:
            COMMUNITY_GEOJSON = json.load(f)
        print(f"Loaded {len(COMMUNITY_GEOJSON.get('features', []))} community polygons from KML")
    except Exception as e:
        print(f"Warning: Could not load community GeoJSON: {e}")


_map_data_cache = {}  # in-memory cache for map data response

@app.get("/api/map_data")
async def get_map_data():
    """
    Get all geographic data for the interactive map.

    Returns layers:
    - areas: Dubai area/district coordinates (from Google Maps scraper)
    - metro: Metro station locations (from RTA data)
    - landmarks: Major landmarks
    - transactions: Bayut transactions with lat/lon (sampled)
    """
    # Return cached response if available (GeoJSON doesn't change at runtime)
    if _map_data_cache.get("response"):
        return _map_data_cache["response"]

    db = get_db()
    layers = {}

    # Layer 1: Area polygons — use REAL community boundaries from KML if available
    if COMMUNITY_GEOJSON and COMMUNITY_GEOJSON.get("features"):
        layers["areas"] = {
            "label": "Dubai Areas",
            "color": "#6366f1",
            "type": "geojson",
            "geojson": COMMUNITY_GEOJSON,
        }
    else:
        # No GeoJSON available — empty layer (no more scraper squares)
        layers["areas"] = {
            "label": "Dubai Areas",
            "color": "#6366f1",
            "type": "polygon",
            "data": [],
        }

    # Layer 2: Landmarks (point markers only — no square polygons)
    landmarks = [a for a in AREA_COORDINATES if a.get("category") in ("landmark", "commercial")]
    layers["landmarks"] = {
        "label": "Landmarks",
        "color": "#f59e0b",
        "type": "mixed",
        "data": landmarks,
        "polygons": [],   # removed old scraper squares; real boundaries come from GeoJSON areas layer
    }

    # Layer 3: Metro stations (from DB)
    try:
        metro_df = pd.read_sql_query(
            """SELECT location_name_english as name,
                      station_location_latitude as lat,
                      station_location_longitude as lng,
                      line_name
               FROM metro_stations
               WHERE station_location_latitude IS NOT NULL
                 AND station_location_longitude IS NOT NULL""",
            db,
        )
        metro_data = json.loads(metro_df.to_json(orient="records"))
        layers["metro"] = {
            "label": "Metro Stations",
            "color": "#10b981",
            "data": metro_data,
        }
    except Exception:
        layers["metro"] = {"label": "Metro Stations", "color": "#10b981", "data": []}

    # Layer 4: Transport (airports etc)
    transport = [a for a in AREA_COORDINATES if a.get("category") == "transport"]
    layers["transport"] = {
        "label": "Transport",
        "color": "#06b6d4",
        "data": transport,
    }

    result = {"layers": layers, "center": {"lat": 25.2048, "lng": 55.2708}, "zoom": 11}
    _map_data_cache["response"] = result
    return result


@app.get("/api/map_data/transactions")
async def get_map_transactions(
    area: Optional[str] = Query(None, description="Filter by area name"),
    limit: int = Query(500, ge=1, le=2000),
):
    """
    Get Bayut transactions with lat/lon for map plotting.
    These are actual property transactions with geographic coordinates.
    """
    db = get_db()

    try:
        where_clause = "WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND latitude != 0 AND longitude != 0"
        params = []
        if area:
            where_clause += " AND location_area LIKE ?"
            params.append(f"%{area}%")

        # Use rowid sampling instead of ORDER BY RANDOM() (much faster on large tables)
        query = f"""
            SELECT location_area as name,
                   location_sub_area,
                   location_building,
                   latitude as lat,
                   longitude as lng,
                   transaction_amount,
                   transaction_category,
                   beds,
                   builtup_area_sqm,
                   completion_status,
                   developer_name
            FROM bayut_transactions
            {where_clause}
            LIMIT {limit}
        """
        df = pd.read_sql_query(query, db, params=params if params else None)
        data = json.loads(df.to_json(orient="records"))

        return {
            "label": "Property Transactions",
            "color": "#ef4444",
            "data": data,
            "count": len(data),
        }
    except Exception as e:
        return {"label": "Property Transactions", "color": "#ef4444", "data": [], "count": 0, "error": str(e)}


@app.post("/api/map_data/query_results")
async def get_map_query_results(request: ExecuteQueryRequest):
    """
    Execute a SQL query and extract rows with lat/lon for map plotting.
    Automatically detects latitude/longitude columns in the result set.
    """
    db = get_db()

    try:
        sql = request.sql
        if "LIMIT" not in sql.upper():
            sql += f" LIMIT {request.limit}"

        df = pd.read_sql_query(sql, db)

        # Detect lat/lon columns
        lat_col = None
        lng_col = None
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ("latitude", "lat", "station_location_latitude"):
                lat_col = col
            elif col_lower in ("longitude", "lng", "lon", "station_location_longitude"):
                lng_col = col

        if not lat_col or not lng_col:
            return {"data": [], "count": 0, "message": "No lat/lon columns found in query results"}

        # Filter to rows with valid coordinates
        geo_df = df[df[lat_col].notna() & df[lng_col].notna()].copy()
        geo_df = geo_df[(geo_df[lat_col] != 0) & (geo_df[lng_col] != 0)]

        # Rename to standard lat/lng
        geo_df = geo_df.rename(columns={lat_col: "lat", lng_col: "lng"})

        # Add a display name from available columns
        name_candidates = ["name", "location_area", "location_name_english", "area_name_en", "building_name_en", "school_name", "facility_name"]
        name_col = None
        for candidate in name_candidates:
            if candidate in geo_df.columns:
                name_col = candidate
                break
        if name_col and name_col != "name":
            geo_df["name"] = geo_df[name_col]

        data = json.loads(geo_df.to_json(orient="records"))

        return {
            "data": data,
            "count": len(data),
            "lat_column": lat_col,
            "lng_column": lng_col,
        }
    except Exception as e:
        return {"data": [], "count": 0, "error": str(e)}


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


@app.get("/favicon.ico")
async def favicon():
    """Return empty favicon to suppress browser 404."""
    from fastapi.responses import Response
    # 1x1 transparent PNG
    return Response(
        content=b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82',
        media_type="image/png",
    )


@app.get("/")
async def read_root():
    """Serve the main HTML file."""
    return FileResponse(Path(__file__).parent / "index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
