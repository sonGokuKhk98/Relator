"""
Smart Extractor with Multi-Strategy Fallback

Inspired by Netflix's approach of combining LLM capabilities with
deterministic fallback strategies based on confidence levels.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Callable
import json
import re
import time

from .filter_dsl import (
    FilterAST,
    FilterNode,
    FilterGroup,
    FilterOperator,
    OrderByClause,
    LogicalOperator,
)
from .validator import FilterValidator, ValidationResult, SchemaRegistry
from .entity_resolver import EntityResolver, Entity, EntityType
from .feedback_store import FeedbackStore


class ExtractionStrategy(Enum):
    """Extraction strategy used."""
    LLM_FULL = "llm_full"          # High confidence, use LLM result directly
    LLM_VALIDATED = "llm_validated" # Medium confidence, validated against schema
    HYBRID = "hybrid"               # Combined LLM + regex
    REGEX_ONLY = "regex_only"       # LLM failed or unavailable
    CACHED = "cached"               # Retrieved from cache
    FEEDBACK = "feedback"           # Retrieved from feedback overrides


@dataclass
class ExtractionResult:
    """Complete extraction result with metadata."""
    ast: FilterAST
    strategy_used: ExtractionStrategy
    confidence: float

    # Additional metadata
    llm_interpretation: Optional[str] = None
    entities_resolved: List[Entity] = field(default_factory=list)
    validation_result: Optional[ValidationResult] = None
    extraction_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "ast": self.ast.to_dict(),
            "strategy": self.strategy_used.value,
            "confidence": self.confidence,
            "interpretation": self.llm_interpretation,
            "entities": [e.to_dict() for e in self.entities_resolved],
            "validation": self.validation_result.to_dict() if self.validation_result else None,
            "extraction_time_ms": self.extraction_time_ms,
            "warnings": self.warnings,
        }


class PromptBuilder:
    """
    Build optimized prompts for LLM extraction.

    Uses Netflix-inspired structured prompting with:
    - Clear schema context
    - Value conversion rules
    - Example-based learning
    - Output format enforcement
    - Dynamic value lookups from database
    """

    def __init__(self, schema_definition: List[Dict], feedback_store: FeedbackStore = None, db_connection=None):
        self.schema = schema_definition
        self.feedback_store = feedback_store
        self.db = db_connection
        self._value_lookups = {}
        if db_connection:
            self._load_value_lookups()

    def _load_value_lookups(self):
        """Load distinct values from database for LLM reference."""
        if not self.db:
            return

        cursor = self.db.cursor()

        # Load distinct area names
        try:
            cursor.execute("SELECT DISTINCT area_name_en FROM transactions WHERE area_name_en IS NOT NULL AND area_name_en != '' ORDER BY area_name_en LIMIT 50")
            self._value_lookups['areas'] = [row[0] for row in cursor.fetchall()]
        except:
            self._value_lookups['areas'] = []

        # Load distinct property types
        try:
            cursor.execute("SELECT DISTINCT property_type_en FROM transactions WHERE property_type_en IS NOT NULL")
            self._value_lookups['property_types'] = [row[0] for row in cursor.fetchall()]
        except:
            self._value_lookups['property_types'] = []

        # Load distinct transaction groups
        try:
            cursor.execute("SELECT DISTINCT trans_group_en FROM transactions WHERE trans_group_en IS NOT NULL")
            self._value_lookups['transaction_types'] = [row[0] for row in cursor.fetchall()]
        except:
            self._value_lookups['transaction_types'] = []

        # Load distinct areas from lkp_areas for JOIN queries
        try:
            cursor.execute("SELECT DISTINCT name_en FROM lkp_areas WHERE name_en IS NOT NULL AND name_en != '' ORDER BY name_en LIMIT 50")
            self._value_lookups['lkp_areas'] = [row[0] for row in cursor.fetchall()]
        except:
            self._value_lookups['lkp_areas'] = []

    def _build_value_lookup_context(self) -> str:
        """Build value lookup context for prompt."""
        if not self._value_lookups:
            return ""

        lines = ["\n## Valid Values (use EXACT values, no LIKE):\n"]

        if self._value_lookups.get('areas'):
            areas = self._value_lookups['areas'][:30]  # Top 30
            lines.append(f"**Areas (area_name_en)**: {', '.join(areas)}")

        if self._value_lookups.get('property_types'):
            lines.append(f"**Property Types (property_type_en)**: {', '.join(self._value_lookups['property_types'])}")

        if self._value_lookups.get('transaction_types'):
            lines.append(f"**Transaction Types (trans_group_en)**: {', '.join(self._value_lookups['transaction_types'])}")

        return "\n".join(lines)

    def build_extraction_prompt(
        self,
        query: str,
        entities: List[Entity] = None,
        include_examples: bool = True,
    ) -> str:
        """Build comprehensive extraction prompt."""

        # Schema context
        schema_context = self._build_schema_context()

        # Entity context if pre-resolved
        entity_context = ""
        if entities:
            entity_context = "\n## Pre-Resolved Entities:\n"
            for e in entities:
                entity_context += f"- \"{e.raw_text}\" -> {e.entity_type.value}: {e.canonical_value} (confidence: {e.confidence:.0%})\n"

        # Examples
        examples = ""
        if include_examples:
            examples = self._build_examples()

        # Value lookups from database
        value_lookups = self._build_value_lookup_context()

        prompt = f"""You are a precise SQL filter extraction system for Dubai real estate data. Convert natural language queries into structured filter specifications.

## Schema Reference
{schema_context}

## Table Selection Guide (CRITICAL - READ CAREFULLY):

### Primary Data Tables:
- **transactions**: DLD property transactions (sales, mortgages, gifts) - use for transaction/sale queries
- **rent_contracts**: Ejari rental contracts - use for rent/lease/tenant queries
- **units**: Property units registry - use for unit/apartment/property details
- **land_registry**: Land parcels - use for land/parcel queries
- **bayut_transactions**: Bayut market data - use for market analysis, developer info, ready/off-plan status
- **developers**: Developer information - use when asking about developers
- **brokers**: Broker information - use when asking about brokers/agents

### Lookup Tables (for JOINs - include when related data is needed):
- **lkp_areas**: Area/district names - include when query mentions "area info", "area details", "location names", or needs area_name_en
- **lkp_transaction_groups**: Transaction type groups - include when query needs transaction type details
- **lkp_transaction_procedures**: Transaction procedures - include when query needs procedure details

### When to Use Multiple Tables (JOINs):
- If query asks for "with area info/details" -> add lkp_areas
- If query asks for "with transaction types" -> add lkp_transaction_groups
- If query asks for "with procedure details" -> add lkp_transaction_procedures
- If query asks for "with developer info" -> add developers
- If query mentions "property details" with rent -> use rent_contracts + related lookups
- If query asks for data "across" or "combined" or "joined" -> use multiple tables
- If query asks for "related" data -> include the related lookup tables

## Valid Operators:
- Comparison: =, !=, >, >=, <, <=
- Pattern: LIKE (use % for wildcards)
- Range: BETWEEN (for numeric ranges)
- List: IN (for multiple values)

## Value Conversion Rules:
- Prices (AED): "5M" -> 5000000, "2 million" -> 2000000, "under 3M" -> 3000000
- Rooms: "2 bedroom" -> rooms_en = "2 B/R", "studio" -> rooms_en = "Studio"
- **AREAS (IMPORTANT)**: Use EXACT area names with = operator, NEVER use LIKE for areas:
  - "Dubai Marina" or "marina" -> area_name_en = "Marsa Dubai"
  - "Downtown" or "downtown dubai" -> area_name_en = "Burj Khalifa"
  - "Palm" or "palm jumeirah" -> area_name_en = "Palm Jumeirah"
  - "Mirdif" -> area_name_en = "Mirdif"
  - "Business Bay" -> area_name_en = "Business Bay"
- Transaction types: "sales" -> trans_group_en = "Sales", "mortgage" -> trans_group_en = "Mortgages"
- Property types: "villa" -> property_type_en = "Villa", "apartment"/"flat" -> property_type_en = "Unit"
- Status (Bayut): "ready" -> completion_status = "Ready", "off-plan" -> completion_status = "Off Plan"
{value_lookups}
{entity_context}
## User Query:
"{query}"

## Output Format (STRICT JSON):
{{
  "tables": ["primary_table", "lookup_table1", "lookup_table2"],
  "filters": [
    {{"table": "string", "column": "string", "operator": "string", "value": "string_or_number"}}
  ],
  "order_by": {{"column": "string", "direction": "ASC|DESC"}} | null,
  "confidence": 0.0-1.0,
  "interpretation": "Brief explanation of query interpretation"
}}
{examples}
## Critical Rules:
1. ONLY reference tables and columns from the schema above
2. Use correct data types (numbers for numeric columns)
3. **TABLE SELECTION IS CRITICAL**:
   - "transactions" for DLD sales/mortgage/gift transactions
   - "rent_contracts" for rental/lease/Ejari queries
   - "units" for property unit details
   - "bayut_transactions" for market data with developer info
   - Include lookup tables (lkp_areas, lkp_transaction_groups, etc.) when query asks for related/detailed info
4. **ALWAYS include multiple tables when**:
   - Query asks for "with [X] info/details"
   - Query mentions joining or combining data
   - Query needs data from related entities (areas, transaction types, developers)
5. Never guess values - only extract what's explicitly stated
6. Set confidence < 0.7 if interpretation is uncertain
7. For sorting: "cheapest" -> ASC, "most expensive" -> DESC
8. **NEVER use LIKE for area names** - always use exact match (=) with correct area name
9. **Only ONE filter per column** - do not create multiple filters for the same column

Return ONLY the JSON object, no markdown, no explanation."""

        return prompt

    def _build_schema_context(self) -> str:
        """Build schema context string."""
        lines = []
        for table in self.schema:
            cols = []
            for col in table.get("columns", []):
                col_name = col["name"] if isinstance(col, dict) else col
                col_type = col.get("type", "TEXT") if isinstance(col, dict) else "TEXT"
                cols.append(f"{col_name} ({col_type})")

            lines.append(f"### {table['name']}")
            if table.get("description"):
                lines.append(f"  {table['description']}")
            lines.append(f"  Columns: {', '.join(cols[:10])}")
            lines.append("")

        return "\n".join(lines)

    def _build_examples(self) -> str:
        """Build few-shot examples for Dubai Proptech data."""
        base_examples = """
## Examples:

### Single Table Queries:

Query: "2 bedroom apartments in Business Bay under 2 million"
{{
  "tables": ["transactions"],
  "filters": [
    {{"table": "transactions", "column": "rooms_en", "operator": "LIKE", "value": "%2%"}},
    {{"table": "transactions", "column": "area_name_en", "operator": "=", "value": "Business Bay"}},
    {{"table": "transactions", "column": "actual_worth", "operator": "<", "value": 2000000}}
  ],
  "order_by": null,
  "confidence": 0.95,
  "interpretation": "2 bedroom properties in Business Bay under AED 2M"
}}

Query: "villas over 5 million AED"
{{
  "tables": ["transactions"],
  "filters": [
    {{"table": "transactions", "column": "property_type_en", "operator": "=", "value": "Villa"}},
    {{"table": "transactions", "column": "actual_worth", "operator": ">", "value": 5000000}}
  ],
  "order_by": null,
  "confidence": 0.85,
  "interpretation": "Villa transactions over AED 5 million"
}}

Query: "ready properties by EMAAR"
{{
  "tables": ["bayut_transactions"],
  "filters": [
    {{"table": "bayut_transactions", "column": "completion_status", "operator": "=", "value": "Ready"}},
    {{"table": "bayut_transactions", "column": "developer_name", "operator": "LIKE", "value": "%EMAAR%"}}
  ],
  "order_by": null,
  "confidence": 0.88,
  "interpretation": "Ready/completed properties developed by EMAAR"
}}

### Multi-Table Queries (JOINs) - IMPORTANT:

Query: "Get rent contracts with property details, area info, and related project data"
{{
  "tables": ["rent_contracts", "lkp_areas"],
  "filters": [],
  "order_by": null,
  "confidence": 0.92,
  "interpretation": "Rental contracts joined with area lookup for location details"
}}

Query: "Show all rental contracts with area names"
{{
  "tables": ["rent_contracts", "lkp_areas"],
  "filters": [],
  "order_by": null,
  "confidence": 0.9,
  "interpretation": "Rent contracts with area information from lookup table"
}}

Query: "transactions with area details and transaction type information"
{{
  "tables": ["transactions", "lkp_areas", "lkp_transaction_groups"],
  "filters": [],
  "order_by": null,
  "confidence": 0.9,
  "interpretation": "Property transactions joined with area and transaction type lookups"
}}

Query: "sales transactions in Dubai Marina with area info"
{{
  "tables": ["transactions", "lkp_areas"],
  "filters": [
    {{"table": "transactions", "column": "trans_group_en", "operator": "=", "value": "Sales"}},
    {{"table": "lkp_areas", "column": "name_en", "operator": "=", "value": "Marsa Dubai"}}
  ],
  "order_by": null,
  "confidence": 0.92,
  "interpretation": "Sales transactions in Marsa Dubai (Dubai Marina) with area lookup"
}}

Query: "Show transactions with procedure details"
{{
  "tables": ["transactions", "lkp_transaction_procedures"],
  "filters": [],
  "order_by": null,
  "confidence": 0.88,
  "interpretation": "Transactions joined with procedure lookup for detailed procedure info"
}}

Query: "units registered in Downtown Dubai with area information"
{{
  "tables": ["units", "lkp_areas"],
  "filters": [
    {{"table": "lkp_areas", "column": "name_en", "operator": "LIKE", "value": "%Downtown%"}}
  ],
  "order_by": null,
  "confidence": 0.9,
  "interpretation": "Property units in Downtown Dubai with area details"
}}

Query: "rental contracts with tenant info and area details"
{{
  "tables": ["rent_contracts", "lkp_areas"],
  "filters": [],
  "order_by": null,
  "confidence": 0.88,
  "interpretation": "Rent contracts joined with area lookup"
}}

Query: "land parcels with area names"
{{
  "tables": ["land_registry", "lkp_areas"],
  "filters": [],
  "order_by": null,
  "confidence": 0.9,
  "interpretation": "Land registry joined with area lookup"
}}

Query: "transactions with all related lookup data"
{{
  "tables": ["transactions", "lkp_areas", "lkp_transaction_groups", "lkp_transaction_procedures"],
  "filters": [],
  "order_by": null,
  "confidence": 0.85,
  "interpretation": "Transactions with all related lookups for comprehensive view"
}}

Query: "mortgage transactions with area and procedure details"
{{
  "tables": ["transactions", "lkp_areas", "lkp_transaction_procedures"],
  "filters": [
    {{"table": "transactions", "column": "trans_group_en", "operator": "=", "value": "Mortgages"}}
  ],
  "order_by": null,
  "confidence": 0.9,
  "interpretation": "Mortgage transactions with area and procedure lookups"
}}

Query: "bayut transactions with area information"
{{
  "tables": ["bayut_transactions", "lkp_areas"],
  "filters": [],
  "order_by": null,
  "confidence": 0.88,
  "interpretation": "Bayut market data joined with area lookup"
}}

Query: "off-plan properties by Damac with location details"
{{
  "tables": ["bayut_transactions", "lkp_areas"],
  "filters": [
    {{"table": "bayut_transactions", "column": "completion_status", "operator": "=", "value": "Off Plan"}},
    {{"table": "bayut_transactions", "column": "developer_name", "operator": "LIKE", "value": "%Damac%"}}
  ],
  "order_by": null,
  "confidence": 0.9,
  "interpretation": "Off-plan Damac properties with area details"
}}

Query: "show me developers with their projects"
{{
  "tables": ["developers"],
  "filters": [],
  "order_by": null,
  "confidence": 0.85,
  "interpretation": "Developer information"
}}

Query: "brokers licensed in Dubai"
{{
  "tables": ["brokers"],
  "filters": [],
  "order_by": null,
  "confidence": 0.85,
  "interpretation": "Licensed real estate brokers"
}}

Query: "Get all areas in Dubai"
{{
  "tables": ["lkp_areas"],
  "filters": [],
  "order_by": null,
  "confidence": 0.95,
  "interpretation": "List of all Dubai areas from lookup table"
}}

Query: "transaction types and procedures"
{{
  "tables": ["lkp_transaction_groups", "lkp_transaction_procedures"],
  "filters": [],
  "order_by": null,
  "confidence": 0.9,
  "interpretation": "Transaction groups joined with procedures"
}}
"""
        if not self.feedback_store:
            return base_examples

        feedback_examples = self._build_feedback_examples()
        if not feedback_examples:
            return base_examples

        return base_examples + "\n" + feedback_examples

    def _build_feedback_examples(self) -> str:
        """Append user-provided feedback examples."""
        records = self.feedback_store.get_prompt_examples(limit=5)
        if not records:
            return ""

        blocks = ["## Feedback Examples:"]
        for rec in records:
            response = rec.to_llm_response()
            response_json = json.dumps(response, ensure_ascii=True)
            blocks.append(f'\nQuery: "{rec.query}"\n{response_json}')

        return "\n".join(blocks)


class RegexExtractor:
    """
    Regex-based filter extraction as fallback.

    Handles common patterns without requiring LLM.
    """

    # Price patterns
    PRICE_PATTERNS = [
        (r'\$?([\d,]+\.?\d*)\s*[mM](?:illion)?', 1000000),  # $1M, 1 million
        (r'\$?([\d,]+)\s*[kK]', 1000),  # $300k, 300K
        (r'\$(\d{1,3}(?:,\d{3})+)', 1),  # $1,000,000
        (r'\$([\d]+)', 1),  # $500000
        (r'(\d+)\s*(?:thousand)', 1000),  # 500 thousand
    ]

    # Keyword mappings for Dubai Proptech data
    KEYWORD_FILTERS = {
        # Property types
        "villa": ("transactions", "property_type_en", "=", "Villa"),
        "apartment": ("transactions", "property_type_en", "=", "Unit"),
        "flat": ("transactions", "property_type_en", "=", "Unit"),
        "land": ("transactions", "property_type_en", "=", "Land"),
        "building": ("transactions", "property_type_en", "=", "Building"),
        "commercial": ("transactions", "property_usage_en", "LIKE", "%Commercial%"),
        "residential": ("transactions", "property_usage_en", "=", "Residential"),

        # Transaction types
        "sales": ("transactions", "trans_group_en", "=", "Sales"),
        "sale": ("transactions", "trans_group_en", "=", "Sales"),
        "mortgage": ("transactions", "trans_group_en", "=", "Mortgages"),
        "gift": ("transactions", "trans_group_en", "=", "Gifts"),

        # Bayut status
        "ready": ("bayut_transactions", "completion_status", "=", "Ready"),
        "off-plan": ("bayut_transactions", "completion_status", "=", "Off Plan"),
        "offplan": ("bayut_transactions", "completion_status", "=", "Off Plan"),

        # Price ranges (AED)
        "affordable": ("transactions", "actual_worth", "<", 1500000),
        "expensive": ("transactions", "actual_worth", ">", 5000000),
        "luxury": ("transactions", "actual_worth", ">", 10000000),

        # Popular areas - using actual DB values
        # Note: Area resolution is handled by EntityResolver with proper aliases
        # "dubai marina" -> "Marsa Dubai", "downtown" -> "Burj Khalifa", etc.
        "palm jumeirah": ("transactions", "area_name_en", "=", "Palm Jumeirah"),
        "mirdif": ("transactions", "area_name_en", "=", "Mirdif"),
        "business bay": ("transactions", "area_name_en", "=", "Business Bay"),
    }

    # Keywords that trigger specific primary tables
    TABLE_KEYWORDS = {
        # Rent contracts keywords
        "rent": "rent_contracts",
        "rental": "rent_contracts",
        "lease": "rent_contracts",
        "tenant": "rent_contracts",
        "ejari": "rent_contracts",
        "rent contract": "rent_contracts",
        "rental contract": "rent_contracts",

        # Units keywords
        "unit": "units",
        "units": "units",
        "registered property": "units",
        "property unit": "units",

        # Land registry keywords
        "land parcel": "land_registry",
        "parcel": "land_registry",
        "land registry": "land_registry",

        # Bayut keywords
        "bayut": "bayut_transactions",
        "market data": "bayut_transactions",
        "developer": "bayut_transactions",

        # Developer keywords
        "developers": "developers",
        "developer list": "developers",

        # Broker keywords
        "broker": "brokers",
        "brokers": "brokers",
        "agent": "brokers",

        # Lookup table keywords
        "areas": "lkp_areas",
        "districts": "lkp_areas",
        "transaction types": "lkp_transaction_groups",
        "procedures": "lkp_transaction_procedures",
    }

    # Keywords that trigger additional tables for JOINs
    JOIN_KEYWORDS = {
        "area info": ["lkp_areas"],
        "area details": ["lkp_areas"],
        "area names": ["lkp_areas"],
        "location details": ["lkp_areas"],
        "with areas": ["lkp_areas"],
        "transaction type": ["lkp_transaction_groups"],
        "transaction types": ["lkp_transaction_groups"],
        "type details": ["lkp_transaction_groups"],
        "procedure details": ["lkp_transaction_procedures"],
        "with procedures": ["lkp_transaction_procedures"],
        "all details": ["lkp_areas", "lkp_transaction_groups", "lkp_transaction_procedures"],
        "related data": ["lkp_areas", "lkp_transaction_groups"],
        "full details": ["lkp_areas", "lkp_transaction_groups", "lkp_transaction_procedures"],
        "project data": ["lkp_areas"],
        "property details": ["lkp_areas"],
    }

    # Order patterns
    ORDER_PATTERNS = {
        "cheapest": ("actual_worth", "ASC"),
        "lowest price": ("actual_worth", "ASC"),
        "most expensive": ("actual_worth", "DESC"),
        "highest price": ("actual_worth", "DESC"),
        "newest": ("instance_date", "DESC"),
        "latest": ("instance_date", "DESC"),
        "largest": ("procedure_area", "DESC"),
        "biggest": ("procedure_area", "DESC"),
        "smallest": ("procedure_area", "ASC"),
    }

    def __init__(self, entity_resolver: EntityResolver = None):
        self.entity_resolver = entity_resolver

    def extract(self, query: str) -> FilterAST:
        """Extract filters using regex patterns for Dubai Proptech data."""
        query_lower = query.lower()
        filters = FilterGroup(LogicalOperator.AND)
        tables = set()
        order_by = None
        has_rooms = False

        # Step 1: Determine primary table based on keywords
        primary_table = "transactions"  # default
        for keyword, table in self.TABLE_KEYWORDS.items():
            if keyword in query_lower:
                primary_table = table
                break
        tables.add(primary_table)

        # Step 2: Check for JOIN keywords to add lookup tables
        for keyword, join_tables in self.JOIN_KEYWORDS.items():
            if keyword in query_lower:
                for t in join_tables:
                    tables.add(t)

        # Step 3: Extract filters from keywords
        for keyword, (table, column, op, value) in self.KEYWORD_FILTERS.items():
            if keyword in query_lower:
                # Only add filter if the table is relevant to the primary table
                # or if it's the transactions table (default)
                if table == primary_table or table == "transactions" or table in tables:
                    tables.add(table)
                    filters.add(FilterNode(
                        table=table,
                        column=column,
                        operator=FilterOperator.from_string(op),
                        value=value,
                        original_text=keyword,
                    ))

        # Extract bedrooms/rooms
        bed_match = re.search(r'(\d+)\+?\s*(?:bed|bedroom|br|b/r|room)', query_lower)
        if bed_match:
            rooms_value = bed_match.group(1)
            filters.add(FilterNode(
                table="transactions",
                column="rooms_en",
                operator=FilterOperator.LIKE,
                value=f"%{rooms_value}%",
                original_text=bed_match.group(0),
            ))
            has_rooms = True

        # Extract studio
        if "studio" in query_lower and not has_rooms:
            filters.add(FilterNode(
                table="transactions",
                column="rooms_en",
                operator=FilterOperator.LIKE,
                value="%Studio%",
                original_text="studio",
            ))
            has_rooms = True

        # Extract price (AED)
        price_value, price_context = self._extract_price(query_lower)
        if price_value:
            operator = "<" if any(w in query_lower for w in ["under", "less", "below", "max", "up to"]) else ">"
            filters.add(FilterNode(
                table="transactions",
                column="actual_worth",
                operator=FilterOperator.from_string(operator),
                value=price_value,
                original_text=price_context,
            ))

        # Extract area in sqm
        sqm_match = re.search(r'(\d+)\+?\s*(?:sqm|sq\.?m|square meter)', query_lower)
        if sqm_match:
            filters.add(FilterNode(
                table="transactions",
                column="procedure_area",
                operator=FilterOperator.GTE,
                value=float(sqm_match.group(1)),
                original_text=sqm_match.group(0),
            ))

        # Resolve entities (cities, states, etc.)
        if self.entity_resolver:
            entities = self.entity_resolver.resolve_all_entities(query)
            for entity in entities:
                if entity.confidence >= 0.7:
                    filters.add(FilterNode(
                        table=entity.table,
                        column=entity.column,
                        operator=FilterOperator.EQ,
                        value=entity.canonical_value,
                        original_text=entity.raw_text,
                        confidence=entity.confidence,
                    ))

        # Extract order
        for pattern, (column, direction) in self.ORDER_PATTERNS.items():
            if pattern in query_lower:
                order_by = OrderByClause(column=column, direction=direction, table="transactions")
                break

        return FilterAST(
            tables=list(tables),
            filters=filters,
            order_by=order_by,
            original_query=query,
            confidence=0.6,  # Regex extraction has lower confidence
        )

    def _extract_bed_bath_shorthand(self, query_lower: str) -> Optional[tuple]:
        """Parse realtor shorthand like 2b2b, 2/2, 3-2."""
        patterns = [
            r'(\d+)\s*(?:b|bd|br)\s*(\d+(?:\.\d+)?)\s*(?:ba|bath|bth|b)',
            r'(\d+)\s*[/x-]\s*(\d+(?:\.\d+)?)',
        ]
        context_tokens = {"bed", "bath", "bdrm", "bth", "condo", "home", "house", "townhome", "apartment", "apt"}
        has_context = any(tok in query_lower for tok in context_tokens)

        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if not match:
                continue
            if pattern == patterns[1] and not has_context:
                continue
            beds = int(match.group(1))
            baths = float(match.group(2))
            return beds, baths, match.group(0)

        return None

    def _extract_price(self, query: str) -> tuple:
        """Extract price value from query."""
        for pattern, multiplier in self.PRICE_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    val = float(match.group(1).replace(",", ""))
                    return int(val * multiplier), match.group(0)
                except ValueError:
                    continue
        return None, None


class SmartExtractor:
    """
    Netflix-inspired multi-strategy extraction system.

    Combines LLM extraction with validation and intelligent fallback.
    """

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    MEDIUM_CONFIDENCE_THRESHOLD = 0.60

    def __init__(
        self,
        llm_model,
        schema_definition: List[Dict],
        db_connection=None,
        cache=None,
    ):
        self.llm = llm_model
        self.schema_definition = schema_definition
        self.db = db_connection
        self.cache = cache

        # Initialize components
        self.feedback_store = FeedbackStore()
        self.prompt_builder = PromptBuilder(schema_definition, feedback_store=self.feedback_store, db_connection=db_connection)
        self.entity_resolver = EntityResolver(db_connection)
        self.regex_extractor = RegexExtractor(self.entity_resolver)
        self.validator = FilterValidator(
            SchemaRegistry.from_schema_definition(schema_definition)
        )

    def _apply_entities_to_ast(self, ast: FilterAST, entities: List[Entity]) -> FilterAST:
        """Inject resolved entities into the AST as filters + tables."""
        if not entities:
            return ast

        existing = set()
        for node in ast.filters.flatten():
            existing.add((node.table, node.column, node.operator.value, str(node.value)))

        for entity in entities:
            if not entity.table or not entity.column:
                continue
            key = (entity.table, entity.column, "=", str(entity.canonical_value))
            if key in existing:
                continue
            ast.filters.add(
                FilterNode(
                    table=entity.table,
                    column=entity.column,
                    operator=FilterOperator.EQ,
                    value=entity.canonical_value,
                    original_text=entity.raw_text,
                    confidence=entity.confidence,
                )
            )
            existing.add(key)

        for entity in entities:
            if entity.table and entity.table not in ast.tables:
                ast.tables.append(entity.table)

        return ast

    async def extract(self, query: str) -> ExtractionResult:
        """
        Extract filters using the best available strategy.

        Strategy selection based on:
        1. Cache availability
        2. LLM confidence
        3. Validation results
        """
        start_time = time.time()

        # Check feedback override first
        feedback = self.feedback_store.get_override(query)
        if feedback:
            feedback_ast = FilterAST.from_llm_response(feedback.to_llm_response(), query)
            validation = self.validator.validate(feedback_ast, query)
            if validation.valid:
                return ExtractionResult(
                    ast=feedback_ast,
                    strategy_used=ExtractionStrategy.FEEDBACK,
                    confidence=feedback.confidence,
                    llm_interpretation=feedback.interpretation,
                    entities_resolved=[],
                    validation_result=validation,
                )

        # Check cache first
        if self.cache:
            cached = self.cache.get(query)
            if cached:
                cached.strategy_used = ExtractionStrategy.CACHED
                return cached

        # Pre-resolve entities
        entities = self.entity_resolver.resolve_all_entities(query)

        # Try LLM extraction
        llm_result = None
        if self.llm:
            try:
                llm_result = await self._extract_with_llm(query, entities)
            except Exception as e:
                print(f"LLM extraction failed: {e}")

        # Determine strategy based on LLM result
        if llm_result and llm_result.get("confidence", 0) >= self.HIGH_CONFIDENCE_THRESHOLD:
            result = await self._process_high_confidence(query, llm_result, entities)
        elif llm_result and llm_result.get("confidence", 0) >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            result = await self._process_medium_confidence(query, llm_result, entities)
        else:
            result = self._process_regex_only(query, entities)

        result.extraction_time_ms = (time.time() - start_time) * 1000

        # Cache result
        if self.cache and result.confidence >= 0.7:
            self.cache.set(query, result)

        return result

    async def _extract_with_llm(
        self,
        query: str,
        entities: List[Entity],
    ) -> Optional[Dict]:
        """Call LLM for extraction."""
        prompt = self.prompt_builder.build_extraction_prompt(query, entities)

        try:
            response = self.llm.generate_content(prompt)
            response_text = response.text.strip()

            # Extract JSON from response
            response_text = self._extract_json(response_text)

            return json.loads(response_text)
        except Exception as e:
            print(f"LLM parsing failed: {e}")
            return None

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response (handle markdown blocks)."""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return text.strip()

    async def _process_high_confidence(
        self,
        query: str,
        llm_result: Dict,
        entities: List[Entity],
    ) -> ExtractionResult:
        """Process high confidence LLM result."""
        ast = FilterAST.from_llm_response(llm_result, query)
        ast = self._apply_entities_to_ast(ast, entities)

        # Validate
        validation = self.validator.validate(ast, query)

        if validation.valid:
            return ExtractionResult(
                ast=ast,
                strategy_used=ExtractionStrategy.LLM_FULL,
                confidence=llm_result.get("confidence", 0.9),
                llm_interpretation=llm_result.get("interpretation"),
                entities_resolved=entities,
                validation_result=validation,
            )
        else:
            # Validation failed - fall back to validated strategy
            return await self._process_medium_confidence(query, llm_result, entities)

    async def _process_medium_confidence(
        self,
        query: str,
        llm_result: Dict,
        entities: List[Entity],
    ) -> ExtractionResult:
        """Process medium confidence with hybrid approach."""
        llm_ast = FilterAST.from_llm_response(llm_result, query)
        regex_ast = self.regex_extractor.extract(query)

        # Merge results, preferring validated LLM filters
        merged_filters = FilterGroup(LogicalOperator.AND)
        llm_columns = set()

        # Add validated LLM filters
        for node in llm_ast.filters.flatten():
            if self._is_valid_filter(node):
                merged_filters.add(node)
                llm_columns.add(f"{node.table}.{node.column}")

        # Add regex filters for uncovered columns
        for node in regex_ast.filters.flatten():
            key = f"{node.table}.{node.column}"
            if key not in llm_columns:
                merged_filters.add(node)

        # Merge tables
        merged_tables = list(set(llm_ast.tables + regex_ast.tables))

        # Prefer LLM order_by
        order_by = llm_ast.order_by or regex_ast.order_by

        merged_ast = FilterAST(
            tables=merged_tables,
            filters=merged_filters,
            order_by=order_by,
            original_query=query,
            interpretation=llm_result.get("interpretation"),
            confidence=0.75,
        )
        merged_ast = self._apply_entities_to_ast(merged_ast, entities)

        validation = self.validator.validate(merged_ast, query)

        return ExtractionResult(
            ast=merged_ast,
            strategy_used=ExtractionStrategy.HYBRID,
            confidence=0.75,
            llm_interpretation=llm_result.get("interpretation"),
            entities_resolved=entities,
            validation_result=validation,
            warnings=["Used hybrid extraction due to medium confidence"],
        )

    def _process_regex_only(
        self,
        query: str,
        entities: List[Entity],
    ) -> ExtractionResult:
        """Process using regex only."""
        ast = self.regex_extractor.extract(query)
        ast = self._apply_entities_to_ast(ast, entities)
        validation = self.validator.validate(ast, query)

        return ExtractionResult(
            ast=ast,
            strategy_used=ExtractionStrategy.REGEX_ONLY,
            confidence=0.6,
            entities_resolved=entities,
            validation_result=validation,
            warnings=["LLM unavailable or failed, using regex extraction"],
        )

    def _is_valid_filter(self, node: FilterNode) -> bool:
        """Check if a single filter is valid against schema."""
        for table in self.schema_definition:
            if table["name"] == node.table:
                columns = [
                    c["name"] if isinstance(c, dict) else c
                    for c in table.get("columns", [])
                ]
                return node.column in columns
        return False

    def extract_sync(self, query: str) -> ExtractionResult:
        """Synchronous extraction - direct implementation without async."""
        start_time = time.time()

        # Check feedback override first
        feedback = self.feedback_store.get_override(query)
        if feedback:
            feedback_ast = FilterAST.from_llm_response(feedback.to_llm_response(), query)
            validation = self.validator.validate(feedback_ast, query)
            if validation.valid:
                return ExtractionResult(
                    ast=feedback_ast,
                    strategy_used=ExtractionStrategy.FEEDBACK,
                    confidence=feedback.confidence,
                    llm_interpretation=feedback.interpretation,
                    entities_resolved=[],
                    validation_result=validation,
                )

        # Check cache first
        if self.cache:
            cached = self.cache.get(query)
            if cached:
                cached.strategy_used = ExtractionStrategy.CACHED
                return cached

        # Pre-resolve entities
        entities = self.entity_resolver.resolve_all_entities(query)

        # Try LLM extraction
        llm_result = None
        if self.llm:
            try:
                llm_result = self._extract_with_llm_sync(query, entities)
            except Exception as e:
                print(f"LLM extraction failed: {e}")

        # Determine strategy based on LLM result
        if llm_result and llm_result.get("confidence", 0) >= self.HIGH_CONFIDENCE_THRESHOLD:
            result = self._process_high_confidence_sync(query, llm_result, entities)
        elif llm_result and llm_result.get("confidence", 0) >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            result = self._process_medium_confidence_sync(query, llm_result, entities)
        else:
            result = self._process_regex_only(query, entities)

        result.extraction_time_ms = (time.time() - start_time) * 1000

        # Deduplicate filters before returning
        result.ast = self._deduplicate_ast_filters(result.ast)

        # Cache result
        if self.cache and result.confidence >= 0.7:
            self.cache.set(query, result)

        return result

    def _deduplicate_ast_filters(self, ast: FilterAST) -> FilterAST:
        """Remove duplicate filters from AST."""
        seen = set()
        unique_filters = FilterGroup(LogicalOperator.AND)

        for node in ast.filters.flatten():
            # Create a key based on table, column, operator, and normalized value
            value_str = str(node.value).lower().strip() if node.value else ""
            key = (
                node.table.lower() if node.table else "",
                node.column.lower() if node.column else "",
                node.operator.value,
                value_str
            )
            if key not in seen:
                seen.add(key)
                unique_filters.add(node)

        ast.filters = unique_filters
        return ast

    def _extract_with_llm_sync(self, query: str, entities: List[Entity]) -> Optional[Dict]:
        """Synchronous LLM extraction."""
        prompt = self.prompt_builder.build_extraction_prompt(query, entities)

        try:
            response = self.llm.generate_content(prompt)
            response_text = response.text.strip()
            response_text = self._extract_json(response_text)
            return json.loads(response_text)
        except Exception as e:
            print(f"LLM parsing failed: {e}")
            return None

    def _process_high_confidence_sync(
        self,
        query: str,
        llm_result: Dict,
        entities: List[Entity],
    ) -> ExtractionResult:
        """Process high confidence LLM result (sync version)."""
        ast = FilterAST.from_llm_response(llm_result, query)
        ast = self._apply_entities_to_ast(ast, entities)
        validation = self.validator.validate(ast, query)

        if validation.valid:
            return ExtractionResult(
                ast=ast,
                strategy_used=ExtractionStrategy.LLM_FULL,
                confidence=llm_result.get("confidence", 0.9),
                llm_interpretation=llm_result.get("interpretation"),
                entities_resolved=entities,
                validation_result=validation,
            )
        else:
            return self._process_medium_confidence_sync(query, llm_result, entities)

    def _process_medium_confidence_sync(
        self,
        query: str,
        llm_result: Dict,
        entities: List[Entity],
    ) -> ExtractionResult:
        """Process medium confidence with hybrid approach (sync version)."""
        llm_ast = FilterAST.from_llm_response(llm_result, query)
        regex_ast = self.regex_extractor.extract(query)

        merged_filters = FilterGroup(LogicalOperator.AND)
        llm_columns = set()

        for node in llm_ast.filters.flatten():
            if self._is_valid_filter(node):
                merged_filters.add(node)
                llm_columns.add(f"{node.table}.{node.column}")

        for node in regex_ast.filters.flatten():
            key = f"{node.table}.{node.column}"
            if key not in llm_columns:
                merged_filters.add(node)

        merged_tables = list(set(llm_ast.tables + regex_ast.tables))
        order_by = llm_ast.order_by or regex_ast.order_by

        merged_ast = FilterAST(
            tables=merged_tables,
            filters=merged_filters,
            order_by=order_by,
            original_query=query,
            interpretation=llm_result.get("interpretation"),
            confidence=0.75,
        )
        merged_ast = self._apply_entities_to_ast(merged_ast, entities)

        validation = self.validator.validate(merged_ast, query)

        return ExtractionResult(
            ast=merged_ast,
            strategy_used=ExtractionStrategy.HYBRID,
            confidence=0.75,
            llm_interpretation=llm_result.get("interpretation"),
            entities_resolved=entities,
            validation_result=validation,
            warnings=["Used hybrid extraction due to medium confidence"],
        )
