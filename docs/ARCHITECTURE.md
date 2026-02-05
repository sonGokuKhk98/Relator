# Dynamic Filtering System: Complete Technical & Business Guide

## Executive Summary

This system converts **natural language queries** into **SQL filters** for a real estate database. A user types "3 bedroom condo in Austin under $500k" and instantly sees structured filters, an auto-generated SQL query, and results.

**Inspired by Netflix's engineering approach** to their internal data exploration tools, this system combines:
- LLM-powered extraction (when available)
- Deterministic fallback (always works)
- User-editable filters (full control)
- Feedback loop (continuous improvement)

---

## The Problem

### Business Problem
Real estate agents and analysts need to search property listings using natural language. Traditional approaches require:
- Learning SQL syntax
- Knowing exact column names
- Understanding database schema

### Technical Problem
Converting natural language to SQL is hard because:
- Ambiguity: "cheap" means different things to different people
- Shorthand: "2b2b" = 2 bedrooms, 2 bathrooms (realtor jargon)
- Entities: "Austin" could be a city, neighborhood, or street name
- Joins: User says "city" but data lives in `dim_cities` table

---

## Netflix's Approach (and How We Use It)

Netflix's engineering blog describes building a **Graph Search** system for their internal tools. Key insights:

| Netflix Concept | Our Implementation | Why It Matters |
|-----------------|-------------------|----------------|
| **AST-based filters** | `FilterAST`, `FilterNode`, `FilterGroup` | Composable, validatable, convertible |
| **Multi-strategy extraction** | LLM → Hybrid → Regex fallback | Always works, even without LLM |
| **@Mention entity resolution** | `EntityResolver` with fuzzy matching | Reduces ambiguity |
| **UI Chips** | `ChipGenerator` → visual chips | Users can see and edit filters |
| **Schema validation** | `FilterValidator` | Catch errors before SQL execution |
| **Prompt engineering** | `PromptBuilder` with examples | Better LLM accuracy |
| **Caching** | `ExtractionCache` | Reduce latency and cost |
| **Feedback loop** | `llm_feedback.jsonl` | Learn from failures |

---

## Complete User Flow: Step-by-Step

### Example Query
```
"3 bedroom 2 bathroom condo in NC under $400k"
```

---

## STEP 1: User Types Query

### What Happens
User types in the search box. After 500ms of no typing (debounce), the UI sends a request.

### Technical Details
```
File: index.html
Component: App
State: query = "3 bedroom 2 bathroom condo in NC under $400k"

useEffect(() => {
    const timer = setTimeout(() => {
        fetch(`${API_URL}/suggest_filters`, {
            method: "POST",
            body: JSON.stringify({ query })
        })
    }, 500);  // Debounce
}, [query]);
```

### Business Value
- No "Search" button needed — feels instant
- Debounce prevents API spam while typing

---

## STEP 2: API Receives Request

### What Happens
FastAPI endpoint receives the query and starts the extraction pipeline.

### Technical Details
```
File: app.py
Endpoint: POST /api/suggest_filters

@app.post("/api/suggest_filters")
async def suggest_filters(request: SuggestFiltersRequest):
    query = request.query  # "3 bedroom 2 bathroom condo in NC under $400k"
    
    # Bust cache for fresh results
    extraction_cache.invalidate(query)
    
    # Run extraction pipeline
    result = smart_extractor.extract_sync(query)
```

### Business Value
- Single entry point for all extraction logic
- Easy to monitor and debug

---

## STEP 3: Check Feedback Overrides

### What Happens
Before any ML/regex, we check if this exact query has a **known-good answer** stored in our feedback file.

### Technical Details
```
File: llm_layer/feedback_store.py
File: feedback/llm_feedback.jsonl

# Feedback file contains corrections:
{"query":"2b2b condo","tables":["fact_listings"],"filters":[...]}

# SmartExtractor checks first:
feedback = self.feedback_store.get_override(query)
if feedback:
    return ExtractionResult(
        strategy_used=ExtractionStrategy.FEEDBACK,
        ...
    )
```

### Example Feedback Entry
```json
{
  "query": "2b2b condo",
  "tables": ["fact_listings"],
  "filters": [
    {"table": "fact_listings", "column": "bedrooms", "operator": ">=", "value": 2},
    {"table": "fact_listings", "column": "bathrooms", "operator": ">=", "value": 2},
    {"table": "fact_listings", "column": "property_type", "operator": "LIKE", "value": "%Condo%"}
  ],
  "confidence": 0.95,
  "interpretation": "2 bed, 2 bath condos",
  "include_in_prompt": true
}
```

### Netflix Insight
> "Let humans correct the system, and use those corrections to improve future extractions."

### Business Value
- Instant fix for known failures
- No code deploy needed — just edit a JSON file
- Examples also improve LLM prompts

---

## STEP 4: Check Cache

### What Happens
If no feedback override, check if we've seen this query recently.

### Technical Details
```
File: llm_layer/cache.py

# Normalize query for better cache hits
"3 bedroom 2 bathroom condo in NC" 
→ "3 bedroom 2 bathroom condo in nc" (lowercased, whitespace collapsed)

# Check cache with TTL
cached = self.cache.get(query)
if cached:
    return cached  # Skip LLM/regex entirely
```

### Business Value
- Reduces LLM API costs ($$)
- Faster response for repeated queries
- 1-hour TTL balances freshness vs cost

---

## STEP 5: Resolve Entities

### What Happens
Before extraction, we identify **known entities** (cities, states, property types) to reduce ambiguity.

### Technical Details
```
File: llm_layer/entity_resolver.py

# Input: "3 bedroom 2 bathroom condo in NC under $400k"

# Step 5a: Check aliases
"condo" → ALIASES["condo"] = "Residential"
"NC" → matches STATE vocabulary

# Step 5b: Fuzzy match against vocabularies
vocabularies = {
    CITY: ["Austin", "Charlotte", "Raleigh", ...],
    STATE: ["NC", "TX", "FL", ...],
    PROPERTY_TYPE: ["Residential", "Commercial", ...],
    STATUS: ["A", "S", "P", ...]
}

# Step 5c: Map to dim tables
TYPE_MAPPING = {
    CITY: ("dim_cities", "city_name"),
    STATE: ("dim_states", "state_code"),
    PROPERTY_TYPE: ("dim_property_types", "property_type_name"),
    STATUS: ("dim_listing_status", "status_code"),
}

# Output: List of Entity objects
entities = [
    Entity(raw_text="condo", entity_type=PROPERTY_TYPE, 
           canonical_value="Residential", table="dim_property_types", 
           column="property_type_name", confidence=1.0),
    Entity(raw_text="NC", entity_type=STATE,
           canonical_value="NC", table="dim_states",
           column="state_code", confidence=1.0),
]
```

### Netflix Insight
> "Use @mentions to let users constrain input to known entities, reducing ambiguity."

### Business Value
- "NC" always means North Carolina, not a typo
- "condo" maps to correct property type
- Enables automatic JOINs to dimension tables

---

## STEP 6: LLM or Regex Extraction

### What Happens
Now we extract the actual filters. Two paths:

### Path A: LLM Extraction (if API key available)
```
File: llm_layer/smart_extractor.py

# Build prompt with schema context
prompt = """
You are a precise SQL filter extraction system.

## Schema Reference
### fact_listings
  Columns: listing_id, bedrooms, bathrooms, list_price, city, state_code...

## Value Conversion Rules:
- Prices: "300K" -> 300000, "$1M" -> 1000000
- Bedrooms: "3 bed" -> bedrooms >= 3
- Status: "active" -> "A"

## Pre-Resolved Entities:
- "condo" -> property_type: Residential
- "NC" -> state: NC

## User Query:
"3 bedroom 2 bathroom condo in NC under $400k"

## Output Format (JSON):
{
  "tables": ["fact_listings"],
  "filters": [...],
  "confidence": 0.0-1.0
}
"""

# Call Gemini
response = gemini_model.generate_content(prompt)
llm_result = json.loads(response.text)

# Check confidence
if llm_result["confidence"] >= 0.85:
    strategy = LLM_FULL
elif llm_result["confidence"] >= 0.60:
    strategy = HYBRID  # Combine with regex
else:
    strategy = REGEX_ONLY  # Fallback
```

### Path B: Regex Extraction (fallback)
```
File: llm_layer/smart_extractor.py → RegexExtractor

# Extract bedrooms
match = re.search(r'(\d+)\+?\s*(?:bed|bedroom|br)', query)
# "3 bedroom" → bedrooms >= 3

# Extract bathrooms
match = re.search(r'(\d+)\+?\s*(?:bath|bathroom|ba)', query)
# "2 bathroom" → bathrooms >= 2

# Extract realtor shorthand
match = re.search(r'(\d+)\s*[/x-]\s*(\d+)', query)
# "3/2" → bedrooms >= 3, bathrooms >= 2
# "2b2b" → bedrooms >= 2, bathrooms >= 2

# Extract price
match = re.search(r'\$?([\d,]+)\s*[kK]', query)
# "$400k" → 400000
# "under" → operator = "<"

# Keyword mappings
KEYWORD_FILTERS = {
    "active": ("fact_listings", "status", "=", "A"),
    "luxury": ("fact_listings", "list_price", ">", 1000000),
    "cheap": ("fact_listings", "list_price", "<", 250000),
}
```

### Netflix Insight
> "Use LLM for complex interpretation, but always have deterministic fallback."

### Business Value
- Works even without LLM (API down, no key, cost savings)
- Regex handles realtor jargon reliably
- LLM handles novel phrasing

---

## STEP 7: Build Filter AST

### What Happens
Combine extraction results into a structured **Abstract Syntax Tree**.

### Technical Details
```
File: llm_layer/filter_dsl.py

# Individual filter nodes
node1 = FilterNode(
    table="fact_listings",
    column="bedrooms",
    operator=FilterOperator.GTE,  # >=
    value=3,
    original_text="3 bedroom"
)

node2 = FilterNode(
    table="fact_listings",
    column="bathrooms",
    operator=FilterOperator.GTE,
    value=2,
    original_text="2 bathroom"
)

node3 = FilterNode(
    table="dim_property_types",
    column="property_type_name",
    operator=FilterOperator.EQ,
    value="Residential",
    original_text="condo"
)

node4 = FilterNode(
    table="dim_states",
    column="state_code",
    operator=FilterOperator.EQ,
    value="NC",
    original_text="NC"
)

node5 = FilterNode(
    table="fact_listings",
    column="list_price",
    operator=FilterOperator.LT,  # <
    value=400000,
    original_text="under $400k"
)

# Combine with AND
filter_group = FilterGroup(LogicalOperator.AND)
filter_group.add(node1)
filter_group.add(node2)
filter_group.add(node3)
filter_group.add(node4)
filter_group.add(node5)

# Full AST
ast = FilterAST(
    tables=["fact_listings", "dim_property_types", "dim_states"],
    filters=filter_group,
    order_by=None,
    original_query="3 bedroom 2 bathroom condo in NC under $400k"
)
```

### Netflix Insight
> "AST representation allows filters to be validated, transformed, and rendered independently."

### Business Value
- Same AST → SQL, UI chips, validation
- Easy to add/remove filters programmatically
- Supports complex nested conditions (AND/OR)

---

## STEP 8: Inject Entities into AST

### What Happens
Merge resolved entities (from Step 5) into the AST, ensuring dimension tables are used.

### Technical Details
```
File: llm_layer/smart_extractor.py

def _apply_entities_to_ast(self, ast, entities):
    for entity in entities:
        # Add filter for this entity
        ast.filters.add(FilterNode(
            table=entity.table,        # dim_states
            column=entity.column,      # state_code
            operator=FilterOperator.EQ,
            value=entity.canonical_value,  # NC
        ))
        
        # Ensure table is included
        if entity.table not in ast.tables:
            ast.tables.append(entity.table)
```

### Business Value
- Automatic JOINs to dimension tables
- Consistent entity handling across LLM and regex paths

---

## STEP 9: Validate AST

### What Happens
Check the AST against the database schema before generating SQL.

### Technical Details
```
File: llm_layer/validator.py

# Validation levels:
# 1. Syntactic: Is the structure correct?
# 2. Semantic: Do tables/columns exist?
# 3. Pragmatic: Are values reasonable?

validation = validator.validate(ast)

# Example checks:
- Does "fact_listings" exist? ✓
- Does "fact_listings.bedrooms" exist? ✓
- Is bedrooms an INTEGER? ✓
- Is value 3 valid for INTEGER? ✓

# Result
ValidationResult(
    valid=True,
    errors=[],
    warnings=[]
)
```

### Netflix Insight
> "Validate before execution to catch errors early and provide helpful feedback."

### Business Value
- No cryptic SQL errors for users
- Clear feedback on what went wrong
- Prevents injection attacks

---

## STEP 10: Generate UI Chips

### What Happens
Convert each FilterNode into a visual **chip** for the UI.

### Technical Details
```
File: llm_layer/chip_generator.py

# For each filter node:
def _chip_from_filter(self, node):
    # Determine chip type
    if node.column in ["city", "state_code", "neighborhood"]:
        chip_type = ChipType.ENTITY
        color = "blue"
    elif node.operator in [GTE, LTE, BETWEEN]:
        chip_type = ChipType.RANGE
        color = "green"
    elif node.operator in [EQ, NE]:
        chip_type = ChipType.COMPARISON
        color = "purple"
    
    # Generate display text
    if node.column == "bedrooms":
        display_text = f"Beds: {node.value}+"
    elif node.column == "list_price" and node.operator == LT:
        display_text = f"Price: < ${node.value/1000:.0f}K"
    ...
    
    return UIChip(
        id=f"{node.table}-{node.column}-{uuid}",
        display_text=display_text,
        chip_type=chip_type,
        color=color,
        filter_node=node,
        removable=True,
        editable=True
    )

# Output chips:
[
    UIChip(display_text="Beds: 3+", color="green"),
    UIChip(display_text="Baths: 2+", color="green"),
    UIChip(display_text="Property Type: Residential", color="purple"),
    UIChip(display_text="State: NC", color="blue"),
    UIChip(display_text="Price: < $400K", color="green"),
]
```

### Netflix Insight
> "Map AST filters to visual chips that users can see, understand, and modify."

### Business Value
- Users see exactly what was understood
- Can remove incorrect filters with one click
- Can edit values without re-typing query

---

## STEP 11: Log for Analysis

### What Happens
Every extraction is logged for quality monitoring and debugging.

### Technical Details
```
File: app.py → _append_analysis_log()
File: feedback/analysis_log.jsonl

# Log entry:
{
    "ts": "2024-02-01T17:53:19.163Z",
    "query": "3 bedroom 2 bathroom condo in NC under $400k",
    "detected_tables": ["fact_listings", "dim_property_types", "dim_states"],
    "filters": [...],
    "strategy": "regex_only",
    "confidence": 0.6,
    "validation": {"valid": true},
    "is_true": true
}
```

### Business Value
- Track extraction quality over time
- Identify common failure patterns
- Build feedback entries from failed queries

---

## STEP 12: Return Response to UI

### What Happens
API returns structured response with all extraction results.

### Technical Details
```
File: app.py

return SuggestFiltersResponse(
    detected_tables=["fact_listings", "dim_property_types", "dim_states"],
    extracted_filters=[...],  # Raw filter dicts
    chips=[...],              # UI chip dicts
    entities=[...],           # Resolved entities
    order_by=None,
    extraction_method="regex_only",
    confidence=0.6,
    interpretation="3+ bed, 2+ bath Residential in NC under $400K",
    validation={"valid": True},
    warnings=[]
)
```

---

## STEP 13: UI Updates State

### What Happens
React state updates with new tables, filters, and chips.

### Technical Details
```
File: index.html

// Update active tables (for graph visualization)
setActiveTables(prev => {
    const newTables = new Set(prev);
    data.detected_tables.forEach(t => newTables.add(t));
    return newTables;
});

// Update active filters (as chips)
setActiveFilters(prev => {
    const existingIds = new Set(prev.map(f => f.id));
    const newChips = data.chips.filter(c => !existingIds.has(c.id));
    return [...prev, ...newChips];
});
```

### Business Value
- Graph shows which tables are involved
- Users see chips immediately
- State is reactive — changes trigger SQL regeneration

---

## STEP 14: Generate SQL

### What Happens
UI sends filters to `/api/generate_sql` to build the actual query.

### Technical Details
```
File: app.py

# Step 14a: Remap fact_listings filters to dim tables
# This ensures JOINs happen even if extraction used fact_listings

def _remap_fact_filters(filters):
    mapping = {
        "city": ("dim_cities", "city_name"),
        "state_code": ("dim_states", "state_code"),
        "neighborhood": ("dim_neighborhoods", "neighborhood_name"),
        "property_type": ("dim_property_types", "property_type_name"),
        "status": ("dim_listing_status", "status_code"),
    }
    
    for filter in filters:
        if filter.table == "fact_listings" and filter.column in mapping:
            filter.table, filter.column = mapping[filter.column]

# Step 14b: Normalize and dedupe filters
# Multiple "status = A" becomes single filter
# Multiple "city = X" becomes "city IN (X, Y)"

def _normalize_filter_items(filters):
    # Group by (table, column)
    # Merge = operators into IN
    # Remove exact duplicates

# Step 14c: Merge tables
# Ensure all filter tables are in table list
# Ensure fact_listings is first (base table)

# Step 14d: Build JOIN path
# Use SCHEMA_EDGES to find multi-hop joins

SCHEMA_EDGES = [
    {"from": "fact_listings", "to": "dim_cities"},
    {"from": "fact_listings", "to": "dim_states"},
    {"from": "fact_listings", "to": "dim_property_types"},
    {"from": "dim_cities", "to": "dim_neighborhoods"},
    ...
]

def _build_join_clauses(base_table, target_tables):
    # BFS to find shortest path
    # Build JOIN ON conditions
    
JOIN_CONDITIONS = {
    ("fact_listings", "dim_states"): 
        "fact_listings.state_code = dim_states.state_code",
    ("fact_listings", "dim_property_types"): 
        "fact_listings.property_type = dim_property_types.property_type_name",
    ...
}
```

### Generated SQL
```sql
SELECT fact_listings.listing_id, 
       fact_listings.mls_number, 
       fact_listings.city, 
       fact_listings.state_code, 
       fact_listings.bedrooms, 
       fact_listings.bathrooms, 
       fact_listings.list_price,
       dim_property_types.property_type_name, 
       dim_states.state_code AS dim_states_state_code
FROM fact_listings
INNER JOIN dim_property_types 
    ON fact_listings.property_type = dim_property_types.property_type_name
INNER JOIN dim_states 
    ON fact_listings.state_code = dim_states.state_code
WHERE dim_property_types.property_type_name = 'Residential' 
  AND dim_states.state_code = 'NC' 
  AND fact_listings.bedrooms >= 3 
  AND fact_listings.bathrooms >= 2 
  AND fact_listings.list_price < 400000
LIMIT 100
```

### Business Value
- Automatic multi-table JOINs
- No duplicate columns (aliased)
- WHERE columns always in SELECT (visible in results)

---

## STEP 15: User Edits Filters (Optional)

### What Happens
User can click "Edit" on any chip to modify operator/value.

### Technical Details
```
File: index.html

// Toggle edit panel
const toggleFilterExpanded = (key) => {
    setExpandedFilters(prev => {
        const next = new Set(prev);
        next.has(key) ? next.delete(key) : next.add(key);
        return next;
    });
};

// Update filter value
const updateFilterAt = (idx, updater) => {
    setActiveFilters(prev => prev.map((item, i) => {
        if (i !== idx) return item;
        const updated = updater(item);
        // Also update chip display_text
        updated.display_text = buildDisplayText(updated.filter);
        return updated;
    }));
};
```

### Business Value
- Users can fine-tune without re-typing
- Changes trigger SQL regeneration
- Full control over final query

---

## STEP 16: Execute Query

### What Happens
User clicks "Run Query" → SQL executes against database.

### Technical Details
```
File: app.py

@app.post("/api/query")
async def execute_query(request: ExecuteQueryRequest):
    sql = request.sql
    
    # Execute on SQLite
    df = pd.read_sql_query(sql, DB_CONNECTION)
    
    return {
        "columns": list(df.columns),
        "data": df.to_dict(orient="records"),
        "row_count": len(df)
    }
```

### Business Value
- Real results from real data
- Column metadata for table rendering
- Row count for pagination

---

## STEP 17: Display Results

### What Happens
UI renders results in a table.

### Technical Details
```
File: index.html

// Results table component
{queryResults && (
    <table>
        <thead>
            <tr>
                {queryResults.columns.map(col => (
                    <th>{col}</th>
                ))}
            </tr>
        </thead>
        <tbody>
            {queryResults.data.map(row => (
                <tr>
                    {queryResults.columns.map(col => (
                        <td>{row[col]}</td>
                    ))}
                </tr>
            ))}
        </tbody>
    </table>
)}
```

### Business Value
- Immediate feedback on query results
- All filtered columns visible
- Ready for export/analysis

---

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                USER INPUT                                    │
│  "3 bedroom 2 bathroom condo in NC under $400k"                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            POST /api/suggest_filters                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Check feedback overrides (llm_feedback.jsonl)                           │
│  2. Check cache                                                              │
│  3. Resolve entities (city, state, property_type)                           │
│  4. Extract filters (LLM or Regex)                                          │
│  5. Build FilterAST                                                          │
│  6. Inject entities into AST                                                 │
│  7. Validate against schema                                                  │
│  8. Generate UI chips                                                        │
│  9. Log to analysis_log.jsonl                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              UI STATE UPDATE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  • activeTables = [fact_listings, dim_property_types, dim_states]           │
│  • activeFilters = [Beds: 3+, Baths: 2+, Type: Residential, ...]           │
│  • Graph shows connected tables                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            POST /api/generate_sql                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Remap fact_listings filters → dim tables                                │
│  2. Normalize and dedupe filters                                             │
│  3. Merge tables from filters                                                │
│  4. Build JOIN path via SCHEMA_EDGES (BFS)                                  │
│  5. Build SELECT (include WHERE columns)                                     │
│  6. Build WHERE clause                                                       │
│  7. Build ORDER BY                                                           │
│  8. Add LIMIT                                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GENERATED SQL                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  SELECT fact_listings.listing_id, fact_listings.bedrooms, ...               │
│  FROM fact_listings                                                          │
│  INNER JOIN dim_property_types ON ...                                        │
│  INNER JOIN dim_states ON ...                                                │
│  WHERE dim_property_types.property_type_name = 'Residential'                │
│    AND dim_states.state_code = 'NC'                                         │
│    AND fact_listings.bedrooms >= 3                                          │
│    AND fact_listings.bathrooms >= 2                                         │
│    AND fact_listings.list_price < 400000                                    │
│  LIMIT 100                                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            POST /api/query                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Execute SQL on SQLite                                                     │
│  • Return columns, data, row_count                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RESULTS TABLE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  listing_id | bedrooms | bathrooms | list_price | city    | state_code     │
│  123456     | 3        | 2         | 350000     | Raleigh | NC             │
│  234567     | 4        | 2         | 399000     | Durham  | NC             │
│  ...                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

### For Technical Teams
1. **Layered architecture** — extraction, validation, rendering are separate
2. **Fallback strategy** — LLM → Hybrid → Regex ensures reliability
3. **AST representation** — single source of truth for filters
4. **Schema-driven joins** — SCHEMA_EDGES enables multi-hop paths
5. **Feedback loop** — corrections improve system over time

### For Business Stakeholders
1. **Natural language works** — no SQL knowledge needed
2. **Transparent** — users see exactly what was understood
3. **Editable** — full control over final query
4. **Reliable** — works even without LLM
5. **Improvable** — feedback loop means continuous learning

### Netflix Principles Applied
1. **Trust but verify** — LLM results are validated
2. **Graceful degradation** — always have a fallback
3. **User control** — chips let users see and edit
4. **Observability** — log everything for debugging
5. **Iteration** — feedback loop closes the learning cycle
