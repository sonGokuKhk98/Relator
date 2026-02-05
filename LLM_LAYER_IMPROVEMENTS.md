# Netflix-Inspired LLM Layer - Architecture Documentation

This document describes the production-grade LLM layer implementation inspired by [Netflix's AI Graph Search](https://netflixtechblog.com/the-ai-evolution-of-graph-search-at-netflix-d416ec5b1151).

---

## Implementation Status: COMPLETE

All major features from Netflix's approach have been implemented:

| Feature | Status | Module |
|---------|--------|--------|
| AST-based Filter DSL | Implemented | `llm_layer/filter_dsl.py` |
| 3-Layer Validation (Syntactic/Semantic/Pragmatic) | Implemented | `llm_layer/validator.py` |
| Entity Resolution with @Mentions | Implemented | `llm_layer/entity_resolver.py` |
| Multi-Strategy Extraction (LLM/Hybrid/Regex) | Implemented | `llm_layer/smart_extractor.py` |
| UI Chip Generation | Implemented | `llm_layer/chip_generator.py` |
| Caching Layer | Implemented | `llm_layer/cache.py` |
| Metrics & Observability | Implemented | `llm_layer/metrics.py` |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Query                                   │
│              "3 bedroom homes under $500k in @Charlotte"             │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SMART EXTRACTOR                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐    │
│  │   Cache     │  │   Entity     │  │    Strategy Selection    │    │
│  │   Check     │─▶│   Resolver   │─▶│   (LLM/Hybrid/Regex)    │    │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FILTER AST                                    │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  FilterGroup(AND)                                              │  │
│  │    ├─ FilterNode(bedrooms >= 3)                               │  │
│  │    ├─ FilterNode(list_price < 500000)                         │  │
│  │    └─ FilterNode(city = 'Charlotte')                          │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       VALIDATOR                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐    │
│  │  Syntactic  │─▶│   Semantic   │─▶│      Pragmatic          │    │
│  │  (Grammar)  │  │  (Schema)    │  │   (Logic/Consistency)   │    │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CHIP GENERATOR                                  │
│  ┌─────────┐  ┌─────────────┐  ┌───────────────┐                   │
│  │ Beds: 3+│  │ Price: <$500K│  │ City: Charlotte│                  │
│  │ (purple)│  │   (green)    │  │    (blue)      │                  │
│  └─────────┘  └─────────────┘  └───────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Details

### 1. Filter DSL (`llm_layer/filter_dsl.py`)

Abstract Syntax Tree representation of filters that can be parsed, validated, and converted to SQL.

**Key Classes:**
- `FilterOperator` - Enum of all supported operators (=, !=, >, >=, <, <=, LIKE, IN, BETWEEN, etc.)
- `FilterNode` - Single filter condition (table.column operator value)
- `FilterGroup` - Logical grouping of filters (AND/OR)
- `FilterAST` - Complete query representation
- `DSLParser` - Parse text DSL into AST

**Example:**
```python
from llm_layer import FilterNode, FilterGroup, FilterAST, FilterOperator, LogicalOperator

# Build filter tree
filters = FilterGroup(LogicalOperator.AND)
filters.add(FilterNode('fact_listings', 'bedrooms', FilterOperator.GTE, 3))
filters.add(FilterNode('fact_listings', 'list_price', FilterOperator.LT, 500000))

# Generate SQL
print(filters.to_sql())
# Output: fact_listings.bedrooms >= 3 AND fact_listings.list_price < 500000
```

---

### 2. Three-Layer Validator (`llm_layer/validator.py`)

Ensures filters are correct at syntactic, semantic, and pragmatic levels.

**Validation Levels:**

| Level | What it checks | Example |
|-------|----------------|---------|
| Syntactic | Grammar & structure | LIKE requires string value |
| Semantic | Schema validity | Column exists in table |
| Pragmatic | Logical consistency | price > 500000 AND price < 300000 (contradiction) |

**Example:**
```python
from llm_layer import FilterValidator
from llm_layer.validator import SchemaRegistry

registry = SchemaRegistry.from_schema_definition(SCHEMA)
validator = FilterValidator(registry)
result = validator.validate(ast)

if not result.valid:
    for error in result.errors:
        print(f"{error.level.value}: {error.message}")
```

---

### 3. Entity Resolver (`llm_layer/entity_resolver.py`)

Netflix-style entity resolution with @mention support for disambiguation.

**Features:**
- `@Charlotte` - Explicit entity mention
- `@city:Charlotte` - Typed mention
- Fuzzy matching against vocabularies
- Autocomplete suggestions

**Example:**
```python
from llm_layer import EntityResolver

resolver = EntityResolver(db_connection)

# Resolve entities from query
entities = resolver.resolve_all_entities("homes in @Charlotte, NC")

# Autocomplete for UI
suggestions = resolver.suggest_completions("Char", limit=5)
# Returns: [{"value": "Charlotte", "type": "city", "score": 100}, ...]
```

---

### 4. Smart Extractor (`llm_layer/smart_extractor.py`)

Multi-strategy extraction with intelligent fallback based on confidence.

**Strategies:**
| Strategy | Confidence | Description |
|----------|------------|-------------|
| `LLM_FULL` | > 85% | High-confidence LLM result |
| `LLM_VALIDATED` | 60-85% | LLM + validation |
| `HYBRID` | < 60% | Combine LLM + regex |
| `REGEX_ONLY` | N/A | LLM failed, use regex |
| `CACHED` | 100% | Retrieved from cache |

**Example:**
```python
from llm_layer import SmartExtractor

extractor = SmartExtractor(
    llm_model=gemini_model,
    schema_definition=SCHEMA,
    db_connection=db,
    cache=cache,
)

result = extractor.extract_sync("3 bed homes under $500k in Charlotte")
print(f"Strategy: {result.strategy_used.value}")
print(f"Confidence: {result.confidence}")
print(f"Interpretation: {result.llm_interpretation}")
```

---

### 5. Chip Generator (`llm_layer/chip_generator.py`)

Converts filter AST to visual UI chips for interactive display.

**Chip Types:**
| Type | Color | Example |
|------|-------|---------|
| `entity` | Blue | City: Charlotte |
| `range` | Green | Price: < $500K |
| `comparison` | Purple | Beds: 3+ |
| `status` | Teal | Active |
| `sort` | Orange | Sort: Price ↑ |

**Example:**
```python
from llm_layer import ChipGenerator

generator = ChipGenerator()
chips = generator.generate(extraction_result)

for chip in chips:
    print(f"{chip.display_text} - {chip.chip_type.value} ({chip.color})")
```

---

### 6. Caching (`llm_layer/cache.py`)

LRU cache with query normalization and TTL expiration.

**Features:**
- Query normalization for better hit rates
- TTL-based expiration (default 1 hour)
- Hit tracking for analytics
- Memory-bounded storage

**Example:**
```python
from llm_layer import ExtractionCache

cache = ExtractionCache(max_size=1000, ttl_seconds=3600)
cache.set("query", result)
cached = cache.get("query")  # Returns None if expired

print(cache.get_stats())
# {"size": 1, "hits": 1, "misses": 0, "hit_rate": 1.0}
```

---

### 7. Metrics (`llm_layer/metrics.py`)

Observability for monitoring extraction quality and performance.

**Tracked Metrics:**
- Queries per hour
- Average confidence
- Average latency
- Strategy distribution
- Validation failure rate
- Cache hit rate

**Example:**
```python
from llm_layer import MetricsCollector

collector = MetricsCollector()
collector.record_extraction(query, result, latency_ms)

# Dashboard data
print(collector.get_dashboard_data())
```

---

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/suggest_filters` | POST | Extract filters from natural language |
| `/api/generate_sql` | POST | Generate SQL from filters |
| `/api/query` | POST | Execute SQL query |

### Entity Resolution

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/entity/suggest` | GET | Autocomplete for @mentions |
| `/api/entity/vocabulary/{type}` | GET | Get full vocabulary |

### Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/metrics/summary` | GET | Extraction metrics summary |
| `/api/metrics/dashboard` | GET | Full dashboard data |
| `/api/metrics/strategies` | GET | Strategy analysis |
| `/api/cache/stats` | GET | Cache statistics |
| `/health` | GET | Health check |

---

## Response Format

### `/api/suggest_filters` Response

```json
{
  "detected_tables": ["fact_listings"],
  "extracted_filters": [
    {"table": "fact_listings", "column": "bedrooms", "operator": ">=", "value": 3},
    {"table": "fact_listings", "column": "list_price", "operator": "<", "value": 500000},
    {"table": "fact_listings", "column": "city", "operator": "=", "value": "Charlotte"}
  ],
  "order_by": null,
  "extraction_method": "llm_full",
  "confidence": 0.95,
  "interpretation": "Properties with 3+ bedrooms, under $500K, in Charlotte",
  "chips": [
    {"id": "filter_0", "display_text": "Beds: 3+", "chip_type": "comparison", "color": "purple"},
    {"id": "filter_1", "display_text": "Price: < $500K", "chip_type": "range", "color": "green"},
    {"id": "filter_2", "display_text": "City: Charlotte", "chip_type": "entity", "color": "blue"}
  ],
  "entities": [
    {"raw_text": "Charlotte", "entity_type": "city", "canonical_value": "Charlotte", "confidence": 1.0}
  ],
  "validation": {
    "valid": true,
    "syntactic_valid": true,
    "semantic_valid": true,
    "pragmatic_valid": true,
    "errors": [],
    "warnings": []
  },
  "warnings": []
}
```

---

## Usage Examples

### Basic Query
```bash
curl -X POST http://localhost:8000/api/suggest_filters \
  -H "Content-Type: application/json" \
  -d '{"query": "3 bedroom homes under $500k in Charlotte"}'
```

### @Mention Query
```bash
curl -X POST http://localhost:8000/api/suggest_filters \
  -H "Content-Type: application/json" \
  -d '{"query": "luxury homes in @city:Austin with pool"}'
```

### Entity Autocomplete
```bash
curl "http://localhost:8000/api/entity/suggest?q=Char&limit=5"
```

---

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export GOOGLE_API_KEY="your-gemini-api-key"

# Run server
python app.py
# or
uvicorn app:app --reload --port 8000
```

---

## References

- [Netflix TechBlog: The AI Evolution of Graph Search](https://netflixtechblog.com/the-ai-evolution-of-graph-search-at-netflix-d416ec5b1151)
- [Netflix: Knowledge Graph Enhancement with LLMs](https://www.zenml.io/llmops-database/knowledge-graph-enhancement-with-llms-for-content-understanding)
