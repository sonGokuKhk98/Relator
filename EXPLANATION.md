# Schema Knowledge Graph: A Deep Dive

## Table of Contents
1. [The Problem: Why RAG Fails for SQL Generation](#1-the-problem-why-rag-fails-for-sql-generation)
2. [The Solution: Knowledge Graph Approach](#2-the-solution-knowledge-graph-approach)
3. [Graph Theory Fundamentals](#3-graph-theory-fundamentals)
4. [NetworkX Library](#4-networkx-library)
5. [Data Structures & Design Patterns](#5-data-structures--design-patterns)
6. [Core Components Explained](#6-core-components-explained)
7. [Join Path Algorithm](#7-join-path-algorithm)
8. [Semantic Layer Concepts](#8-semantic-layer-concepts)
9. [PropTech Schema Design](#9-proptech-schema-design)
10. [Integration with LLM Text-to-SQL](#10-integration-with-llm-text-to-sql)
11. [Production Considerations](#11-production-considerations)

---

## 1. The Problem: Why RAG Fails for SQL Generation

### What is RAG?
**Retrieval-Augmented Generation (RAG)** is a technique where:
1. User asks a question
2. System searches a vector database for relevant context
3. Retrieved context is passed to an LLM
4. LLM generates a response using that context

### Why RAG Struggles with SQL

```
User Query: "Show me active users who bought Nike products"
```

**What RAG does:**
- Converts query to embedding vector
- Finds semantically similar table descriptions
- Might retrieve: `users` table (similar to "users")
- Might retrieve: `products` table (similar to "Nike products")

**What RAG misses:**
- The JOIN path: `users` → `orders` → `order_items` → `products`
- Business logic: "active" means `last_login > 30 days ago`
- Exact filter value: Is it "Nike" or "Nike Inc." or "NIKE"?

### The Core Issues

| Issue | Description | Example |
|-------|-------------|---------|
| **Missing Joins** | RAG doesn't understand FK relationships | Forgets the `order_items` junction table |
| **Business Logic** | Semantic similarity ≠ business rules | "active user" has a specific definition |
| **Value Matching** | LLM guesses filter values | "Nike" vs "Nike, Inc." vs "NIKE CORPORATION" |
| **Schema Topology** | No understanding of table connections | Doesn't know shortest path between tables |

---

## 2. The Solution: Knowledge Graph Approach

### What is a Knowledge Graph?

A **Knowledge Graph** represents information as a network of:
- **Nodes**: Entities (in our case, database tables)
- **Edges**: Relationships (in our case, foreign keys)
- **Properties**: Metadata on nodes/edges

```
┌─────────────┐         ┌─────────────┐
│   users     │────────▶│   orders    │
│  (node)     │  FK     │   (node)    │
└─────────────┘ (edge)  └─────────────┘
                              │
                              │ FK
                              ▼
                        ┌─────────────┐
                        │ order_items │
                        └─────────────┘
```

### Why Knowledge Graphs Work for SQL

1. **Deterministic**: Graph traversal is 100% accurate
2. **Complete**: Captures ALL relationships, not just similar ones
3. **Topology-Aware**: Knows the "shape" of your data
4. **Queryable**: Can answer "how do I get from A to B?"

### The Hybrid Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    User Natural Language Query             │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│              Intent Classifier (LLM or ML Model)           │
│  Extracts: Tables, Columns, Filters, Aggregations          │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                    Schema Knowledge Graph                   │
│  - Resolves aliases ("homes" → "listings")                 │
│  - Finds JOIN path using graph traversal                   │
│  - Validates column existence                              │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                    SQL Query Builder                        │
│  - Constructs SELECT, FROM, JOIN, WHERE clauses            │
│  - Uses exact values from metadata index                   │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                    Generated SQL Query                      │
└────────────────────────────────────────────────────────────┘
```

---

## 3. Graph Theory Fundamentals

### Key Concepts Used

#### Directed Graph (DiGraph)
A graph where edges have direction. We use this because foreign keys have direction:
- `orders.user_id` → `users.user_id` (orders REFERENCES users)

```python
# In our code:
self.graph = nx.DiGraph()
```

#### Nodes
Vertices in the graph. Each table is a node with properties:

```python
self.graph.add_node(
    "listings",           # Node ID
    schema="public",      # Property
    description="...",    # Property
    columns=[...],        # Property
    primary_keys=[...]    # Property
)
```

#### Edges
Connections between nodes. Each foreign key is an edge:

```python
self.graph.add_edge(
    "listings",           # Source node
    "neighborhoods",      # Target node
    from_column="neighborhood_id",
    to_column="neighborhood_id",
    join_condition="listings.neighborhood_id = neighborhoods.neighborhood_id"
)
```

#### Shortest Path
The minimum number of edges to traverse from node A to node B.

```
users → brokerages

Path 1: users → favorites → listings → agents → brokerages (4 hops)
Path 2: users → viewings → agents → brokerages (3 hops) ✓ SHORTEST
```

#### Bidirectional Edges
JOINs work both ways in SQL. If `orders` references `users`, you can:
- Start from `orders`, join to `users`
- Start from `users`, join to `orders`

That's why we add edges in both directions:

```python
# Forward edge
self.graph.add_edge("orders", "users", ...)

# Reverse edge (for bidirectional traversal)
self.graph.add_edge("users", "orders", ...)
```

---

## 4. NetworkX Library

### What is NetworkX?

NetworkX is a Python library for creating, manipulating, and studying complex networks (graphs).

```python
import networkx as nx
```

### Key Functions Used

#### Creating a Graph
```python
G = nx.DiGraph()  # Directed graph
```

#### Adding Nodes
```python
G.add_node("listings", description="Property listings", columns=[...])
```

#### Adding Edges
```python
G.add_edge("listings", "neighborhoods",
           join_condition="listings.neighborhood_id = neighborhoods.neighborhood_id")
```

#### Finding Shortest Path
```python
path = nx.shortest_path(G, source="users", target="brokerages")
# Returns: ['users', 'viewings', 'agents', 'brokerages']
```

#### Getting Neighbors
```python
neighbors = list(G.neighbors("listings"))
# Returns: ['property_types', 'neighborhoods', 'agents', ...]
```

#### Accessing Edge Data
```python
edge_data = G.edges["listings", "neighborhoods"]
# Returns: {'from_column': 'neighborhood_id', 'to_column': 'neighborhood_id', ...}
```

### Why NetworkX?

| Feature | Benefit |
|---------|---------|
| Pure Python | No external dependencies, easy deployment |
| Well-tested | Used in production by thousands of projects |
| Efficient | Optimized BFS/DFS algorithms |
| Rich API | Lots of graph algorithms built-in |

---

## 5. Data Structures & Design Patterns

### Dataclasses

Python's `@dataclass` decorator auto-generates `__init__`, `__repr__`, etc.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class Column:
    name: str
    data_type: str
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references_table: Optional[str] = None
    references_column: Optional[str] = None
    description: str = ""
    sample_values: List[Any] = field(default_factory=list)
```

#### Why Dataclasses?

1. **Less boilerplate**: No manual `__init__` writing
2. **Type hints**: IDE autocomplete and validation
3. **Immutability option**: `frozen=True` for immutable objects
4. **Default values**: Easy optional fields

### The Table Dataclass

```python
@dataclass
class Table:
    name: str                                    # "listings"
    schema: str = "public"                       # Database schema
    columns: Dict[str, Column] = field(...)     # Column definitions
    description: str = ""                        # Human-readable description
    aliases: List[str] = field(...)             # ["homes", "properties"]
    business_terms: List[str] = field(...)      # ["for sale", "real estate"]
```

**Key insight**: `aliases` and `business_terms` enable semantic matching without RAG.

### The JoinEdge Dataclass

```python
@dataclass
class JoinEdge:
    from_table: str      # "listings"
    from_column: str     # "neighborhood_id"
    to_table: str        # "neighborhoods"
    to_column: str       # "neighborhood_id"
    relationship_type: str = "many-to-one"

    @property
    def join_condition(self) -> str:
        return f"{self.from_table}.{self.from_column} = {self.to_table}.{self.to_column}"
```

### Design Pattern: Registry/Index

We use dictionaries as indexes for O(1) lookup:

```python
class SchemaGraph:
    def __init__(self):
        self.tables: Dict[str, Table] = {}           # Table registry
        self.alias_map: Dict[str, str] = {}          # Alias → Table name
        self.term_map: Dict[str, List[str]] = {}     # Business term → Tables
```

**Example:**
```python
alias_map = {
    "homes": "listings",
    "properties": "listings",
    "areas": "neighborhoods",
    "realtors": "agents"
}
```

---

## 6. Core Components Explained

### Component 1: Table Registration

```python
def add_table(self, table: Table) -> None:
    # 1. Store table in registry
    self.tables[table.name] = table

    # 2. Add node to graph with metadata
    self.graph.add_node(
        table.name,
        schema=table.schema,
        description=table.description,
        columns=list(table.columns.keys()),
        primary_keys=table.primary_keys
    )

    # 3. Index aliases for quick lookup
    for alias in table.aliases:
        self.alias_map[alias.lower()] = table.name

    # 4. Index business terms
    for term in table.business_terms:
        self.term_map[term.lower()].append(table.name)
```

**What this enables:**
- `resolve_table_name("homes")` → `"listings"`
- `resolve_table_name("for sale")` → `"listings"`

### Component 2: Relationship Registration

```python
def add_relationship(self, edge: JoinEdge) -> None:
    # Forward edge (following FK direction)
    self.graph.add_edge(
        edge.from_table,
        edge.to_table,
        from_column=edge.from_column,
        to_column=edge.to_column,
        relationship=edge.relationship_type,
        join_condition=edge.join_condition
    )

    # Reverse edge (for bidirectional JOIN capability)
    self.graph.add_edge(
        edge.to_table,
        edge.from_table,
        from_column=edge.to_column,
        to_column=edge.from_column,
        relationship=self._reverse_relationship(edge.relationship_type),
        join_condition=reverse_condition
    )
```

**Why bidirectional?**
```sql
-- Both are valid:
SELECT * FROM orders JOIN users ON orders.user_id = users.user_id
SELECT * FROM users JOIN orders ON users.user_id = orders.user_id
```

### Component 3: Alias Resolution

```python
def resolve_table_name(self, name: str) -> Optional[str]:
    name_lower = name.lower()

    # Try direct match or alias
    if name_lower in self.alias_map:
        return self.alias_map[name_lower]

    # Try business term match
    if name_lower in self.term_map:
        return self.term_map[name_lower][0]

    return None
```

**Example flow:**
```
Input: "homes"
Step 1: "homes".lower() → "homes"
Step 2: self.alias_map["homes"] → "listings"
Output: "listings"
```

### Component 4: Join Path Discovery

```python
def find_join_path(self, source: str, target: str) -> List[Tuple]:
    # Resolve aliases first
    source_table = self.resolve_table_name(source)  # "homes" → "listings"
    target_table = self.resolve_table_name(target)  # "areas" → "neighborhoods"

    # Use NetworkX shortest path
    path = nx.shortest_path(self.graph, source_table, target_table)
    # path = ["listings", "neighborhoods"]

    # Convert to edges with join info
    edges = []
    for i in range(len(path) - 1):
        edge_data = self.graph.edges[path[i], path[i + 1]]
        edges.append((path[i], path[i + 1], edge_data))

    return edges
```

### Component 5: SQL JOIN Generation

```python
def generate_join_sql(self, tables: List[str], join_type: str = "INNER JOIN") -> str:
    base_table = self.resolve_table_name(tables[0])
    joined_tables = {base_table}
    sql_parts = [base_table]

    for table in tables[1:]:
        resolved = self.resolve_table_name(table)

        # Find path from any already-joined table
        for joined in joined_tables:
            path = self.find_join_path(joined, resolved)
            if path:
                # Add JOIN clauses for each hop
                for from_t, to_t, edge_data in path:
                    if to_t not in joined_tables:
                        sql_parts.append(
                            f"{join_type} {to_t} ON {edge_data['join_condition']}"
                        )
                        joined_tables.add(to_t)
                break

    return "\n".join(sql_parts)
```

**Example:**
```python
generate_join_sql(["listings", "amenities", "neighborhoods"])
```
**Output:**
```sql
listings
INNER JOIN listing_amenities ON listings.listing_id = listing_amenities.listing_id
INNER JOIN amenities ON listing_amenities.amenity_id = amenities.amenity_id
INNER JOIN neighborhoods ON listings.neighborhood_id = neighborhoods.neighborhood_id
```

---

## 7. Join Path Algorithm

### The Problem
Given tables A and D, find the sequence of JOINs to connect them.

```
A ←──── B ←──── C ←──── D
        │
        └───── E ←──── F
```

### Breadth-First Search (BFS)

NetworkX's `shortest_path` uses BFS for unweighted graphs:

```
Start: A
Queue: [A]
Visited: {A}

Step 1: Dequeue A, enqueue neighbors [B]
Queue: [B]
Visited: {A, B}

Step 2: Dequeue B, enqueue neighbors [C, E]
Queue: [C, E]
Visited: {A, B, C, E}

Step 3: Dequeue C, enqueue neighbors [D]
Queue: [E, D]
Visited: {A, B, C, D, E}

Step 4: D is target! Reconstruct path: A → B → C → D
```

### Time Complexity

| Operation | Complexity |
|-----------|------------|
| Add table | O(1) |
| Add relationship | O(1) |
| Find path (BFS) | O(V + E) where V=tables, E=relationships |
| Generate SQL | O(T × (V + E)) where T=tables requested |

For typical schemas (< 100 tables), this is effectively instant.

### Handling Junction Tables

**Problem:** User asks for `listings` and `amenities`, but they're connected via `listing_amenities`.

```
listings ←───── listing_amenities ─────→ amenities
```

**Solution:** BFS naturally finds the path through junction tables:

```python
path = find_join_path("listings", "amenities")
# Returns: [
#   ("listings", "listing_amenities", {...}),
#   ("listing_amenities", "amenities", {...})
# ]
```

---

## 8. Semantic Layer Concepts

### What is a Semantic Layer?

A **Semantic Layer** sits between users and the database, translating:
- Business terms → Technical columns
- Metrics → SQL aggregations
- Dimensions → Groupable columns

### Our Implementation

#### 1. Aliases (Table-Level Semantics)
```python
Table(
    name="listings",
    aliases=["homes", "properties", "houses", "real_estate"]
)
```

User says "homes" → System knows they mean `listings` table.

#### 2. Business Terms (Concept-Level Semantics)
```python
Table(
    name="listings",
    business_terms=["for sale", "for rent", "property search"]
)
```

User says "properties for sale" → System includes `listings` table.

#### 3. Column Descriptions (Field-Level Semantics)
```python
Column(
    name="status",
    description="active, pending, sold, withdrawn"
)
```

LLM knows valid values for `WHERE status = ?`.

#### 4. Relationship Semantics
```python
JoinEdge(
    from_table="listings",
    to_table="agents",
    relationship_type="many-to-one"  # Many listings per agent
)
```

System knows cardinality for optimization.

### Semantic Layer vs RAG

| Aspect | RAG | Semantic Layer |
|--------|-----|----------------|
| Matching | Probabilistic (cosine similarity) | Deterministic (exact lookup) |
| Coverage | Finds "similar" things | Finds "correct" things |
| Maintenance | Requires re-embedding | Update dictionary |
| Explainability | "This seemed relevant" | "homes maps to listings" |

---

## 9. PropTech Schema Design

### Entity-Relationship Model

```
┌─────────────────┐
│   brokerages    │
└────────┬────────┘
         │ 1:N
         ▼
┌─────────────────┐       ┌─────────────────┐
│     agents      │◄──────│    viewings     │
└────────┬────────┘  N:1  └────────┬────────┘
         │ 1:N                     │ N:1
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│    listings     │◄──────│     users       │
└┬───────┬───────┬┘       └────────┬────────┘
 │       │       │                 │ 1:N
 │       │       │                 ▼
 │       │       │        ┌─────────────────┐
 │       │       └───────▶│   favorites     │
 │       │                └─────────────────┘
 │       │
 │       │ N:1            ┌─────────────────┐
 │       └───────────────▶│  neighborhoods  │
 │                        └─────────────────┘
 │ N:1
 │                        ┌─────────────────┐
 └───────────────────────▶│ property_types  │
                          └─────────────────┘

┌─────────────────┐       ┌─────────────────┐
│listing_amenities│◄─────▶│   amenities     │
└─────────────────┘  N:M  └─────────────────┘
```

### Table Purposes

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `listings` | Core property data | price, bedrooms, sqft, status |
| `neighborhoods` | Location intelligence | walk_score, school_rating, crime_index |
| `property_types` | Classification | single_family, condo, townhouse |
| `amenities` | Features | pool, garage, fireplace |
| `listing_amenities` | Junction table | Links listings ↔ amenities |
| `agents` | Realtor profiles | rating, experience, specialization |
| `brokerages` | Companies | name, contact |
| `users` | Property seekers | budget, pre-approval status |
| `favorites` | Saved properties | user_id, listing_id, notes |
| `viewings` | Appointments | scheduled_at, status, feedback |
| `saved_searches` | Search criteria | price range, min bedrooms |

### Design Decisions

#### 1. Many-to-Many for Amenities
A listing can have multiple amenities, and an amenity applies to multiple listings:
```
listings ←── listing_amenities ──→ amenities
```

#### 2. Neighborhood as Separate Entity
Instead of denormalizing city/area data into listings:
- Enables neighborhood-level analytics
- Stores computed scores (walk_score, etc.)
- Single source of truth for location data

#### 3. Viewings Connect Three Entities
```python
viewings.listing_id → listings
viewings.user_id → users
viewings.agent_id → agents
```
This models the real-world relationship: "User X views Listing Y with Agent Z"

---

## 10. Integration with LLM Text-to-SQL

### The Complete Pipeline

```python
    intent = llm_extract_intent(user_query)
def text_to_sql(user_query: str, schema_graph: SchemaGraph) -> str:
    # Step 1: LLM extracts intent
    intent = llm_extract_intent(user_query)
    # {
    #   "tables": ["listings", "neighborhoods"],
    #   "columns": ["address", "price", "walk_score"],
    #   "filters": [("bedrooms", ">=", 3), ("status", "=", "active")],
    #   "aggregations": [],
    #   "order_by": [("price", "ASC")]
    # }

    # Step 2: Knowledge Graph resolves and validates
    resolved_tables = [schema_graph.resolve_table_name(t) for t in intent["tables"]]

    # Step 3: Generate JOINs deterministically
    join_sql = schema_graph.generate_join_sql(resolved_tables)

    # Step 4: Build complete query
    sql = f"""
    SELECT {', '.join(intent['columns'])}
    FROM {join_sql}
    WHERE {' AND '.join(format_filter(f) for f in intent['filters'])}
    ORDER BY {', '.join(f"{col} {dir}" for col, dir in intent['order_by'])}
    """

    return sql
```

### What the LLM Does vs What the Graph Does

| Task | LLM | Knowledge Graph |
|------|-----|-----------------|
| Understand user intent | ✅ | ❌ |
| Extract entities/filters | ✅ | ❌ |
| Find correct tables | ❌ (guesses) | ✅ (exact lookup) |
| Generate JOIN path | ❌ (often wrong) | ✅ (always correct) |
| Validate columns exist | ❌ | ✅ |
| Handle aliases | ❌ | ✅ |

### Example Integration Prompt

```python
SYSTEM_PROMPT = """
You are a SQL intent extractor. Given a user question about real estate:

1. Identify which tables are needed (use these exact names):
   - listings, neighborhoods, property_types, amenities, agents, brokerages, users, favorites, viewings

2. Identify columns to SELECT

3. Identify filter conditions

4. Identify any aggregations (COUNT, SUM, AVG)

5. Identify ORDER BY

Output JSON only:
{
  "tables": ["table1", "table2"],
  "columns": ["col1", "col2"],
  "filters": [["column", "operator", "value"]],
  "aggregations": [{"function": "COUNT", "column": "*", "alias": "total"}],
  "order_by": [["column", "direction"]]
}
"""
```

---

## 11. Production Considerations

### 1. Schema Synchronization

Keep the graph in sync with your actual database:

```python
def sync_from_database(connection_string: str) -> SchemaGraph:
    """Build graph from live database metadata."""
    engine = create_engine(connection_string)
    inspector = inspect(engine)

    graph = SchemaGraph()

    for table_name in inspector.get_table_names():
        columns = {}
        for col in inspector.get_columns(table_name):
            columns[col['name']] = Column(
                name=col['name'],
                data_type=str(col['type']),
                is_primary_key=col.get('primary_key', False)
            )

        # Get foreign keys
        for fk in inspector.get_foreign_keys(table_name):
            # Add relationship...
            pass

        graph.add_table(Table(name=table_name, columns=columns))

    return graph
```

### 2. Caching

Cache the graph in memory or Redis:

```python
import pickle

# Save
with open('schema_graph.pkl', 'wb') as f:
    pickle.dump(schema_graph, f)

# Load
with open('schema_graph.pkl', 'rb') as f:
    schema_graph = pickle.load(f)
```

### 3. Version Control for Schema

Track schema changes in git:

```yaml
# schema_definition.yaml
tables:
  - name: listings
    aliases: [homes, properties]
    business_terms: [for sale, property search]
    columns:
      - name: listing_id
        type: INTEGER
        primary_key: true
      - name: price
        type: DECIMAL(12,2)
        description: Listing price in USD

relationships:
  - from: listings.neighborhood_id
    to: neighborhoods.neighborhood_id
    type: many-to-one
```

### 4. Query Validation

Before executing generated SQL:

```python
def validate_query(sql: str, schema_graph: SchemaGraph) -> List[str]:
    errors = []

    # Parse SQL (use sqlglot or similar)
    parsed = sqlglot.parse_one(sql)

    # Check all tables exist
    for table in parsed.find_all(sqlglot.exp.Table):
        if table.name not in schema_graph.tables:
            errors.append(f"Unknown table: {table.name}")

    # Check all columns exist
    for column in parsed.find_all(sqlglot.exp.Column):
        table_name = column.table
        col_name = column.name
        if table_name and col_name not in schema_graph.tables[table_name].columns:
            errors.append(f"Unknown column: {table_name}.{col_name}")

    return errors
```

### 5. Metrics and Monitoring

Track query generation quality:

```python
@dataclass
class QueryMetrics:
    query_id: str
    user_query: str
    generated_sql: str
    execution_time_ms: float
    row_count: int
    error: Optional[str]
    user_feedback: Optional[str]  # thumbs up/down
```

### 6. Fallback Strategy

When the graph can't find a path:

```python
def generate_sql_with_fallback(tables: List[str]) -> str:
    try:
        return schema_graph.generate_join_sql(tables)
    except ValueError as e:
        # Log the failure
        logger.warning(f"Graph couldn't find path: {e}")

        # Fallback to LLM-only generation
        return llm_generate_joins(tables)
```

---

## Summary

### Key Takeaways

1. **RAG is probabilistic, Knowledge Graphs are deterministic** - For SQL generation, you need accuracy, not similarity.

2. **Graph traversal solves the JOIN problem** - BFS finds the shortest path between any two tables in O(V+E) time.

3. **Semantic layer bridges human language and database structure** - Aliases and business terms enable natural language understanding without embedding vectors.

4. **The LLM's role changes** - It extracts intent, not generates SQL structure. The graph handles the structural correctness.

5. **This approach is production-ready** - Companies like Airbnb, Uber, and Netflix use similar semantic layer approaches for their data platforms.

### Files in This Project

| File | Purpose |
|------|---------|
| `schema_graph.py` | Core SchemaGraph class with NetworkX |
| `example_usage.py` | PropTech schema and demo queries |
| `requirements.txt` | Dependencies (networkx) |
| `EXPLANATION.md` | This documentation |

### Next Steps

1. **Add your actual schema** - Replace the PropTech example with your database
2. **Integrate with LLM** - Use Claude/GPT for intent extraction
3. **Add value index** - Index actual column values for filter validation
4. **Build API** - Wrap in FastAPI for production use
