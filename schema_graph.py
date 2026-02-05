"""
Schema Knowledge Graph for Dynamic SQL Join Path Discovery

This module builds a graph representation of your database schema where:
- Nodes = Tables (with metadata for semantic matching)
- Edges = Foreign Key relationships (with join conditions)

Uses NetworkX to find the shortest join path between any two tables,
ensuring 100% accurate joins instead of relying on probabilistic RAG.
"""

import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Column:
    """Represents a database column with metadata."""
    name: str
    data_type: str
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references_table: Optional[str] = None
    references_column: Optional[str] = None
    description: str = ""
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class Table:
    """Represents a database table with semantic metadata."""
    name: str
    schema: str = "public"
    columns: Dict[str, Column] = field(default_factory=dict)
    description: str = ""
    aliases: List[str] = field(default_factory=list)  # Alternative names users might use
    business_terms: List[str] = field(default_factory=list)  # e.g., "revenue", "sales"

    @property
    def primary_keys(self) -> List[str]:
        return [col.name for col in self.columns.values() if col.is_primary_key]

    @property
    def foreign_keys(self) -> List[Column]:
        return [col for col in self.columns.values() if col.is_foreign_key]


@dataclass
class JoinEdge:
    """Represents a foreign key relationship between tables."""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship_type: str = "many-to-one"  # one-to-one, one-to-many, many-to-one, many-to-many

    @property
    def join_condition(self) -> str:
        return f"{self.from_table}.{self.from_column} = {self.to_table}.{self.to_column}"


class SchemaGraph:
    """
    Knowledge Graph representation of a database schema.

    Enables deterministic join path discovery using graph traversal
    instead of probabilistic RAG-based guessing.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.tables: Dict[str, Table] = {}
        self.alias_map: Dict[str, str] = {}  # Maps aliases to canonical table names
        self.term_map: Dict[str, List[str]] = defaultdict(list)  # Maps business terms to tables

    def add_table(self, table: Table) -> None:
        """Add a table node to the graph with its metadata."""
        self.tables[table.name] = table
        self.graph.add_node(
            table.name,
            schema=table.schema,
            description=table.description,
            columns=list(table.columns.keys()),
            primary_keys=table.primary_keys
        )

        # Index aliases for quick lookup
        for alias in table.aliases:
            self.alias_map[alias.lower()] = table.name
        self.alias_map[table.name.lower()] = table.name

        # Index business terms
        for term in table.business_terms:
            self.term_map[term.lower()].append(table.name)

    def add_relationship(self, edge: JoinEdge) -> None:
        """Add a foreign key relationship as an edge."""
        # Add bidirectional edges for traversal (joins work both ways)
        self.graph.add_edge(
            edge.from_table,
            edge.to_table,
            from_column=edge.from_column,
            to_column=edge.to_column,
            relationship=edge.relationship_type,
            join_condition=edge.join_condition
        )
        # Reverse edge for bidirectional traversal
        reverse_condition = f"{edge.to_table}.{edge.to_column} = {edge.from_table}.{edge.from_column}"
        self.graph.add_edge(
            edge.to_table,
            edge.from_table,
            from_column=edge.to_column,
            to_column=edge.from_column,
            relationship=self._reverse_relationship(edge.relationship_type),
            join_condition=reverse_condition
        )

    def _reverse_relationship(self, rel_type: str) -> str:
        """Get the reverse relationship type."""
        mapping = {
            "one-to-many": "many-to-one",
            "many-to-one": "one-to-many",
            "one-to-one": "one-to-one",
            "many-to-many": "many-to-many"
        }
        return mapping.get(rel_type, rel_type)

    def resolve_table_name(self, name: str) -> Optional[str]:
        """Resolve an alias or business term to the canonical table name."""
        name_lower = name.lower()

        # Direct match or alias
        if name_lower in self.alias_map:
            return self.alias_map[name_lower]

        # Business term match
        if name_lower in self.term_map:
            tables = self.term_map[name_lower]
            if len(tables) == 1:
                return tables[0]
            # Multiple matches - return all for disambiguation
            return tables[0]  # Or raise an error for ambiguity

        return None

    def find_join_path(
        self,
        source: str,
        target: str,
        through: Optional[List[str]] = None
    ) -> Optional[List[Tuple[str, str, dict]]]:
        """
        Find the shortest join path between two tables.

        Args:
            source: Starting table name (or alias)
            target: Target table name (or alias)
            through: Optional list of tables that must be included in the path

        Returns:
            List of (from_table, to_table, edge_data) tuples representing the join path
        """
        # Resolve aliases
        source_table = self.resolve_table_name(source)
        target_table = self.resolve_table_name(target)

        if not source_table or not target_table:
            return None

        if source_table not in self.graph or target_table not in self.graph:
            return None

        try:
            if through:
                # Find path through specific tables
                path = self._find_path_through(source_table, target_table, through)
            else:
                # Simple shortest path
                path = nx.shortest_path(self.graph, source_table, target_table)

            # Convert path to edges with join info
            edges = []
            for i in range(len(path) - 1):
                edge_data = self.graph.edges[path[i], path[i + 1]]
                edges.append((path[i], path[i + 1], edge_data))

            return edges

        except nx.NetworkXNoPath:
            return None

    def _find_path_through(
        self,
        source: str,
        target: str,
        through: List[str]
    ) -> List[str]:
        """Find a path that goes through all specified tables."""
        resolved_through = [self.resolve_table_name(t) or t for t in through]

        # Build complete path through all waypoints
        waypoints = [source] + resolved_through + [target]
        complete_path = []

        for i in range(len(waypoints) - 1):
            segment = nx.shortest_path(self.graph, waypoints[i], waypoints[i + 1])
            if complete_path:
                segment = segment[1:]  # Avoid duplicating the junction node
            complete_path.extend(segment)

        return complete_path

    def generate_join_sql(
        self,
        tables: List[str],
        join_type: str = "INNER JOIN"
    ) -> str:
        """
        Generate SQL JOIN clause for a list of tables.

        Args:
            tables: List of table names to join (order matters for the base table)
            join_type: Type of join (INNER JOIN, LEFT JOIN, etc.)

        Returns:
            SQL string with proper JOIN syntax
        """
        if len(tables) < 2:
            return tables[0] if tables else ""

        # Find the join path that connects all tables
        base_table = self.resolve_table_name(tables[0])
        if not base_table:
            raise ValueError(f"Unknown table: {tables[0]}")

        joined_tables = {base_table}
        sql_parts = [base_table]

        for table in tables[1:]:
            resolved = self.resolve_table_name(table)
            if not resolved:
                raise ValueError(f"Unknown table: {table}")

            if resolved in joined_tables:
                continue

            # Find path from any already-joined table to this table
            best_path = None
            for joined in joined_tables:
                path = self.find_join_path(joined, resolved)
                if path and (best_path is None or len(path) < len(best_path)):
                    best_path = path

            if not best_path:
                raise ValueError(f"No join path found to table: {table}")

            # Add all intermediate tables
            for _from_t, to_t, edge_data in best_path:
                if to_t not in joined_tables:
                    sql_parts.append(
                        f"{join_type} {to_t} ON {edge_data['join_condition']}"
                    )
                    joined_tables.add(to_t)

        return "\n".join(sql_parts)

    def get_related_tables(
        self,
        table: str,
        max_distance: int = 2
    ) -> Dict[str, int]:
        """
        Get all tables within N hops of the given table.

        Useful for context retrieval - shows what tables are relevant
        when a user mentions a specific entity.
        """
        resolved = self.resolve_table_name(table)
        if not resolved:
            return {}

        distances = {resolved: 0}
        current_level = {resolved}

        for distance in range(1, max_distance + 1):
            next_level = set()
            for t in current_level:
                for neighbor in self.graph.neighbors(t):
                    if neighbor not in distances:
                        distances[neighbor] = distance
                        next_level.add(neighbor)
            current_level = next_level

        return distances

    def visualize(self, output_file: Optional[str] = None) -> str:
        """Generate a text visualization of the schema graph."""
        lines = ["Schema Graph Visualization", "=" * 40]

        for table_name in sorted(self.tables.keys()):
            table = self.tables[table_name]
            lines.append(f"\n[{table_name}]")
            if table.description:
                lines.append(f"  Description: {table.description}")
            if table.aliases:
                lines.append(f"  Aliases: {', '.join(table.aliases)}")

            # Show outgoing relationships
            for _, to_table, edge_data in self.graph.out_edges(table_name, data=True):
                lines.append(f"  --> {to_table} ({edge_data['relationship']})")
                lines.append(f"      ON {edge_data['join_condition']}")

        result = "\n".join(lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(result)

        return result


def build_graph_from_ddl(ddl_statements: List[str]) -> SchemaGraph:
    """
    Parse DDL statements and build a SchemaGraph.

    This is a simplified parser - for production use, consider
    using sqlglot or sqlparse for robust DDL parsing.
    """
    import re

    graph = SchemaGraph()

    # Pattern for CREATE TABLE
    table_pattern = re.compile(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\((.*?)\)',
        re.IGNORECASE | re.DOTALL
    )

    # Pattern for foreign key
    fk_pattern = re.compile(
        r'FOREIGN\s+KEY\s*\((\w+)\)\s*REFERENCES\s+(\w+)\s*\((\w+)\)',
        re.IGNORECASE
    )

    # Pattern for column definition
    col_pattern = re.compile(
        r'(\w+)\s+([\w\(\)]+)(?:\s+(PRIMARY\s+KEY|NOT\s+NULL))?',
        re.IGNORECASE
    )

    edges_to_add = []

    for ddl in ddl_statements:
        match = table_pattern.search(ddl)
        if not match:
            continue

        table_name = match.group(1)
        body = match.group(2)

        columns = {}

        # Parse columns
        for line in body.split(','):
            line = line.strip()

            # Check for foreign key
            fk_match = fk_pattern.search(line)
            if fk_match:
                from_col, ref_table, ref_col = fk_match.groups()
                edges_to_add.append(JoinEdge(
                    from_table=table_name,
                    from_column=from_col,
                    to_table=ref_table,
                    to_column=ref_col
                ))
                if from_col in columns:
                    columns[from_col].is_foreign_key = True
                    columns[from_col].references_table = ref_table
                    columns[from_col].references_column = ref_col
                continue

            # Parse column
            col_match = col_pattern.match(line)
            if col_match:
                col_name = col_match.group(1)
                col_type = col_match.group(2)
                modifiers = col_match.group(3) or ""

                is_pk = 'PRIMARY KEY' in modifiers.upper()

                columns[col_name] = Column(
                    name=col_name,
                    data_type=col_type,
                    is_primary_key=is_pk
                )

        table = Table(name=table_name, columns=columns)
        graph.add_table(table)

    # Add relationships after all tables are added
    for edge in edges_to_add:
        if edge.from_table in graph.tables and edge.to_table in graph.tables:
            graph.add_relationship(edge)

    return graph
