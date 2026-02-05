"""
AST-Based Filter Domain Specific Language

Inspired by Netflix's Graph Search Filter DSL, this module provides
a structured representation of filters that can be:
1. Parsed from LLM output
2. Validated against schema
3. Converted to SQL
4. Rendered as UI chips
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Union, Optional, Any, Dict
import json
import re


class FilterOperator(Enum):
    """Supported filter operators."""
    EQ = "="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    LIKE = "LIKE"
    NOT_LIKE = "NOT LIKE"
    IN = "IN"
    NOT_IN = "NOT IN"
    BETWEEN = "BETWEEN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"

    @classmethod
    def from_string(cls, s: str) -> "FilterOperator":
        """Parse operator from string."""
        mapping = {
            "=": cls.EQ,
            "==": cls.EQ,
            "!=": cls.NE,
            "<>": cls.NE,
            ">": cls.GT,
            ">=": cls.GTE,
            "<": cls.LT,
            "<=": cls.LTE,
            "LIKE": cls.LIKE,
            "like": cls.LIKE,
            "NOT LIKE": cls.NOT_LIKE,
            "IN": cls.IN,
            "in": cls.IN,
            "NOT IN": cls.NOT_IN,
            "BETWEEN": cls.BETWEEN,
            "IS NULL": cls.IS_NULL,
            "IS NOT NULL": cls.IS_NOT_NULL,
        }
        return mapping.get(s, cls.EQ)

    def is_comparison(self) -> bool:
        return self in [self.EQ, self.NE, self.GT, self.GTE, self.LT, self.LTE]

    def is_pattern(self) -> bool:
        return self in [self.LIKE, self.NOT_LIKE]

    def is_collection(self) -> bool:
        return self in [self.IN, self.NOT_IN, self.BETWEEN]

    def is_null_check(self) -> bool:
        return self in [self.IS_NULL, self.IS_NOT_NULL]


@dataclass
class FilterNode:
    """
    AST node representing a single filter condition.

    Example: transactions.actual_worth >= 1000000
    """
    table: str
    column: str
    operator: FilterOperator
    value: Union[str, int, float, List, None]

    # Metadata for UI rendering
    display_value: Optional[str] = None
    original_text: Optional[str] = None  # The original user text that created this
    confidence: float = 1.0

    def __post_init__(self):
        # Ensure operator is FilterOperator enum
        if isinstance(self.operator, str):
            self.operator = FilterOperator.from_string(self.operator)

    def to_sql(self, use_aliases: bool = False) -> str:
        """Convert to SQL WHERE clause fragment."""
        table_ref = self.table if not use_aliases else self.table[0].lower()
        col_ref = f"{table_ref}.{self.column}"

        if self.operator.is_null_check():
            return f"{col_ref} {self.operator.value}"

        if self.operator == FilterOperator.IN:
            if isinstance(self.value, list):
                values = ", ".join(
                    f"'{v}'" if isinstance(v, str) else str(v)
                    for v in self.value
                )
            else:
                values = f"'{self.value}'" if isinstance(self.value, str) else str(self.value)
            return f"{col_ref} IN ({values})"

        if self.operator == FilterOperator.NOT_IN:
            if isinstance(self.value, list):
                values = ", ".join(
                    f"'{v}'" if isinstance(v, str) else str(v)
                    for v in self.value
                )
            else:
                values = f"'{self.value}'" if isinstance(self.value, str) else str(self.value)
            return f"{col_ref} NOT IN ({values})"

        if self.operator == FilterOperator.BETWEEN:
            if isinstance(self.value, list) and len(self.value) == 2:
                return f"{col_ref} BETWEEN {self.value[0]} AND {self.value[1]}"
            return f"{col_ref} = {self.value}"  # Fallback

        if self.operator.is_pattern():
            escaped_value = str(self.value).replace("'", "''")
            return f"{col_ref} {self.operator.value} '{escaped_value}'"

        # Standard comparison
        if isinstance(self.value, str):
            escaped_value = self.value.replace("'", "''")
            return f"{col_ref} {self.operator.value} '{escaped_value}'"

        return f"{col_ref} {self.operator.value} {self.value}"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "table": self.table,
            "column": self.column,
            "operator": self.operator.value,
            "value": self.value,
            "display_value": self.display_value,
            "original_text": self.original_text,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FilterNode":
        """Create from dictionary."""
        return cls(
            table=data["table"],
            column=data["column"],
            operator=FilterOperator.from_string(data["operator"]),
            value=data["value"],
            display_value=data.get("display_value"),
            original_text=data.get("original_text"),
            confidence=data.get("confidence", 1.0),
        )


class LogicalOperator(Enum):
    """Logical operators for combining filters."""
    AND = "AND"
    OR = "OR"


@dataclass
class FilterGroup:
    """
    AST node for grouped conditions (AND/OR).

    Allows complex nested conditions like:
    (bedrooms >= 3 AND bathrooms >= 2) OR (sqft > 2000)
    """
    operator: LogicalOperator
    children: List[Union[FilterNode, "FilterGroup"]] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.operator, str):
            self.operator = LogicalOperator(self.operator.upper())

    def add(self, child: Union[FilterNode, "FilterGroup"]) -> "FilterGroup":
        """Add a child node."""
        self.children.append(child)
        return self

    def is_empty(self) -> bool:
        return len(self.children) == 0

    def to_sql(self, use_aliases: bool = False) -> str:
        """Convert to SQL WHERE clause fragment."""
        if not self.children:
            return "1=1"  # Empty group = always true

        if len(self.children) == 1:
            child = self.children[0]
            if isinstance(child, FilterNode):
                return child.to_sql(use_aliases)
            return child.to_sql(use_aliases)

        parts = []
        for child in self.children:
            if isinstance(child, FilterNode):
                parts.append(child.to_sql(use_aliases))
            else:
                # Nested group - wrap in parentheses
                parts.append(f"({child.to_sql(use_aliases)})")

        return f" {self.operator.value} ".join(parts)

    def flatten(self) -> List[FilterNode]:
        """Get all FilterNodes in this group (recursive)."""
        nodes = []
        for child in self.children:
            if isinstance(child, FilterNode):
                nodes.append(child)
            else:
                nodes.extend(child.flatten())
        return nodes

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "operator": self.operator.value,
            "children": [
                child.to_dict() if isinstance(child, FilterNode)
                else child.to_dict()
                for child in self.children
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FilterGroup":
        """Create from dictionary."""
        children = []
        for child_data in data.get("children", []):
            if "children" in child_data:
                children.append(cls.from_dict(child_data))
            else:
                children.append(FilterNode.from_dict(child_data))
        return cls(
            operator=LogicalOperator(data["operator"]),
            children=children,
        )


@dataclass
class OrderByClause:
    """Represents ORDER BY clause."""
    column: str
    direction: str = "ASC"  # ASC or DESC
    table: Optional[str] = None

    def to_sql(self) -> str:
        if self.table:
            return f"{self.table}.{self.column} {self.direction}"
        return f"{self.column} {self.direction}"

    def to_dict(self) -> Dict:
        return {
            "column": self.column,
            "direction": self.direction,
            "table": self.table,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> Optional["OrderByClause"]:
        if not data:
            return None
        return cls(
            column=data["column"],
            direction=data.get("direction", "ASC"),
            table=data.get("table"),
        )


@dataclass
class FilterAST:
    """
    Complete Abstract Syntax Tree for a filter query.

    This is the main structure that represents a parsed natural language
    query, ready for validation and SQL generation.
    """
    tables: List[str]
    filters: FilterGroup
    order_by: Optional[OrderByClause] = None
    limit: int = 100
    offset: int = 0

    # Metadata
    original_query: Optional[str] = None
    interpretation: Optional[str] = None
    confidence: float = 1.0

    def to_sql(self, join_sql: str = "") -> str:
        """Generate complete SQL query."""
        # SELECT clause - for now, select all columns from first table
        if self.tables:
            select_cols = "*"
            from_clause = self.tables[0]
        else:
            select_cols = "*"
            from_clause = "dual"

        sql_parts = [f"SELECT {select_cols}"]

        # FROM clause with joins
        if join_sql:
            sql_parts.append(f"FROM {join_sql}")
        else:
            sql_parts.append(f"FROM {from_clause}")

        # WHERE clause
        if not self.filters.is_empty():
            sql_parts.append(f"WHERE {self.filters.to_sql()}")

        # ORDER BY
        if self.order_by:
            sql_parts.append(f"ORDER BY {self.order_by.to_sql()}")

        # LIMIT/OFFSET
        sql_parts.append(f"LIMIT {self.limit}")
        if self.offset > 0:
            sql_parts.append(f"OFFSET {self.offset}")

        return "\n".join(sql_parts)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tables": self.tables,
            "filters": self.filters.to_dict(),
            "order_by": self.order_by.to_dict() if self.order_by else None,
            "limit": self.limit,
            "offset": self.offset,
            "original_query": self.original_query,
            "interpretation": self.interpretation,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FilterAST":
        """Create from dictionary."""
        return cls(
            tables=data.get("tables", []),
            filters=FilterGroup.from_dict(data.get("filters", {"operator": "AND", "children": []})),
            order_by=OrderByClause.from_dict(data.get("order_by")),
            limit=data.get("limit", 100),
            offset=data.get("offset", 0),
            original_query=data.get("original_query"),
            interpretation=data.get("interpretation"),
            confidence=data.get("confidence", 1.0),
        )

    @classmethod
    def from_llm_response(cls, response: Dict, original_query: str = "") -> "FilterAST":
        """
        Parse LLM response into FilterAST.

        Expected response format:
        {
            "tables": ["transactions"],
            "filters": [
                {"table": "transactions", "column": "actual_worth", "operator": ">=", "value": "1000000"}
            ],
            "order_by": {"column": "actual_worth", "direction": "ASC"},
            "confidence": 0.95,
            "interpretation": "..."
        }
        """
        # Build filter group from flat list
        filter_group = FilterGroup(operator=LogicalOperator.AND)

        for f in response.get("filters", []):
            node = FilterNode(
                table=f.get("table", "transactions"),
                column=f["column"],
                operator=FilterOperator.from_string(f["operator"]),
                value=cls._parse_value(f["value"]),
                confidence=f.get("confidence", 1.0),
            )
            filter_group.add(node)

        # Parse order_by
        order_by_data = response.get("order_by")
        order_by = None
        if order_by_data:
            col = order_by_data.get("column", "")
            # Handle "table.column" format
            if "." in col:
                table, column = col.rsplit(".", 1)
                order_by = OrderByClause(column=column, table=table, direction=order_by_data.get("direction", "ASC"))
            else:
                order_by = OrderByClause(column=col, direction=order_by_data.get("direction", "ASC"))

        return cls(
            tables=response.get("tables", ["transactions"]),
            filters=filter_group,
            order_by=order_by,
            original_query=original_query,
            interpretation=response.get("interpretation"),
            confidence=response.get("confidence", 1.0),
        )

    @staticmethod
    def _parse_value(value: Any) -> Any:
        """Parse and normalize filter value."""
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return value

        if isinstance(value, str):
            # Try to parse as number
            clean = value.replace(",", "").strip()
            try:
                if "." in clean:
                    return float(clean)
                return int(clean)
            except ValueError:
                return value

        return value


class DSLParser:
    """
    Parser for text-based Filter DSL.

    Syntax examples:
        transactions.actual_worth >= 1000000
        transactions.area_name_en = 'Business Bay' AND transactions.actual_worth < 5000000
        (rooms_en LIKE '%2%' OR property_type_en = 'Villa') AND actual_worth < 4000000
    """

    TOKEN_PATTERN = re.compile(r"""
        (\(|\))                           # Parentheses
        |(\bAND\b|\bOR\b)                 # Logical operators
        |(>=|<=|!=|<>|=|>|<)              # Comparison operators
        |(\bLIKE\b|\bIN\b|\bBETWEEN\b)    # Special operators
        |'([^']*)'                        # Quoted string
        |"([^"]*)"                        # Double quoted string
        |([\w.]+)                         # Identifiers
    """, re.VERBOSE | re.IGNORECASE)

    @classmethod
    def parse(cls, dsl_string: str) -> FilterAST:
        """Parse DSL string into FilterAST."""
        tokens = cls._tokenize(dsl_string)
        filters, idx = cls._parse_expression(tokens, 0)

        return FilterAST(
            tables=cls._extract_tables(filters),
            filters=filters if isinstance(filters, FilterGroup) else FilterGroup(LogicalOperator.AND, [filters]),
            original_query=dsl_string,
        )

    @classmethod
    def _tokenize(cls, dsl_string: str) -> List[str]:
        """Tokenize DSL string."""
        tokens = []
        for match in cls.TOKEN_PATTERN.finditer(dsl_string):
            for group in match.groups():
                if group:
                    tokens.append(group)
                    break
        return tokens

    @classmethod
    def _parse_expression(cls, tokens: List[str], idx: int) -> tuple:
        """Parse expression recursively."""
        left, idx = cls._parse_term(tokens, idx)

        while idx < len(tokens) and tokens[idx].upper() in ("AND", "OR"):
            op = LogicalOperator(tokens[idx].upper())
            idx += 1
            right, idx = cls._parse_term(tokens, idx)

            if isinstance(left, FilterGroup) and left.operator == op:
                left.add(right)
            else:
                left = FilterGroup(op, [left, right])

        return left, idx

    @classmethod
    def _parse_term(cls, tokens: List[str], idx: int) -> tuple:
        """Parse a single term (filter or grouped expression)."""
        if idx >= len(tokens):
            return FilterGroup(LogicalOperator.AND), idx

        if tokens[idx] == "(":
            # Grouped expression
            expr, idx = cls._parse_expression(tokens, idx + 1)
            if idx < len(tokens) and tokens[idx] == ")":
                idx += 1
            return expr, idx

        # Single filter: table.column op value
        if idx + 2 < len(tokens):
            col_ref = tokens[idx]
            op = tokens[idx + 1]
            value = tokens[idx + 2]

            # Parse column reference
            if "." in col_ref:
                table, column = col_ref.rsplit(".", 1)
            else:
                table = "transactions"
                column = col_ref

            node = FilterNode(
                table=table,
                column=column,
                operator=FilterOperator.from_string(op),
                value=value,
            )
            return node, idx + 3

        return FilterGroup(LogicalOperator.AND), idx

    @classmethod
    def _extract_tables(cls, node: Union[FilterNode, FilterGroup]) -> List[str]:
        """Extract all unique tables from filter tree."""
        tables = set()
        if isinstance(node, FilterNode):
            tables.add(node.table)
        elif isinstance(node, FilterGroup):
            for child in node.children:
                tables.update(cls._extract_tables(child))
        return list(tables)
