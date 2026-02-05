"""
Three-Layer Filter Validation

Inspired by Netflix's approach to ensuring LLM outputs are correct at:
1. SYNTACTIC level - follows grammar rules
2. SEMANTIC level - valid tables/columns exist
3. PRAGMATIC level - logically makes sense
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set, Any
import re

from .filter_dsl import (
    FilterAST,
    FilterNode,
    FilterGroup,
    FilterOperator,
    OrderByClause,
)


class ValidationLevel(Enum):
    """Validation levels matching Netflix's framework."""
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"
    PRAGMATIC = "pragmatic"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    level: ValidationLevel
    severity: str  # "error" or "warning"
    message: str
    location: Optional[str] = None  # e.g., "filters[0]"
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Complete validation result."""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    # Per-level results
    syntactic_valid: bool = True
    semantic_valid: bool = True
    pragmatic_valid: bool = True

    def add_issue(self, issue: ValidationIssue):
        self.issues.append(issue)
        if issue.severity == "error":
            self.valid = False
            if issue.level == ValidationLevel.SYNTACTIC:
                self.syntactic_valid = False
            elif issue.level == ValidationLevel.SEMANTIC:
                self.semantic_valid = False
            elif issue.level == ValidationLevel.PRAGMATIC:
                self.pragmatic_valid = False

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_dict(self) -> Dict:
        return {
            "valid": self.valid,
            "syntactic_valid": self.syntactic_valid,
            "semantic_valid": self.semantic_valid,
            "pragmatic_valid": self.pragmatic_valid,
            "errors": [
                {
                    "level": i.level.value,
                    "message": i.message,
                    "location": i.location,
                    "suggestion": i.suggestion,
                }
                for i in self.errors
            ],
            "warnings": [
                {
                    "level": i.level.value,
                    "message": i.message,
                    "location": i.location,
                    "suggestion": i.suggestion,
                }
                for i in self.warnings
            ],
        }


class SchemaRegistry:
    """
    Registry of known tables, columns, and their types.
    Used for semantic validation.
    """

    def __init__(self):
        self.tables: Dict[str, Dict[str, Any]] = {}
        self.column_types: Dict[str, Dict[str, str]] = {}
        self.valid_values: Dict[str, Dict[str, List]] = {}  # column -> allowed values
        self.relationships: List[Tuple[str, str, str, str]] = []  # FK relationships

    def register_table(
        self,
        table_name: str,
        columns: Dict[str, str],  # column_name -> type
        description: str = "",
        valid_values: Optional[Dict[str, List]] = None,
    ):
        """Register a table and its columns."""
        self.tables[table_name] = {
            "columns": list(columns.keys()),
            "description": description,
        }
        self.column_types[table_name] = columns
        if valid_values:
            self.valid_values[table_name] = valid_values

    def register_relationship(
        self,
        from_table: str,
        from_column: str,
        to_table: str,
        to_column: str,
    ):
        """Register a foreign key relationship."""
        self.relationships.append((from_table, from_column, to_table, to_column))

    def has_table(self, table: str) -> bool:
        return table in self.tables

    def has_column(self, table: str, column: str) -> bool:
        if table not in self.tables:
            return False
        return column in self.tables[table]["columns"]

    def get_column_type(self, table: str, column: str) -> Optional[str]:
        if table not in self.column_types:
            return None
        return self.column_types[table].get(column)

    def get_valid_values(self, table: str, column: str) -> Optional[List]:
        if table not in self.valid_values:
            return None
        return self.valid_values[table].get(column)

    def get_similar_column(self, table: str, column: str) -> Optional[str]:
        """Find similar column name (for typo suggestions)."""
        if table not in self.tables:
            return None

        columns = self.tables[table]["columns"]
        column_lower = column.lower()

        # Exact match ignoring case
        for col in columns:
            if col.lower() == column_lower:
                return col

        # Partial match
        for col in columns:
            if column_lower in col.lower() or col.lower() in column_lower:
                return col

        return None

    @classmethod
    def from_schema_definition(cls, schema: List[Dict]) -> "SchemaRegistry":
        """Create registry from schema definition list."""
        registry = cls()
        for table_info in schema:
            columns = {}
            for col in table_info.get("columns", []):
                col_name = col["name"] if isinstance(col, dict) else col
                col_type = col.get("type", "TEXT") if isinstance(col, dict) else "TEXT"
                columns[col_name] = col_type

            registry.register_table(
                table_info["name"],
                columns,
                table_info.get("description", ""),
            )
        return registry


class FilterValidator:
    """
    Netflix-inspired three-layer validator for filter ASTs.
    """

    # Column types that support LIKE operator
    TEXT_TYPES = {"TEXT", "VARCHAR", "CHAR", "STRING"}

    # Column types that support numeric comparisons
    NUMERIC_TYPES = {"INTEGER", "INT", "REAL", "FLOAT", "DOUBLE", "DECIMAL", "NUMBER"}

    def __init__(self, schema: SchemaRegistry):
        self.schema = schema

    def validate(self, ast: FilterAST, original_query: str = "") -> ValidationResult:
        """Run all validation levels."""
        result = ValidationResult(valid=True)

        # Level 1: Syntactic validation
        self._validate_syntactic(ast, result)

        # Level 2: Semantic validation
        self._validate_semantic(ast, result)

        # Level 3: Pragmatic validation
        self._validate_pragmatic(ast, original_query, result)

        return result

    def _validate_syntactic(self, ast: FilterAST, result: ValidationResult):
        """
        Syntactic validation - check structure and grammar.

        Checks:
        - Operators are valid for data types
        - Values match expected formats
        - Required fields are present
        """
        for i, node in enumerate(ast.filters.flatten()):
            location = f"filters[{i}]"

            # Check operator-value compatibility
            if node.operator.is_pattern():
                # LIKE requires string value with wildcards
                if not isinstance(node.value, str):
                    result.add_issue(ValidationIssue(
                        level=ValidationLevel.SYNTACTIC,
                        severity="error",
                        message=f"LIKE operator requires string value, got {type(node.value).__name__}",
                        location=location,
                        suggestion="Convert value to string with wildcards, e.g., '%value%'",
                    ))

            elif node.operator == FilterOperator.BETWEEN:
                # BETWEEN requires list of 2 values
                if not isinstance(node.value, list) or len(node.value) != 2:
                    result.add_issue(ValidationIssue(
                        level=ValidationLevel.SYNTACTIC,
                        severity="error",
                        message="BETWEEN operator requires exactly 2 values",
                        location=location,
                        suggestion="Provide a list with [min, max] values",
                    ))

            elif node.operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
                # IN requires list or single value
                pass  # Both are acceptable

            elif node.operator.is_null_check():
                # IS NULL / IS NOT NULL don't need value
                if node.value is not None:
                    result.add_issue(ValidationIssue(
                        level=ValidationLevel.SYNTACTIC,
                        severity="warning",
                        message=f"{node.operator.value} operator ignores provided value",
                        location=location,
                    ))

            # Check for empty values
            if node.value == "" and not node.operator.is_null_check():
                result.add_issue(ValidationIssue(
                    level=ValidationLevel.SYNTACTIC,
                    severity="warning",
                    message="Empty filter value may produce unexpected results",
                    location=location,
                ))

        # Validate ORDER BY
        if ast.order_by:
            if ast.order_by.direction.upper() not in ("ASC", "DESC"):
                result.add_issue(ValidationIssue(
                    level=ValidationLevel.SYNTACTIC,
                    severity="error",
                    message=f"Invalid ORDER BY direction: {ast.order_by.direction}",
                    location="order_by",
                    suggestion="Use 'ASC' or 'DESC'",
                ))

    def _validate_semantic(self, ast: FilterAST, result: ValidationResult):
        """
        Semantic validation - check against schema.

        Checks:
        - Tables exist
        - Columns exist in their tables
        - Value types match column types
        """
        # Validate tables
        for table in ast.tables:
            if not self.schema.has_table(table):
                similar = self._find_similar_table(table)
                result.add_issue(ValidationIssue(
                    level=ValidationLevel.SEMANTIC,
                    severity="error",
                    message=f"Table '{table}' does not exist",
                    suggestion=f"Did you mean '{similar}'?" if similar else None,
                ))

        # Validate filters
        for i, node in enumerate(ast.filters.flatten()):
            location = f"filters[{i}]"

            # Check table exists
            if not self.schema.has_table(node.table):
                similar = self._find_similar_table(node.table)
                result.add_issue(ValidationIssue(
                    level=ValidationLevel.SEMANTIC,
                    severity="error",
                    message=f"Table '{node.table}' in filter does not exist",
                    location=location,
                    suggestion=f"Did you mean '{similar}'?" if similar else None,
                ))
                continue

            # Check column exists
            if not self.schema.has_column(node.table, node.column):
                similar = self.schema.get_similar_column(node.table, node.column)
                result.add_issue(ValidationIssue(
                    level=ValidationLevel.SEMANTIC,
                    severity="error",
                    message=f"Column '{node.column}' not found in table '{node.table}'",
                    location=location,
                    suggestion=f"Did you mean '{similar}'?" if similar else None,
                ))
                continue

            # Check type compatibility
            col_type = self.schema.get_column_type(node.table, node.column)
            if col_type:
                self._validate_type_compatibility(node, col_type, location, result)

            # Check valid values if defined
            valid_vals = self.schema.get_valid_values(node.table, node.column)
            if valid_vals and node.operator == FilterOperator.EQ:
                if node.value not in valid_vals:
                    result.add_issue(ValidationIssue(
                        level=ValidationLevel.SEMANTIC,
                        severity="warning",
                        message=f"Value '{node.value}' may not be valid for {node.column}",
                        location=location,
                        suggestion=f"Valid values include: {', '.join(str(v) for v in valid_vals[:5])}",
                    ))

        # Validate ORDER BY column
        if ast.order_by:
            table = ast.order_by.table or (ast.tables[0] if ast.tables else None)
            if table and not self.schema.has_column(table, ast.order_by.column):
                result.add_issue(ValidationIssue(
                    level=ValidationLevel.SEMANTIC,
                    severity="error",
                    message=f"ORDER BY column '{ast.order_by.column}' not found",
                    location="order_by",
                ))

    def _validate_type_compatibility(
        self,
        node: FilterNode,
        col_type: str,
        location: str,
        result: ValidationResult,
    ):
        """Validate that operator and value are compatible with column type."""
        col_type_upper = col_type.upper()

        # LIKE on non-text column
        if node.operator.is_pattern() and col_type_upper not in self.TEXT_TYPES:
            result.add_issue(ValidationIssue(
                level=ValidationLevel.SEMANTIC,
                severity="warning",
                message=f"LIKE operator on non-text column '{node.column}' ({col_type})",
                location=location,
                suggestion="Consider using = or other comparison operators",
            ))

        # Numeric comparison on text column
        if node.operator.is_comparison() and node.operator not in [FilterOperator.EQ, FilterOperator.NE]:
            if col_type_upper in self.TEXT_TYPES and isinstance(node.value, (int, float)):
                result.add_issue(ValidationIssue(
                    level=ValidationLevel.SEMANTIC,
                    severity="warning",
                    message=f"Numeric comparison on text column '{node.column}'",
                    location=location,
                ))

        # String value for numeric column
        if col_type_upper in self.NUMERIC_TYPES:
            if isinstance(node.value, str) and not node.value.replace(".", "").replace("-", "").isdigit():
                result.add_issue(ValidationIssue(
                    level=ValidationLevel.SEMANTIC,
                    severity="error",
                    message=f"Non-numeric value '{node.value}' for numeric column '{node.column}'",
                    location=location,
                    suggestion="Provide a numeric value",
                ))

    def _validate_pragmatic(
        self,
        ast: FilterAST,
        original_query: str,
        result: ValidationResult,
    ):
        """
        Pragmatic validation - check logical consistency.

        Checks:
        - Contradictory filters (price > 500000 AND price < 300000)
        - Impossible ranges
        - Duplicate filters
        - Missing expected filters based on query
        """
        filters = ast.filters.flatten()

        # Group filters by column
        filters_by_column: Dict[str, List[FilterNode]] = {}
        for node in filters:
            key = f"{node.table}.{node.column}"
            filters_by_column.setdefault(key, []).append(node)

        # Check for contradictions
        for key, nodes in filters_by_column.items():
            if len(nodes) < 2:
                continue

            # Find min/max bounds
            lower_bounds = []
            upper_bounds = []
            exact_values = []

            for node in nodes:
                if node.operator in [FilterOperator.GT, FilterOperator.GTE]:
                    if isinstance(node.value, (int, float)):
                        lower_bounds.append(node.value)
                elif node.operator in [FilterOperator.LT, FilterOperator.LTE]:
                    if isinstance(node.value, (int, float)):
                        upper_bounds.append(node.value)
                elif node.operator == FilterOperator.EQ:
                    exact_values.append(node.value)

            # Check for impossible ranges
            if lower_bounds and upper_bounds:
                max_lower = max(lower_bounds)
                min_upper = min(upper_bounds)
                if max_lower >= min_upper:
                    result.add_issue(ValidationIssue(
                        level=ValidationLevel.PRAGMATIC,
                        severity="error",
                        message=f"Contradictory range on {key}: values must be > {max_lower} AND < {min_upper}",
                        suggestion="Check your filter ranges",
                    ))

            # Check for multiple exact values
            if len(exact_values) > 1 and len(set(str(v) for v in exact_values)) > 1:
                result.add_issue(ValidationIssue(
                    level=ValidationLevel.PRAGMATIC,
                    severity="error",
                    message=f"Multiple contradictory exact values for {key}",
                    suggestion="Use IN operator for multiple values, or use OR logic",
                ))

        # Check for duplicate filters
        seen_filters = set()
        for i, node in enumerate(filters):
            sig = f"{node.table}.{node.column}{node.operator.value}{node.value}"
            if sig in seen_filters:
                result.add_issue(ValidationIssue(
                    level=ValidationLevel.PRAGMATIC,
                    severity="warning",
                    message=f"Duplicate filter detected: {node.table}.{node.column}",
                    location=f"filters[{i}]",
                ))
            seen_filters.add(sig)

        # Check if query mentions things not captured
        if original_query:
            self._check_query_coverage(ast, original_query, result)

    def _check_query_coverage(
        self,
        ast: FilterAST,
        query: str,
        result: ValidationResult,
    ):
        """Check if important query terms are captured in filters."""
        query_lower = query.lower()
        filter_columns = {n.column.lower() for n in ast.filters.flatten()}

        # Keywords that should typically result in filters
        expected_mappings = {
            "bedroom": "bedrooms",
            "bed": "bedrooms",
            "bathroom": "bathrooms",
            "bath": "bathrooms",
            "price": "list_price",
            "under": "list_price",
            "over": "list_price",
            "sqft": "sqft",
            "square feet": "sqft",
        }

        for keyword, column in expected_mappings.items():
            if keyword in query_lower and column.lower() not in filter_columns:
                result.add_issue(ValidationIssue(
                    level=ValidationLevel.PRAGMATIC,
                    severity="warning",
                    message=f"Query mentions '{keyword}' but no filter on '{column}'",
                    suggestion=f"Consider adding a filter on {column}",
                ))

    def _find_similar_table(self, table: str) -> Optional[str]:
        """Find similar table name (for typo suggestions)."""
        table_lower = table.lower()
        for known in self.schema.tables.keys():
            if table_lower in known.lower() or known.lower() in table_lower:
                return known
        return None


def create_default_registry(schema_definition: List[Dict]) -> SchemaRegistry:
    """Create a default schema registry with common configurations."""
    registry = SchemaRegistry.from_schema_definition(schema_definition)

    # Add common valid values for Dubai Proptech data
    if "transactions" in registry.tables:
        registry.valid_values["transactions"] = {
            "trans_group_en": ["Sales", "Mortgages", "Gifts"],
            "property_type_en": ["Villa", "Unit", "Land", "Building"],
            "property_usage_en": ["Residential", "Commercial", "Residential / Commercial"],
        }
    if "bayut_transactions" in registry.tables:
        registry.valid_values["bayut_transactions"] = {
            "completion_status": ["Ready", "Off Plan"],
            "transaction_category": ["Sales", "Rentals"],
            "seller_type": ["Developer", "Individual"],
        }

    return registry
