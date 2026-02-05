"""
UI Chip Generator

Inspired by Netflix's approach of mapping AST filters to visual
"Chips" and "Facets" that users can see and modify in the UI.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any

from .filter_dsl import (
    FilterAST,
    FilterNode,
    FilterGroup,
    FilterOperator,
    OrderByClause,
)
from .smart_extractor import ExtractionResult


class ChipType(Enum):
    """Types of UI chips."""
    ENTITY = "entity"         # City, state, etc.
    RANGE = "range"           # Price range, sqft range
    COMPARISON = "comparison" # bedrooms >= 3
    SORT = "sort"            # Order by
    STATUS = "status"         # Active, Sold, etc.
    PROPERTY_TYPE = "property_type"
    KEYWORD = "keyword"       # Generic keyword filter


@dataclass
class UIChip:
    """
    Visual representation of a filter for the UI.

    Designed to be easily rendered as interactive chips/tags
    that users can click to remove or modify.
    """
    id: str
    display_text: str
    chip_type: ChipType
    color: str = "blue"  # UI hint for styling

    # Filter data
    filter_node: Optional[FilterNode] = None
    order_by: Optional[OrderByClause] = None

    # Interaction flags
    removable: bool = True
    editable: bool = True

    # Metadata
    confidence: float = 1.0
    original_text: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "display_text": self.display_text,
            "chip_type": self.chip_type.value,
            "color": self.color,
            "removable": self.removable,
            "editable": self.editable,
            "confidence": self.confidence,
            "original_text": self.original_text,
            "filter": self.filter_node.to_dict() if self.filter_node else None,
            "order_by": self.order_by.to_dict() if self.order_by else None,
        }


@dataclass
class ChipGroup:
    """Group of related chips (e.g., all location chips)."""
    name: str
    chips: List[UIChip] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "chips": [c.to_dict() for c in self.chips],
        }


class ChipGenerator:
    """
    Generate UI chips from extraction results.

    Features:
    - Human-readable display text
    - Appropriate chip types and colors
    - Grouping by category
    - Edit/remove capabilities
    """

    # Column display labels
    COLUMN_LABELS = {
        "list_price": "Price",
        "sold_price": "Sold Price",
        "bedrooms": "Beds",
        "bathrooms": "Baths",
        "sqft": "Sq Ft",
        "city": "City",
        "state_code": "State",
        "status": "Status",
        "property_type": "Type",
        "neighborhood": "Neighborhood",
        "zip_code": "ZIP",
        "year_built": "Year Built",
        "days_on_market": "Days on Market",
        "lot_size": "Lot Size",
    }

    # Chip type mapping by column
    COLUMN_CHIP_TYPES = {
        "city": ChipType.ENTITY,
        "state_code": ChipType.ENTITY,
        "neighborhood": ChipType.ENTITY,
        "zip_code": ChipType.ENTITY,
        "status": ChipType.STATUS,
        "property_type": ChipType.PROPERTY_TYPE,
        "list_price": ChipType.RANGE,
        "sold_price": ChipType.RANGE,
        "sqft": ChipType.RANGE,
        "bedrooms": ChipType.COMPARISON,
        "bathrooms": ChipType.COMPARISON,
    }

    # Color mapping by chip type
    CHIP_COLORS = {
        ChipType.ENTITY: "blue",
        ChipType.RANGE: "green",
        ChipType.COMPARISON: "purple",
        ChipType.SORT: "orange",
        ChipType.STATUS: "teal",
        ChipType.PROPERTY_TYPE: "pink",
        ChipType.KEYWORD: "gray",
    }

    # Status display names
    STATUS_NAMES = {
        "A": "Active",
        "S": "Sold",
        "P": "Pending",
        "W": "Withdrawn",
        "C": "Cancelled",
    }

    def generate(self, result: ExtractionResult) -> List[UIChip]:
        """Generate chips from extraction result."""
        chips = []

        # Generate filter chips
        for i, node in enumerate(result.ast.filters.flatten()):
            chip = self._create_filter_chip(node, i)
            chips.append(chip)

        # Generate sort chip
        if result.ast.order_by:
            chips.append(self._create_sort_chip(result.ast.order_by))

        return chips

    def generate_grouped(self, result: ExtractionResult) -> List[ChipGroup]:
        """Generate chips organized into groups."""
        groups = {
            "Location": ChipGroup("Location"),
            "Property": ChipGroup("Property"),
            "Price": ChipGroup("Price"),
            "Status": ChipGroup("Status"),
            "Sort": ChipGroup("Sort"),
            "Other": ChipGroup("Other"),
        }

        chips = self.generate(result)

        for chip in chips:
            if chip.chip_type == ChipType.ENTITY:
                groups["Location"].chips.append(chip)
            elif chip.chip_type == ChipType.RANGE:
                if chip.filter_node and chip.filter_node.column in ["list_price", "sold_price"]:
                    groups["Price"].chips.append(chip)
                else:
                    groups["Property"].chips.append(chip)
            elif chip.chip_type == ChipType.STATUS:
                groups["Status"].chips.append(chip)
            elif chip.chip_type == ChipType.SORT:
                groups["Sort"].chips.append(chip)
            elif chip.chip_type == ChipType.COMPARISON:
                groups["Property"].chips.append(chip)
            else:
                groups["Other"].chips.append(chip)

        # Return non-empty groups
        return [g for g in groups.values() if g.chips]

    def _create_filter_chip(self, node: FilterNode, index: int) -> UIChip:
        """Create a chip from a filter node."""
        chip_type = self.COLUMN_CHIP_TYPES.get(node.column, ChipType.KEYWORD)
        color = self.CHIP_COLORS.get(chip_type, "gray")
        display_text = self._format_display_text(node)

        return UIChip(
            id=f"filter_{index}",
            display_text=display_text,
            chip_type=chip_type,
            color=color,
            filter_node=node,
            confidence=node.confidence,
            original_text=node.original_text,
        )

    def _create_sort_chip(self, order_by: OrderByClause) -> UIChip:
        """Create a chip for ORDER BY clause."""
        label = self.COLUMN_LABELS.get(order_by.column, order_by.column.replace("_", " ").title())
        direction = "↑" if order_by.direction == "ASC" else "↓"
        display_text = f"Sort: {label} {direction}"

        return UIChip(
            id="sort_0",
            display_text=display_text,
            chip_type=ChipType.SORT,
            color=self.CHIP_COLORS[ChipType.SORT],
            order_by=order_by,
        )

    def _format_display_text(self, node: FilterNode) -> str:
        """Format human-readable display text for a filter."""
        label = self.COLUMN_LABELS.get(node.column, node.column.replace("_", " ").title())
        value = self._format_value(node.value, node.column)

        # Special handling for status
        if node.column == "status" and node.value in self.STATUS_NAMES:
            return self.STATUS_NAMES[node.value]

        # Format based on operator
        if node.operator == FilterOperator.EQ:
            return f"{label}: {value}"
        elif node.operator == FilterOperator.NE:
            return f"{label}: not {value}"
        elif node.operator == FilterOperator.GTE:
            return f"{label}: {value}+"
        elif node.operator == FilterOperator.LTE:
            return f"{label}: up to {value}"
        elif node.operator == FilterOperator.GT:
            return f"{label}: > {value}"
        elif node.operator == FilterOperator.LT:
            return f"{label}: < {value}"
        elif node.operator == FilterOperator.LIKE:
            # Remove wildcards for display
            clean_value = str(node.value).replace("%", "").strip()
            return f"{label}: contains \"{clean_value}\""
        elif node.operator == FilterOperator.IN:
            if isinstance(node.value, list):
                values = ", ".join(str(v) for v in node.value[:3])
                if len(node.value) > 3:
                    values += f" +{len(node.value) - 3}"
            else:
                values = str(node.value)
            return f"{label}: {values}"
        elif node.operator == FilterOperator.BETWEEN:
            if isinstance(node.value, list) and len(node.value) == 2:
                v1 = self._format_value(node.value[0], node.column)
                v2 = self._format_value(node.value[1], node.column)
                return f"{label}: {v1} - {v2}"
        else:
            return f"{label} {node.operator.value} {value}"

        return f"{label}: {value}"

    def _format_value(self, value: Any, column: str) -> str:
        """Format value for display."""
        if value is None:
            return "N/A"

        # Price formatting
        if column in ["list_price", "sold_price"]:
            try:
                num_val = float(value)
                if num_val >= 1000000:
                    return f"${num_val/1000000:.1f}M"
                elif num_val >= 1000:
                    return f"${num_val/1000:.0f}K"
                else:
                    return f"${num_val:,.0f}"
            except (ValueError, TypeError):
                return str(value)

        # Sqft formatting
        if column == "sqft":
            try:
                return f"{int(value):,}"
            except (ValueError, TypeError):
                return str(value)

        return str(value)


class ChipEditor:
    """
    Handle chip editing operations.

    Allows users to modify filter values through the chip UI.
    """

    def update_chip_value(self, chip: UIChip, new_value: Any) -> UIChip:
        """Update a chip's filter value."""
        if not chip.filter_node:
            return chip

        # Create updated filter node
        updated_node = FilterNode(
            table=chip.filter_node.table,
            column=chip.filter_node.column,
            operator=chip.filter_node.operator,
            value=new_value,
            display_value=str(new_value),
            original_text=chip.original_text,
        )

        generator = ChipGenerator()
        new_display = generator._format_display_text(updated_node)

        return UIChip(
            id=chip.id,
            display_text=new_display,
            chip_type=chip.chip_type,
            color=chip.color,
            filter_node=updated_node,
            removable=chip.removable,
            editable=chip.editable,
        )

    def update_chip_operator(self, chip: UIChip, new_operator: str) -> UIChip:
        """Update a chip's filter operator."""
        if not chip.filter_node:
            return chip

        updated_node = FilterNode(
            table=chip.filter_node.table,
            column=chip.filter_node.column,
            operator=FilterOperator.from_string(new_operator),
            value=chip.filter_node.value,
            original_text=chip.original_text,
        )

        generator = ChipGenerator()
        new_display = generator._format_display_text(updated_node)

        return UIChip(
            id=chip.id,
            display_text=new_display,
            chip_type=chip.chip_type,
            color=chip.color,
            filter_node=updated_node,
            removable=chip.removable,
            editable=chip.editable,
        )

    def chips_to_ast(self, chips: List[UIChip], original_tables: List[str] = None) -> FilterAST:
        """Convert chips back to FilterAST."""
        from .filter_dsl import LogicalOperator

        filters = FilterGroup(LogicalOperator.AND)
        tables = set(original_tables or ["transactions"])
        order_by = None

        for chip in chips:
            if chip.filter_node:
                filters.add(chip.filter_node)
                tables.add(chip.filter_node.table)
            if chip.order_by:
                order_by = chip.order_by

        return FilterAST(
            tables=list(tables),
            filters=filters,
            order_by=order_by,
        )


def chips_to_json(chips: List[UIChip]) -> List[Dict]:
    """Convert chips to JSON-serializable format."""
    return [chip.to_dict() for chip in chips]


def json_to_chips(data: List[Dict]) -> List[UIChip]:
    """Convert JSON data back to UIChip objects."""
    chips = []
    for item in data:
        filter_node = None
        if item.get("filter"):
            filter_node = FilterNode.from_dict(item["filter"])

        order_by = None
        if item.get("order_by"):
            order_by = OrderByClause.from_dict(item["order_by"])

        chip = UIChip(
            id=item["id"],
            display_text=item["display_text"],
            chip_type=ChipType(item["chip_type"]),
            color=item.get("color", "gray"),
            filter_node=filter_node,
            order_by=order_by,
            removable=item.get("removable", True),
            editable=item.get("editable", True),
            confidence=item.get("confidence", 1.0),
            original_text=item.get("original_text"),
        )
        chips.append(chip)

    return chips
