"""
Netflix-Inspired LLM Layer for Dynamic Filtering

A production-grade natural language to SQL filter extraction system
featuring AST-based parsing, multi-layer validation, entity resolution,
and intelligent fallback strategies.
"""

from .filter_dsl import (
    FilterOperator,
    FilterNode,
    FilterGroup,
    LogicalOperator,
    OrderByClause,
    FilterAST,
)
from .validator import (
    ValidationLevel,
    ValidationResult,
    FilterValidator,
)
from .entity_resolver import (
    Entity,
    EntityType,
    EntityResolver,
)
from .smart_extractor import (
    ExtractionStrategy,
    ExtractionResult,
    SmartExtractor,
)
from .chip_generator import (
    UIChip,
    ChipType,
    ChipGenerator,
)
from .cache import ExtractionCache
from .metrics import MetricsCollector, ExtractionMetrics
from .feedback_store import FeedbackStore, FeedbackRecord

__all__ = [
    # Filter DSL
    "FilterOperator",
    "FilterNode",
    "FilterGroup",
    "LogicalOperator",
    "OrderByClause",
    "FilterAST",
    # Validation
    "ValidationLevel",
    "ValidationResult",
    "FilterValidator",
    # Entity Resolution
    "Entity",
    "EntityType",
    "EntityResolver",
    # Smart Extraction
    "ExtractionStrategy",
    "ExtractionResult",
    "SmartExtractor",
    # UI Chips
    "UIChip",
    "ChipType",
    "ChipGenerator",
    # Utilities
    "ExtractionCache",
    "MetricsCollector",
    "ExtractionMetrics",
    "FeedbackStore",
    "FeedbackRecord",
]

__version__ = "1.0.0"
