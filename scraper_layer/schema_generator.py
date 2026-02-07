"""
Schema Generator - AI creates clean database schemas from raw scraped data.

Takes messy, inconsistent scraped data and produces:
1. Clean column names (snake_case, descriptive)
2. Correct data types (TEXT, INTEGER, REAL, DATE)
3. Primary key detection
4. Data cleaning/normalization
5. Relationship detection (if multiple tables)
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class ColumnSchema:
    name: str
    original_name: str
    data_type: str  # TEXT, INTEGER, REAL, DATE, BOOLEAN
    description: str = ""
    is_primary_key: bool = False
    is_nullable: bool = True
    sample_values: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class TableSchema:
    table_name: str
    description: str
    columns: List[ColumnSchema] = field(default_factory=list)
    row_count: int = 0
    source_url: str = ""

    def to_dict(self):
        return {
            "table_name": self.table_name,
            "description": self.description,
            "columns": [c.to_dict() for c in self.columns],
            "row_count": self.row_count,
            "source_url": self.source_url,
        }


class SchemaGenerator:
    """
    Generate clean database schemas from raw scraped data.

    Usage:
        gen = SchemaGenerator(llm_model=gemini_model)
        schema = gen.generate(data, source_url="https://...")
        clean_data = gen.clean_data(data, schema)
    """

    def __init__(self, llm_model=None):
        self.llm = llm_model

    def generate(
        self,
        data: List[Dict[str, Any]],
        source_url: str = "",
        description: str = "",
    ) -> TableSchema:
        """Generate schema from raw data."""
        if not data:
            return TableSchema(table_name="empty", description="No data")

        # Step 1: Analyze data types from actual values
        column_analysis = self._analyze_columns(data)

        # Step 2: Use AI to enhance schema (better names, descriptions)
        if self.llm:
            schema = self._ai_enhance_schema(column_analysis, data, source_url, description)
        else:
            schema = self._basic_schema(column_analysis, source_url)

        schema.row_count = len(data)
        schema.source_url = source_url
        return schema

    def clean_data(
        self, data: List[Dict[str, Any]], schema: TableSchema
    ) -> List[Dict[str, Any]]:
        """Clean and normalize data according to schema."""
        if not data or not schema.columns:
            return data

        # Build column mapping: original_name -> ColumnSchema
        col_map = {}
        for col in schema.columns:
            col_map[col.original_name] = col

        cleaned = []
        for record in data:
            clean_record = {}
            for orig_name, col_schema in col_map.items():
                value = record.get(orig_name)
                if value is None:
                    clean_record[col_schema.name] = None
                    continue

                # Type conversion
                clean_record[col_schema.name] = self._convert_value(
                    value, col_schema.data_type
                )

            # Also include any unmapped fields
            for key, value in record.items():
                if key not in col_map:
                    clean_key = self._to_snake_case(key)
                    if clean_key not in clean_record:
                        clean_record[clean_key] = value

            cleaned.append(clean_record)

        return cleaned

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _analyze_columns(self, data: List[Dict]) -> Dict[str, Dict]:
        """Analyze column types from actual data values."""
        analysis = {}
        sample_size = min(50, len(data))

        for record in data[:sample_size]:
            if not isinstance(record, dict):
                continue
            for key, value in record.items():
                if key not in analysis:
                    analysis[key] = {
                        "name": key,
                        "types_seen": set(),
                        "non_null_count": 0,
                        "total_count": 0,
                        "samples": [],
                        "max_length": 0,
                    }

                col = analysis[key]
                col["total_count"] += 1

                if value is not None and value != "":
                    col["non_null_count"] += 1
                    col["types_seen"].add(type(value).__name__)

                    if len(col["samples"]) < 5:
                        col["samples"].append(str(value)[:100])

                    str_len = len(str(value))
                    if str_len > col["max_length"]:
                        col["max_length"] = str_len

        # Determine data types
        for key, col in analysis.items():
            col["inferred_type"] = self._infer_type(col)

        return analysis

    def _infer_type(self, col: Dict) -> str:
        """Infer SQL data type from column analysis."""
        types_seen = col.get("types_seen", set())
        samples = col.get("samples", [])

        # Check if all values are int
        if types_seen == {"int"}:
            return "INTEGER"

        # Check if all values are numeric
        if types_seen <= {"int", "float"}:
            return "REAL"

        # Check if values look like booleans
        if types_seen <= {"bool"}:
            return "BOOLEAN"

        # Check samples for patterns
        if samples:
            # Date patterns
            date_patterns = [
                r'^\d{4}-\d{2}-\d{2}',
                r'^\d{2}/\d{2}/\d{4}',
                r'^\w+\s+\d{1,2},?\s+\d{4}',
            ]
            all_dates = all(
                any(re.match(p, s) for p in date_patterns)
                for s in samples
                if s
            )
            if all_dates and samples:
                return "DATE"

            # Price/currency
            all_prices = all(re.match(r'^\$[\d,]+', s) for s in samples if s)
            if all_prices and samples:
                return "REAL"

            # Numbers stored as strings
            all_numeric = all(
                re.match(r'^-?[\d,]+\.?\d*$', s.replace(",", ""))
                for s in samples
                if s
            )
            if all_numeric and samples:
                return "REAL"

        return "TEXT"

    def _basic_schema(self, analysis: Dict, source_url: str) -> TableSchema:
        """Generate basic schema without AI."""
        columns = []
        for key, col in analysis.items():
            columns.append(
                ColumnSchema(
                    name=self._to_snake_case(key),
                    original_name=key,
                    data_type=col["inferred_type"],
                    is_nullable=col["non_null_count"] < col["total_count"],
                    sample_values=col["samples"],
                )
            )

        # Detect primary key (unique ID-like column)
        for col in columns:
            if any(pk in col.name.lower() for pk in ["_id", "id", "key", "code"]):
                col.is_primary_key = True
                break

        return TableSchema(
            table_name="scraped_data",
            description=f"Data scraped from {source_url}",
            columns=columns,
        )

    def _ai_enhance_schema(
        self,
        analysis: Dict,
        data: List[Dict],
        source_url: str,
        description: str,
    ) -> TableSchema:
        """Use AI to generate a better schema with descriptions."""
        sample = json.dumps(data[:3], indent=2, default=str)
        col_info = json.dumps(
            {k: {"type": v["inferred_type"], "samples": v["samples"]}
             for k, v in analysis.items()},
            indent=2,
        )

        prompt = f"""You are a data engineer. Generate a clean database schema for this scraped data.

Source: {source_url}
Description: {description or 'Scraped web data'}

Column analysis:
{col_info}

Sample data (3 records):
{sample}

Respond with ONLY valid JSON:
{{
    "table_name": "clean_table_name_snake_case",
    "description": "What this data represents",
    "columns": [
        {{
            "name": "clean_column_name",
            "original_name": "original_key_from_data",
            "data_type": "TEXT|INTEGER|REAL|DATE|BOOLEAN",
            "description": "What this column contains",
            "is_primary_key": false
        }}
    ]
}}

RULES:
- Use snake_case for all names
- Choose the most appropriate SQL data type
- Add clear descriptions
- Identify the primary key if one exists
- Remove duplicate/redundant columns
- Rename vague columns to be descriptive"""

        try:
            response = self.llm.generate_content(prompt)
            text = response.text.strip()

            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                schema_data = json.loads(json_match.group())

                columns = []
                for col_data in schema_data.get("columns", []):
                    columns.append(
                        ColumnSchema(
                            name=col_data.get("name", "unknown"),
                            original_name=col_data.get("original_name", col_data.get("name", "")),
                            data_type=col_data.get("data_type", "TEXT"),
                            description=col_data.get("description", ""),
                            is_primary_key=col_data.get("is_primary_key", False),
                        )
                    )

                return TableSchema(
                    table_name=schema_data.get("table_name", "scraped_data"),
                    description=schema_data.get("description", ""),
                    columns=columns,
                )
        except Exception as e:
            print(f"  [schema_gen] AI schema generation failed: {e}")

        return self._basic_schema(analysis, source_url)

    def _convert_value(self, value: Any, target_type: str) -> Any:
        """Convert a value to the target type."""
        if value is None:
            return None

        try:
            if target_type == "INTEGER":
                if isinstance(value, str):
                    value = value.replace(",", "").replace("$", "").strip()
                return int(float(value))
            elif target_type == "REAL":
                if isinstance(value, str):
                    value = value.replace(",", "").replace("$", "").strip()
                return float(value)
            elif target_type == "BOOLEAN":
                if isinstance(value, str):
                    return value.lower() in ("true", "yes", "1", "on")
                return bool(value)
            elif target_type == "DATE":
                return str(value)
            else:
                return str(value)
        except (ValueError, TypeError):
            return str(value) if value else None

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert any string to clean snake_case."""
        # Handle camelCase
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
        # Replace non-alphanumeric with underscore
        name = re.sub(r'[^a-zA-Z0-9]', '_', name)
        # Remove multiple underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name.lower()
