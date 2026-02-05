"""
Feedback Store for LLM extraction overrides and examples.

Stores JSONL records that can be used to:
1) Override extraction for exact queries
2) Provide additional prompt examples
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import os
import re


@dataclass
class FeedbackRecord:
    query: str
    tables: List[str]
    filters: List[Dict[str, Any]]
    order_by: Optional[Dict[str, Any]] = None
    confidence: float = 0.95
    interpretation: Optional[str] = None
    include_in_prompt: bool = True

    def to_llm_response(self) -> Dict[str, Any]:
        return {
            "tables": self.tables,
            "filters": self.filters,
            "order_by": self.order_by,
            "confidence": self.confidence,
            "interpretation": self.interpretation or "",
        }


class FeedbackStore:
    """Load feedback records from a JSONL file."""

    def __init__(self, path: Optional[str] = None):
        default_path = os.environ.get("FEEDBACK_FILE", "feedback/llm_feedback.jsonl")
        self.path = Path(path or default_path)

    def _normalize_query(self, query: str) -> str:
        normalized = query.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _load_records(self) -> List[FeedbackRecord]:
        if not self.path.exists():
            return []
        records: List[FeedbackRecord] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "query" not in data or "filters" not in data or "tables" not in data:
                    continue
                record = FeedbackRecord(
                    query=data["query"],
                    tables=data["tables"],
                    filters=data["filters"],
                    order_by=data.get("order_by"),
                    confidence=float(data.get("confidence", 0.95)),
                    interpretation=data.get("interpretation"),
                    include_in_prompt=bool(data.get("include_in_prompt", True)),
                )
                records.append(record)
        return records

    def get_prompt_examples(self, limit: int = 5) -> List[FeedbackRecord]:
        records = [r for r in self._load_records() if r.include_in_prompt]
        return records[:limit]

    def get_override(self, query: str) -> Optional[FeedbackRecord]:
        target = self._normalize_query(query)
        for record in self._load_records():
            if self._normalize_query(record.query) == target:
                return record
        return None
