"""
Entity Resolution with @Mentions Support

Inspired by Netflix's approach to handling ambiguity by letting users
constrain input to known entities using @mentions (similar to Slack).
Provides access to controlled vocabularies for cities, states, etc.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import re
from difflib import SequenceMatcher


class EntityType(Enum):
    """Types of entities we can resolve."""
    CITY = "city"
    STATE = "state"
    NEIGHBORHOOD = "neighborhood"
    PROPERTY_TYPE = "property_type"
    STATUS = "status"
    AGENT = "agent"
    AMENITY = "amenity"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """A resolved entity with type and canonical form."""
    raw_text: str
    entity_type: EntityType
    canonical_value: str
    confidence: float
    table: str
    column: str

    # Additional metadata
    display_name: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)
    is_explicit_mention: bool = False  # True if user used @mention

    def to_dict(self) -> Dict:
        return {
            "raw_text": self.raw_text,
            "entity_type": self.entity_type.value,
            "canonical_value": self.canonical_value,
            "confidence": self.confidence,
            "table": self.table,
            "column": self.column,
            "display_name": self.display_name or self.canonical_value,
            "is_explicit_mention": self.is_explicit_mention,
        }


@dataclass
class EntityMatch:
    """A potential entity match with position info."""
    text: str
    start: int
    end: int
    entity: Optional[Entity] = None


class EntityResolver:
    """
    Netflix-style entity resolution with @mention support.

    Features:
    - @mention syntax for explicit entity references
    - Fuzzy matching against controlled vocabularies
    - Autocomplete suggestions for UI
    - Disambiguation handling
    """

    # Entity type to table.column mapping for Dubai Proptech
    TYPE_MAPPING = {
        EntityType.CITY: ("lkp_areas", "name_en"),  # Dubai areas
        EntityType.STATE: ("transactions", "trans_group_en"),  # Transaction types
        EntityType.NEIGHBORHOOD: ("transactions", "area_name_en"),  # Areas
        EntityType.PROPERTY_TYPE: ("transactions", "property_type_en"),
        EntityType.STATUS: ("bayut_transactions", "completion_status"),
    }

    # Common abbreviations and aliases for Dubai
    # NOTE: Area names map to actual values in the database
    ALIASES = {
        # Area abbreviations - mapped to actual DB values
        "jvc": "Jumeirah Village Circle",
        "jlt": "Jumeirah Lake Towers",
        "jbr": "Jumeirah Beach Residence",
        "difc": "DIFC",
        "dip": "Dubai Investment Park",
        "dso": "Dubai Silicon Oasis",
        "marina": "Marsa Dubai",  # Dubai Marina = Marsa Dubai in DB
        "dubai marina": "Marsa Dubai",
        "downtown": "Burj Khalifa",  # Downtown Dubai = Burj Khalifa area in DB
        "downtown dubai": "Burj Khalifa",
        "business bay": "Business Bay",
        "palm": "Palm Jumeirah",
        "palm jumeirah": "Palm Jumeirah",
        "mirdif": "Mirdif",
        "al barsha": "Al Barsha South Fourth",
        "barsha": "Al Barsha South Fourth",

        # Transaction types
        "sale": "Sales",
        "sales": "Sales",
        "buy": "Sales",
        "purchase": "Sales",
        "mortgage": "Mortgages",
        "gift": "Gifts",

        # Property types
        "villa": "Villa",
        "villas": "Villa",
        "apartment": "Unit",
        "apartments": "Unit",
        "flat": "Unit",
        "flats": "Unit",
        "land": "Land",
        "building": "Building",

        # Status (Bayut)
        "ready": "Ready",
        "completed": "Ready",
        "off-plan": "Off Plan",
        "offplan": "Off Plan",
        "under construction": "Off Plan",
    }

    # Explicit alias type mapping
    ALIAS_TYPES = {
        # Areas
        "jvc": EntityType.NEIGHBORHOOD,
        "jlt": EntityType.NEIGHBORHOOD,
        "jbr": EntityType.NEIGHBORHOOD,
        "difc": EntityType.NEIGHBORHOOD,
        "dip": EntityType.NEIGHBORHOOD,
        "dso": EntityType.NEIGHBORHOOD,
        "marina": EntityType.NEIGHBORHOOD,
        "dubai marina": EntityType.NEIGHBORHOOD,
        "downtown": EntityType.NEIGHBORHOOD,
        "downtown dubai": EntityType.NEIGHBORHOOD,
        "business bay": EntityType.NEIGHBORHOOD,
        "palm": EntityType.NEIGHBORHOOD,
        "palm jumeirah": EntityType.NEIGHBORHOOD,
        "mirdif": EntityType.NEIGHBORHOOD,
        "al barsha": EntityType.NEIGHBORHOOD,
        "barsha": EntityType.NEIGHBORHOOD,

        # Transaction types
        "sale": EntityType.STATE,
        "sales": EntityType.STATE,
        "buy": EntityType.STATE,
        "purchase": EntityType.STATE,
        "mortgage": EntityType.STATE,
        "gift": EntityType.STATE,

        # Property types
        "villa": EntityType.PROPERTY_TYPE,
        "villas": EntityType.PROPERTY_TYPE,
        "apartment": EntityType.PROPERTY_TYPE,
        "apartments": EntityType.PROPERTY_TYPE,
        "flat": EntityType.PROPERTY_TYPE,
        "flats": EntityType.PROPERTY_TYPE,
        "land": EntityType.PROPERTY_TYPE,
        "building": EntityType.PROPERTY_TYPE,

        # Status
        "ready": EntityType.STATUS,
        "completed": EntityType.STATUS,
        "off-plan": EntityType.STATUS,
        "offplan": EntityType.STATUS,
        "under construction": EntityType.STATUS,
    }

    def __init__(self, db_connection=None):
        self.db = db_connection
        self.vocabularies: Dict[EntityType, List[str]] = {}
        self._load_vocabularies()

    def _load_vocabularies(self):
        """Load controlled vocabularies from database or defaults."""
        if self.db:
            self._load_from_db()
        else:
            self._load_defaults()

    def _load_from_db(self):
        """Load vocabularies from database for Dubai Proptech data."""
        cursor = self.db.cursor()

        # Areas from lkp_areas
        try:
            cursor.execute("SELECT DISTINCT name_en FROM lkp_areas WHERE name_en IS NOT NULL AND name_en != ''")
            self.vocabularies[EntityType.CITY] = [row[0] for row in cursor.fetchall()]
        except:
            self.vocabularies[EntityType.CITY] = []

        # Transaction types
        try:
            cursor.execute("SELECT DISTINCT trans_group_en FROM transactions WHERE trans_group_en IS NOT NULL")
            self.vocabularies[EntityType.STATE] = [row[0] for row in cursor.fetchall()]
        except:
            self.vocabularies[EntityType.STATE] = []

        # Area names from transactions
        try:
            cursor.execute("SELECT DISTINCT area_name_en FROM transactions WHERE area_name_en IS NOT NULL AND area_name_en != ''")
            self.vocabularies[EntityType.NEIGHBORHOOD] = [row[0] for row in cursor.fetchall()]
        except:
            self.vocabularies[EntityType.NEIGHBORHOOD] = []

        # Property types
        try:
            cursor.execute("SELECT DISTINCT property_type_en FROM transactions WHERE property_type_en IS NOT NULL")
            self.vocabularies[EntityType.PROPERTY_TYPE] = [row[0] for row in cursor.fetchall()]
        except:
            self.vocabularies[EntityType.PROPERTY_TYPE] = []

        # Completion status from Bayut
        try:
            cursor.execute("SELECT DISTINCT completion_status FROM bayut_transactions WHERE completion_status IS NOT NULL")
            self.vocabularies[EntityType.STATUS] = [row[0] for row in cursor.fetchall()]
        except:
            self.vocabularies[EntityType.STATUS] = []

    def _load_defaults(self):
        """Load default vocabularies for Dubai when no database available."""
        self.vocabularies = {
            EntityType.CITY: [
                "Dubai Marina", "Business Bay", "Downtown Dubai", "Palm Jumeirah",
                "Jumeirah Village Circle", "Jumeirah Lake Towers", "DIFC",
                "Arabian Ranches", "Dubai Hills Estate", "Meydan City",
            ],
            EntityType.STATE: ["Sales", "Mortgages", "Gifts"],
            EntityType.PROPERTY_TYPE: ["Villa", "Unit", "Land", "Building"],
            EntityType.STATUS: ["Ready", "Off Plan"],
            EntityType.NEIGHBORHOOD: [
                "Business Bay", "Dubai Marina", "Downtown Dubai", "JVC",
                "Palm Jumeirah", "Al Barsha", "Jumeirah", "Deira",
            ],
        }

    def extract_mentions(self, query: str) -> List[EntityMatch]:
        """
        Extract @mentions from query.

        Supports:
        - @word (single word entity)
        - @"multi word phrase" (quoted multi-word entity)
        - @type:value (typed mention, e.g., @city:Charlotte)
        """
        mentions = []

        # Pattern for typed mentions: @type:value or @type:"value"
        typed_pattern = r'@(\w+):(?:"([^"]+)"|(\w+))'
        for match in re.finditer(typed_pattern, query):
            entity_type_str = match.group(1).lower()
            value = match.group(2) or match.group(3)

            entity_type = self._parse_entity_type(entity_type_str)
            mentions.append(EntityMatch(
                text=value,
                start=match.start(),
                end=match.end(),
                entity=self._create_explicit_entity(value, entity_type),
            ))

        # Pattern for simple mentions: @"value" or @value
        simple_pattern = r'@(?:"([^"]+)"|(\w+))(?!:)'
        for match in re.finditer(simple_pattern, query):
            # Skip if this was part of a typed mention
            if any(m.start <= match.start() < m.end for m in mentions):
                continue

            value = match.group(1) or match.group(2)
            entity = self.resolve_entity(value)
            if entity:
                entity.is_explicit_mention = True
            mentions.append(EntityMatch(
                text=value,
                start=match.start(),
                end=match.end(),
                entity=entity,
            ))

        return mentions

    def _parse_entity_type(self, type_str: str) -> EntityType:
        """Parse entity type from string."""
        mapping = {
            "city": EntityType.CITY,
            "state": EntityType.STATE,
            "neighborhood": EntityType.NEIGHBORHOOD,
            "type": EntityType.PROPERTY_TYPE,
            "property_type": EntityType.PROPERTY_TYPE,
            "status": EntityType.STATUS,
        }
        return mapping.get(type_str.lower(), EntityType.UNKNOWN)

    def _create_explicit_entity(self, value: str, entity_type: EntityType) -> Optional[Entity]:
        """Create entity from explicit typed mention."""
        if entity_type == EntityType.UNKNOWN:
            return self.resolve_entity(value)

        # Resolve within specific vocabulary
        vocab = self.vocabularies.get(entity_type, [])
        match = self._fuzzy_match(value, vocab)

        if match:
            table, column = self.TYPE_MAPPING.get(entity_type, ("transactions", "unknown"))
            return Entity(
                raw_text=value,
                entity_type=entity_type,
                canonical_value=match[0],
                confidence=match[1],
                table=table,
                column=column,
                is_explicit_mention=True,
            )

        return None

    def resolve_entity(self, text: str, context: str = "") -> Optional[Entity]:
        """
        Resolve text to a known entity using fuzzy matching.

        Tries all entity types and returns the best match.
        """
        # First check aliases
        text_lower = text.lower()
        if re.search(r"\d", text_lower):
            return None
        if text_lower in self.ALIASES:
            canonical = self.ALIASES[text_lower]
            entity_type = self.ALIAS_TYPES.get(text_lower) or self._infer_type_from_value(canonical)
            table, column = self.TYPE_MAPPING.get(entity_type, ("transactions", "unknown"))
            return Entity(
                raw_text=text,
                entity_type=entity_type,
                canonical_value=canonical,
                confidence=1.0,
                table=table,
                column=column,
            )

        # Try each vocabulary
        best_match = None
        best_score = 0
        best_type = EntityType.UNKNOWN

        for entity_type, vocab in self.vocabularies.items():
            if not vocab:
                continue

            match = self._fuzzy_match(text, vocab)
            if match and match[1] > best_score:
                best_score = match[1]
                best_match = match[0]
                best_type = entity_type

        if best_match and best_score >= 0.6:  # 60% threshold
            table, column = self.TYPE_MAPPING.get(best_type, ("transactions", "unknown"))
            return Entity(
                raw_text=text,
                entity_type=best_type,
                canonical_value=best_match,
                confidence=best_score,
                table=table,
                column=column,
            )

        return None

    def resolve_all_entities(self, query: str) -> List[Entity]:
        """
        Extract and resolve all entities from query.

        Process:
        1. Extract explicit @mentions
        2. Find implicit entity references
        3. Return deduplicated list
        """
        entities = []
        seen_values = set()

        # Handle explicit @mentions first
        mentions = self.extract_mentions(query)
        for mention in mentions:
            if mention.entity and mention.entity.canonical_value not in seen_values:
                entities.append(mention.entity)
                seen_values.add(mention.entity.canonical_value)

        # Remove @mentions from query for implicit resolution
        clean_query = self._remove_mentions(query)

        # Find implicit entities
        implicit = self._find_implicit_entities(clean_query)
        for entity in implicit:
            if entity.canonical_value not in seen_values:
                entities.append(entity)
                seen_values.add(entity.canonical_value)

        return entities

    def _remove_mentions(self, query: str) -> str:
        """Remove @mentions from query."""
        # Remove typed mentions
        query = re.sub(r'@\w+:(?:"[^"]+"|[\w]+)', '', query)
        # Remove simple mentions
        query = re.sub(r'@(?:"[^"]+"|[\w]+)', '', query)
        return query.strip()

    def _find_implicit_entities(self, query: str) -> List[Entity]:
        """Find entities mentioned without @ syntax."""
        entities = []
        words = query.split()

        # Try single words
        for word in words:
            # Skip common words
            if word.lower() in {'a', 'an', 'the', 'in', 'on', 'at', 'for', 'with', 'and', 'or'}:
                continue
            if re.search(r"\d", word):
                continue

            entity = self.resolve_entity(word)
            if entity and entity.confidence >= 0.8:
                entities.append(entity)

        # Try bigrams for multi-word entities
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if re.search(r"\d", bigram):
                continue
            entity = self.resolve_entity(bigram)
            if entity and entity.confidence >= 0.8:
                entities.append(entity)

        return entities

    def _fuzzy_match(self, text: str, vocabulary: List[str]) -> Optional[Tuple[str, float]]:
        """
        Find best fuzzy match in vocabulary.

        Returns (matched_value, confidence_score) or None.
        """
        if not vocabulary:
            return None

        text_lower = text.lower()
        best_match = None
        best_score = 0

        for item in vocabulary:
            item_lower = item.lower()

            # Exact match
            if text_lower == item_lower:
                return (item, 1.0)

            # Prefix match
            if item_lower.startswith(text_lower):
                score = len(text_lower) / len(item_lower) * 0.9
                if score > best_score:
                    best_score = score
                    best_match = item

            # Contains match
            elif text_lower in item_lower:
                score = len(text_lower) / len(item_lower) * 0.7
                if score > best_score:
                    best_score = score
                    best_match = item

            # Sequence matching
            else:
                score = SequenceMatcher(None, text_lower, item_lower).ratio()
                if score > best_score:
                    best_score = score
                    best_match = item

        if best_match and best_score >= 0.5:
            return (best_match, best_score)

        return None

    def _infer_type_from_value(self, value: str) -> EntityType:
        """Infer entity type from resolved value."""
        for entity_type, vocab in self.vocabularies.items():
            if value in vocab:
                return entity_type
        return EntityType.UNKNOWN

    def suggest_completions(
        self,
        partial: str,
        entity_type: Optional[EntityType] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Autocomplete suggestions for @mention UI.

        Returns ranked list of suggestions.
        """
        suggestions = []
        partial_lower = partial.lower()

        vocabs_to_search = (
            {entity_type: self.vocabularies.get(entity_type, [])}
            if entity_type
            else self.vocabularies
        )

        for etype, vocab in vocabs_to_search.items():
            for item in vocab:
                item_lower = item.lower()
                score = 0

                # Prefix match (highest priority)
                if item_lower.startswith(partial_lower):
                    score = 100 + (len(partial_lower) / len(item_lower) * 50)
                # Contains match
                elif partial_lower in item_lower:
                    score = 50 + (len(partial_lower) / len(item_lower) * 25)
                # Fuzzy match
                else:
                    ratio = SequenceMatcher(None, partial_lower, item_lower).ratio()
                    if ratio > 0.5:
                        score = ratio * 40

                if score > 0:
                    suggestions.append({
                        "value": item,
                        "type": etype.value,
                        "score": score,
                        "display": f"{item} ({etype.value})",
                    })

        # Sort by score descending
        suggestions.sort(key=lambda x: -x["score"])
        return suggestions[:limit]

    def get_vocabulary(self, entity_type: EntityType) -> List[str]:
        """Get vocabulary for an entity type."""
        return self.vocabularies.get(entity_type, [])

    def refresh_vocabularies(self):
        """Refresh vocabularies from database."""
        if self.db:
            self._load_from_db()


class QueryPreprocessor:
    """
    Preprocess queries to expand entities and normalize text.
    """

    def __init__(self, resolver: EntityResolver):
        self.resolver = resolver

    def preprocess(self, query: str) -> Tuple[str, List[Entity]]:
        """
        Preprocess query: resolve entities and clean up text.

        Returns (processed_query, resolved_entities).
        """
        # Resolve all entities
        entities = self.resolver.resolve_all_entities(query)

        # Create processed query with entities marked
        processed = query

        # Replace @mentions with canonical values
        for entity in entities:
            if entity.is_explicit_mention:
                # Replace the @mention syntax with canonical value
                patterns = [
                    rf'@{entity.raw_text}\b',
                    rf'@"{re.escape(entity.raw_text)}"',
                    rf'@\w+:{re.escape(entity.raw_text)}\b',
                    rf'@\w+:"{re.escape(entity.raw_text)}"',
                ]
                for pattern in patterns:
                    processed = re.sub(pattern, entity.canonical_value, processed, flags=re.IGNORECASE)

        return processed, entities
