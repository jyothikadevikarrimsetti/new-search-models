"""Utilities for generating and managing Pinecone metadata filters."""

from typing import Dict, Any
from scripts.entity_utils import normalize_entity, get_spacy_nlp
from scripts.intent_utils import get_intent
import re

# Only keep what is needed for filter generation and metadata extraction

def extract_query_metadata(query: str) -> Dict[str, Any]:
    """Extract metadata fields from a query for filtering."""
    nlp = get_spacy_nlp()
    doc = nlp(query)
    entities = [normalize_entity(ent.text) for ent in doc.ents]
    keywords = [normalize_entity(token.text) for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    detected_intent, _, _ = get_intent(query)
    return {
        "entities": entities,
        "intent": detected_intent,
        "keywords": keywords
    }

def generate_filter(query: str) -> dict:
    """
    Hybrid filter generation with robust normalization and relaxed matching:
    - Always use full, normalized entity strings for matching.
    - For party/case names, match if any entity OR keyword matches (not strict AND).
    - Ensure both filter and chunk metadata use the same normalization logic.
    - Only require intent match if the query is not an entity/party/case lookup (scalable solution).
    """
    nlp = get_spacy_nlp()
    doc = nlp(query) if nlp else None
    entities = [normalize_entity(ent.text) for ent in doc.ents] if doc else []
    keywords = [normalize_entity(token.text) for token in doc if token.pos_ in ["NOUN", "PROPN"]] if doc else []
    party_names = []
    party_pattern = re.compile(r'([A-Z]\.[A-Z]\.[A-Za-z]+|[A-Z]\.[A-Za-z]+|[A-Z][a-z]+)', re.IGNORECASE)
    for match in party_pattern.findall(query):
        party_names.append(normalize_entity(match))
    all_names = set(entities + keywords + party_names)
    all_names = list(set(all_names))
    filter_dict = {}
    if all_names:
        filter_dict["$or"] = [
            {"entities": {"$in": all_names}},
            {"keywords": {"$in": all_names}}
        ]
    detected_intent, _, _ = get_intent(query)
    is_entity_query = (not detected_intent or detected_intent == "general_info") and (len(all_names) > 0)
    if detected_intent and detected_intent != "general_info" and not is_entity_query:
        filter_dict["intent"] = {"$eq": detected_intent}
    if not filter_dict:
        filter_dict["intent"] = {"$eq": "general_info"}
    return filter_dict

def combine_filters(filters: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine multiple filters with AND logic."""
    if not filters:
        return {}
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}

def is_entity_lookup_query(metadata):
    return (
        not metadata.get("intent") and (
            (metadata.get("entities") and len(metadata["entities"]) > 0) or
            (metadata.get("keywords") and len(metadata["keywords"]) > 0)
        )
    )
