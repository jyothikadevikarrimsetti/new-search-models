"""Utilities for generating and managing Pinecone metadata filters."""

from typing import Dict, Any
from scripts.entity_utils import normalize_entity, get_spacy_nlp
from scripts.intent_utils import get_intent
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
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
    nlp = get_spacy_nlp()
    doc = nlp(query) if nlp else None
    # Only keep meaningful tokens (nouns, proper nouns), not stopwords or auxiliaries
    keywords = [
        normalize_entity(token.text)
        for token in doc
        if token.pos_ in ["NOUN", "PROPN"]
        and not token.is_stop
        and token.lemma_.lower() not in ENGLISH_STOP_WORDS
        and len(token.text) > 2  # avoid single letters
    ] if doc else []

    # Entities (from NER)
    entities = [
        normalize_entity(ent.text)
        for ent in doc.ents
        if not nlp.vocab[ent.text].is_stop
        and ent.text.lower() not in ENGLISH_STOP_WORDS
    ] if doc else []

    # Party/case names (regex, as before)
    party_names = []
    party_pattern = re.compile(r'([A-Z]\.[A-Z]\.[A-Za-z]+|[A-Z]\.[A-Za-z]+|[A-Z][a-z]+)', re.IGNORECASE)
    for match in party_pattern.findall(query):
        norm = normalize_entity(match)
        if norm not in ENGLISH_STOP_WORDS and len(norm) > 2:
            party_names.append(norm)

    all_names = set(entities + keywords + party_names)
    all_names = [name for name in all_names if name not in ENGLISH_STOP_WORDS and len(name) > 2]

    filter_dict = {}
    detected_intent, _, _ = get_intent(query)
    # If we have any names, build $or for entities/keywords, and also include intent in $or if it's not general_info
    if all_names:
        or_clauses = [
            {"entities": {"$in": all_names}},
            {"keywords": {"$in": all_names}}
        ]
        if detected_intent and detected_intent != "general_info":
            or_clauses.append({"intent": detected_intent})
        filter_dict["$or"] = or_clauses
    elif detected_intent and detected_intent != "general_info":
        filter_dict["intent"] = {"$eq": detected_intent}
    else:
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
