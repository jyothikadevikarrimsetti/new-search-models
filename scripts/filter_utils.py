"""Utilities for generating and managing Pinecone metadata filters."""

from typing import Dict, Any, Optional
import spacy
import json
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import re
import subprocess

def initialize_spacy():
    """Initialize spaCy with error handling and automatic model download."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy English model...")
        try:
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            return spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"Error downloading/loading spaCy model: {e}")
            print("Please try manually running: python -m spacy download en_core_web_sm")
            raise

# Initialize spaCy model with better error handling
try:
    nlp = initialize_spacy()
except Exception as e:
    print(f"Failed to initialize spaCy: {e}")
    raise

# Define default intent keywords
INTENT_KEYWORDS = {
    "claim_process": ["claim", "process", "file", "submit", "insurance"],
    "case_status": ["status", "case", "update", "progress"],
    "document_request": ["document", "request", "copies", "forms"],
    "technical_support": ["error", "issue", "problem", "technical"],
    "general_info": ["information", "contact", "hours", "location"],
    "resume_skills": ["skills", "resume", "cv", "proficiencies", "abilities", "expertise", "competencies", "qualifications"]
}

# Load the sentence transformer model for semantic similarity
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Load intent examples from JSON
INTENT_EXAMPLES_PATH = Path(__file__).parent.parent / "data/intent_categories/intent_examples.json"

def load_intent_examples():
    """Load and cache intent examples."""
    try:
        with open(INTENT_EXAMPLES_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback to basic intent keywords if examples file not found
        return {
            "claim_process": ["claim", "process", "file", "submit", "insurance"],
            "case_status": ["status", "case", "update", "progress"],
            "document_request": ["document", "request", "copies", "forms"],
            "technical_support": ["error", "issue", "problem", "technical"],
            "general_info": ["information", "contact", "hours", "location"],
            "resume_skills": ["skills", "resume", "cv", "proficiencies", "abilities", "expertise", "competencies", "qualifications"]
        }

def extract_query_metadata(query: str) -> Dict[str, Any]:
    """Extract metadata fields from a query for filtering."""
    doc = nlp(query)
    # Extract entities (normalize: strip, lowercase)
    entities = [ent.text.strip().lower() for ent in doc.ents]
    # Load intent examples
    intent_examples = load_intent_examples()
    # Encode query
    query_embedding = model.encode(query.lower(), convert_to_tensor=True)
    # Calculate similarity with each intent's examples
    intent_scores = {}
    for intent, examples in intent_examples.items():
        example_embeddings = model.encode([ex.lower() for ex in examples], convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, example_embeddings)[0]
        intent_scores[intent] = float(max(similarities))
    max_score = max(intent_scores.values())
    # Lowered threshold for more permissive intent detection
    if max_score > 0.35:
        detected_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
    else:
        detected_intent = None
    # Extract potential keywords
    keywords = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    # Keyword-based fallback: if no intent detected, use keyword match
    if not detected_intent and keywords:
        for intent, kw_list in INTENT_KEYWORDS.items():
            if any(kw in keywords for kw in kw_list):
                detected_intent = intent
                print(f"[FilterUtils] Fallback: intent '{intent}' matched by keyword.")
                break
    print(f"[FilterUtils] Query: {query}")
    print(f"[FilterUtils] Extracted entities: {entities}")
    print(f"[FilterUtils] Detected intent: {detected_intent}")
    print(f"[FilterUtils] Extracted keywords: {keywords}")
    return {
        "entities": entities,
        "intent": detected_intent,
        "keywords": keywords
    }

def generate_filter(query: str) -> Optional[Dict[str, Any]]:
    """Generate a Pinecone metadata filter based on query analysis."""
    metadata = extract_query_metadata(query)
    filter_dict = {}
    # Add intent filter if detected
    if metadata["intent"]:
        filter_dict["intent"] = {"$eq": metadata["intent"]}
    # Add entity filters if found
    norm_entities = [e.strip().lower() for e in metadata["entities"]]
    if norm_entities:
        filter_dict["entities"] = {"$in": norm_entities}
    # Fallback: if no intent/entities, but keywords exist, use keywords as entity filter
    if not filter_dict and metadata["keywords"]:
        norm_keywords = [k.strip().lower() for k in metadata["keywords"]]
        if norm_keywords:
            filter_dict["entities"] = {"$in": norm_keywords}
            print(f"[FilterUtils] Fallback: using keywords as entity filter.")
    print(f"[FilterUtils] Final generated filter: {filter_dict if filter_dict else None}")
    return filter_dict if filter_dict else None

def combine_filters(filters: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine multiple filters with AND logic."""
    if not filters:
        return {}
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}

def is_entity_lookup_query(metadata):
    # Entity lookup: no intent, but entities or keywords present
    return (
        not metadata.get("intent") and (
            (metadata.get("entities") and len(metadata["entities"]) > 0) or
            (metadata.get("keywords") and len(metadata["keywords"]) > 0)
        )
    )
