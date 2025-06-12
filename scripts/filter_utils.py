"""Utilities for generating and managing Pinecone metadata filters."""

from typing import Dict, Any, Optional
import spacy
import json
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import re

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
    "general_info": ["information", "contact", "hours", "location"]
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
            "general_info": ["information", "contact", "hours", "location"]
        }

def extract_query_metadata(query: str) -> Dict[str, Any]:
    """Extract metadata fields from a query for filtering."""
    doc = nlp(query)
    
    # Extract entities
    entities = [ent.text for ent in doc.ents]
    
    # Load intent examples
    intent_examples = load_intent_examples()
    
    # Encode query
    query_embedding = model.encode(query.lower(), convert_to_tensor=True)
    
    # Calculate similarity with each intent's examples
    intent_scores = {}
    for intent, examples in intent_examples.items():
        # Encode all examples for this intent
        example_embeddings = model.encode([ex.lower() for ex in examples], convert_to_tensor=True)
        # Calculate similarity scores
        similarities = util.cos_sim(query_embedding, example_embeddings)[0]
        # Use max similarity as the score for this intent
        intent_scores[intent] = float(max(similarities))
    
    # Get highest scoring intent if above threshold
    max_score = max(intent_scores.values())
    if max_score > 0.5:  # Confidence threshold
        detected_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
    else:
        detected_intent = None
    
    # Extract potential keywords
    keywords = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    
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
    if metadata["entities"]:
        # Look for entities in the comma-separated entities field
        entity_pattern = "|".join(re.escape(e) for e in metadata["entities"])
        filter_dict["entities"] = {"$contains": entity_pattern}
    
    # No filter if no meaningful criteria found
    return filter_dict if filter_dict else None

def combine_filters(filters: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine multiple filters with AND logic."""
    if not filters:
        return {}
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}
