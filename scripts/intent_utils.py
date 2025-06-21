# scripts/intent_utils.py
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy
from sentence_transformers import SentenceTransformer, util
import os
import json

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_trf")
except Exception:
    nlp = spacy.load("en_core_web_sm")

def normalize_entity(e):
    text = re.sub(r'\s+', ' ', re.sub(r'\.', '', e.lower())).strip()
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

intent_keywords = {
    "claim_process": ["claim", "process", "file", "submit", "insurance"],
    "case_status": ["status", "case", "update", "progress", "judgement", "order", "appeal"],
    "document_request": ["document", "request", "copies", "forms"],
    # "technical_support": ["error", "issue", "problem", "technical"],
    "general_info": ["information", "contact", "hours", "location"],
    "resume_info": [
        "skills", "resume", "cv", "proficiencies", "abilities", "expertise", "competencies", "qualifications",
        "experience", "work", "education", "background", "certifications", "projects", "programming", "languages",
        "achievements", "awards", "contact", "career", "summary", "tools", "technologies", "roles", "responsibilities",
        "soft skills", "applicant", "candidate", "developer", "engineer", "profile", "professional", "employment", "history",
        "management", "software", "admin", "implementation", "tracking", "project", "system", "solution", "platform", "application",
        "team", "lead", "player", "restocking", "inventory", "tournament", "manual", "implemented"
    ]
}
project_root = os.environ.get('PROJECT_ROOT', os.getcwd())
intent_examples_path = os.path.join(project_root, 'data', 'intent_categories', 'intent_examples.json')
with open(intent_examples_path, 'r', encoding='utf-8') as f:
    intent_examples = json.load(f)
intent_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def get_intent(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc]
    detected_intent = None
    max_matches = 0
    for intent, keywords in intent_keywords.items():
        matches = sum(kw in tokens for kw in keywords)
        if matches > max_matches:
            max_matches = matches
            detected_intent = intent
    intent_confidence = max_matches / max(1, len(intent_keywords.get(detected_intent, []))) if detected_intent else 0.0
    if not detected_intent or max_matches == 0:
        query_emb = intent_model.encode(text, convert_to_tensor=True)
        best_intent, best_score = None, 0
        for intent, examples in intent_examples.items():
            example_embs = intent_model.encode(examples, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(query_emb, example_embs)
            max_score = scores.max().item()
            if max_score > best_score:
                best_score = max_score
                best_intent = intent
        if best_score > 0.35:
            detected_intent = best_intent
            intent_confidence = best_score
    if not detected_intent:
        detected_intent = "general_info"
        intent_confidence = 0.0
    return detected_intent, intent_confidence, None

