import json
from fastapi import logger
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from keybert import KeyBERT
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import numpy as np
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy
import yaml
from scripts.entity_utils import normalize_entity, get_spacy_nlp
from scripts.search_pipeline import get_openai_embedding
from scripts.intent_utils import get_intent
import tiktoken
from sentence_transformers import SentenceTransformer , util
import glob
from scripts.entity_utils import _spacy_nlp as nlp

keyword_model = KeyBERT()
ner_pipeline = pipeline(
    "ner",
    model=AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english"),
    tokenizer=AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english"),
    aggregation_strategy="simple",
    device=-1  # Force CPU to avoid meta tensor error
)
load_dotenv("config/.env")
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

project_root = os.environ.get('PROJECT_ROOT', os.getcwd())
intent_examples_path = os.path.join(project_root, 'data', 'intent_categories', 'intent_examples.json')
with open(intent_examples_path, 'r', encoding='utf-8') as f:
    intent_examples = json.load(f)

intent_labels = list(intent_examples.keys())
intent_texts = [item for sublist in intent_examples.values() for item in sublist]


CONFIG_PATH = f"{project_root}/config.yaml"



with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
intent_threshold = float(os.getenv("INTENT_THRESHOLD", config.get("intent_threshold", 0.2)))
# custom_stopwords = set(config.get("custom_stopwords", []))

case_number_patterns = config.get("case_number_patterns", [])

# Example: intent_keywords can be loaded from config.yaml or defined here
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

# Load the model ONCE at the module level
intent_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def get_intent(text, _doc_emb=None):
    # Use the global model
    global intent_model
    nlp = get_spacy_nlp()
    # Lemmatize and lowercase query
    if nlp:
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc]
    else:
        tokens = text.lower().split()
    detected_intent = None
    max_matches = 0
    # 1. Rule-based keyword/lemmatized match
    for intent, keywords in intent_keywords.items():
        matches = sum(kw in tokens for kw in keywords)
        if matches > max_matches:
            max_matches = matches
            detected_intent = intent
    # 2. Fallback: embedding similarity if no keyword match
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

def extract_names_regex(text):
    # Regex for names: capitalized words, possibly with middle initials, dots, underscores, etc.
    # Also handle all-caps names and multi-part names
    name_patterns = [
        r'([A-Z][a-z]+(?: [A-Z][a-z]+)+)',  # John Smith, Jimson Ratnam
        r'([A-Z][a-z]+_[A-Z][a-z]+)',      # John_Smith
        r'([A-Z][a-z]+\.[A-Z][a-z]+)',    # John.Smith
        r'([A-Z][a-z]+[A-Z][a-z]+)',       # JohnSmith (camel case)
        r'([A-Z]{2,}(?: [A-Z]{2,})+)',     # ALL CAPS NAMES: JIMSON RATNAM
    ]
    found = set()
    for pat in name_patterns:
        for match in re.findall(pat, text):
            found.add(match.strip().lower())
    # Also split all found names into parts (space, underscore, dot)
    parts = set()
    for name in found:
        for part in re.split(r'[ _\.]', name):
            if part and len(part) > 1:
                parts.add(part.lower())
        # Add all substrings (prefixes of length >= 3) for each part
        for part in re.split(r'[ _\.]', name):
            part = part.lower()
            for i in range(3, len(part)):
                substr = part[:i]
                if len(substr) >= 3:
                    parts.add(substr)
    return list(found | parts)

def extract_keywords(text):
    keywords_with_scores = keyword_model.extract_keywords(text, top_n=10)
    return [kw for kw, _ in keywords_with_scores]

def extract_entities(text):
    stopwords = set(ENGLISH_STOP_WORDS)
    entities = []
    batch_size = 2000
    for i in range(0, len(text), batch_size):
        batch = text[i:i+batch_size]
        for ent in ner_pipeline(batch):
            if ent['score'] > 0.8:
                entities.append({
                    'text': ent['word'].strip().lower(),
                    'type': ent['entity_group'],
                    'score': float(ent['score'])
                })
    extra_entities = set()
    for ent in entities:
        ent_text = ent['text']
        ent_type = ent.get('type', '')
        if ent_type not in ('PER', 'ORG'):
            continue
        ent_text = ent_text.lower().strip()
        if not ent_text or ent_text in stopwords:
            continue
        extra_entities.add(ent_text)
        parts = re.split(r'[ _\.]', ent_text)
        if len(parts) > 1:
            extra_entities.add(parts[0])
            extra_entities.add(parts[-1])
            initials = ''.join([p[0] for p in parts if p])
            if len(initials) > 1:
                extra_entities.add(initials)
    regex_names = extract_names_regex(text)
    for name in regex_names:
        name = name.lower().strip()
        if name and name not in stopwords:
            extra_entities.add(name)
    for pat in case_number_patterns:
        for match in re.findall(pat, text, re.IGNORECASE):
            cleaned = normalize_entity(match)
            if cleaned and cleaned not in stopwords:
                extra_entities.add(cleaned)
    all_entities = set([normalize_entity(e['text']) for e in entities if len(e['text']) >= 4 and e['text'] not in stopwords]) | set([normalize_entity(e) for e in extra_entities])
    all_entities = set([e for e in all_entities if len(e) >= 3 and e not in stopwords])
    return sorted(list(all_entities)), [e['type'] for e in entities], entities

def summarize_text(text):
    # Truncate text to fit within LLM context window (e.g., 8000 tokens for GPT-4o)
    max_tokens = 8000
    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    summary_prompt = (
        "Summarize the following text in 1-2 sentences. "
        "Be extremely concise and do not include bullets or extra details. "
        "Text: {text}"
    )
    summary = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": summary_prompt.format(text=text)}],
        max_tokens=256,
        temperature=0.2
    )
    return summary.choices[0].message.content.strip()

def build_metadata(keywords, detected_intent, intent_confidence, entities, entity_types, entity_details, summary, doc_emb, text, document_name=None):
    metadata = {
        "keywords": keywords,
        "intent": detected_intent,
        "intent_confidence": intent_confidence,
        "entities": entities,
        "entity_types": list(set(entity_types)),
        "entity_details": entity_details,
        "summary": summary,
        "embedding": doc_emb,
        "text": text
    }
    if document_name:
        metadata["document_name"] = document_name
    return metadata


def extract_metadata(text, document_name=None):
    doc_emb = get_openai_embedding(text)
    detected_intent, intent_confidence, _ = get_intent(text, doc_emb)
    keywords = extract_keywords(text)
    entities, entity_types, entity_details = extract_entities(text)
    summary = summarize_text(text)
    return build_metadata(keywords, detected_intent, intent_confidence, entities, entity_types, entity_details, summary, doc_emb, text, document_name)


def extract_document_level_entities(pdf_name, chunk_dir="data/chunks"):
    """
    Extract entities for a document by aggregating over all its chunk files.
    """
    chunk_pattern = f"{chunk_dir}/{pdf_name.replace('.pdf', '')}_chunk*.txt"
    chunk_files = sorted(glob.glob(chunk_pattern))
    all_entities = set()
    for chunk_file in chunk_files:
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunk_text = f.read()
        # Use your existing entity extraction logic for each chunk
        entities = extract_entities(chunk_text)
        all_entities.update(entities)
    return sorted(all_entities)