import json
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from keybert import KeyBERT
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import numpy as np
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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

with open("data/intent_categories/intent_examples.json", "r", encoding="utf-8") as f:
    intent_examples = json.load(f)

intent_labels = list(intent_examples.keys())
intent_texts = [item for sublist in intent_examples.values() for item in sublist]

def get_openai_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

intent_embs = [get_openai_embedding(t) for t in intent_texts]

# Map each embedding to its label for debug logging
intent_label_map = []
for label, examples in intent_examples.items():
    for _ in examples:
        intent_label_map.append(label)

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

def normalize_entity(e):
    # Lowercase, remove dots, extra spaces, collapse multiple spaces
    return re.sub(r'\s+', ' ', re.sub(r'\.', '', e.lower())).strip()

def extract_metadata(text, document_name=None):
    doc_emb = get_openai_embedding(text)
    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    intent_scores = [cosine_sim(doc_emb, emb) for emb in intent_embs]
    # Print all intent scores for debug
    debug_scores = {}
    for idx, score in enumerate(intent_scores):
        label = intent_label_map[idx]
        debug_scores.setdefault(label, []).append(score)
    print("[Intent Detection] All intent scores:")
    for label, scores in debug_scores.items():
        print(f"  {label}: max={max(scores):.3f}, avg={np.mean(scores):.3f}, all={[round(s,3) for s in scores]}")
    # Use the max score per label for intent assignment
    label_max_scores = {label: max(scores) for label, scores in debug_scores.items()}
    best_label = max(label_max_scores, key=label_max_scores.get)
    intent_confidence = float(label_max_scores[best_label])
    # Lowered threshold for more permissive intent detection
    detected_intent = best_label if intent_confidence > 0.2 else "general_info"
    print(f"[Intent Detection] intent_confidence={intent_confidence:.3f}, detected_intent={detected_intent}")

    # Extract keywords
    keywords_with_scores = keyword_model.extract_keywords(text, top_n=10)
    keywords = [kw for kw, _ in keywords_with_scores]

    # Get named entities with details (NER)
    entities = []
    for ent in ner_pipeline(text[:5000]):  # Limit text length for NER
        if ent['score'] > 0.8:
            entities.append({
                'text': ent['word'].strip().lower(),
                'type': ent['entity_group'],
                'score': float(ent['score'])
            })
    # --- Hybrid entity extraction: NER + regex, using standard stopwords ---
    custom_stopwords = set([
        'pdf', 'doc', 'file', 'info', 'data', 'case', 'user', 'test', 'testcase', 'docx', 'page', 'form', 'type', 'role', 'team', 'work', 'year', 'date', 'time', 'list', 'desc', 'desc.', 'desc:', 'desc;'
    ])
    stopwords = set(ENGLISH_STOP_WORDS) | custom_stopwords
    extra_entities = set()
    for ent in entities:
        ent_text = ent['text']
        ent_type = ent.get('type', '')
        if ent_type not in ('PER', 'ORG'):
            continue
        ent_text = ent_text.lower().strip()
        if not ent_text or ent_text in stopwords:
            continue
        # Add full name
        extra_entities.add(ent_text)
        # Add first and last name if multi-part
        parts = re.split(r'[ _\.]', ent_text)
        if len(parts) > 1:
            extra_entities.add(parts[0])  # first name
            extra_entities.add(parts[-1]) # last name
        # Add initials (if multi-part)
        if len(parts) > 1:
            initials = ''.join([p[0] for p in parts if p])
            if len(initials) > 1:
                extra_entities.add(initials)
    # Add regex-based names (full matches only, no substrings)
    regex_names = extract_names_regex(text)
    for name in regex_names:
        name = name.lower().strip()
        if name and name not in stopwords:
            extra_entities.add(name)
    # --- Add legal case number extraction (more flexible) ---
    case_number_patterns = [
        r'\b(?:[A-Z]{1,4}\.?\s*)?S\.?\s*No\.?\s*\d+\s*(?:of|/)?\s*\d{4}\b',  # O.S.No.15 of 2007, S No 15/2007
        r'\b(?:[A-Z]{1,4}\.?\s*)?No\.?\s*\d+\s*(?:of|/)?\s*\d{4}\b',  # No.15 of 2007, No 15/2007
        r'\b\d+\s*(?:of|/)?\s*\d{4}\b',  # 15 of 2007, 123/2020
        r'\b[A-Z]{1,4}\.?\s*No\.?\s*\d+\b',  # C.R.P.No.1234
        r'\b[A-Z]{1,4}\.?\s*\d+\b',  # CRP.1234
    ]
    for pat in case_number_patterns:
        for match in re.findall(pat, text, re.IGNORECASE):
            cleaned = normalize_entity(match)
            if cleaned and cleaned not in stopwords:
                extra_entities.add(cleaned)
    # Only keep normalized, deduplicated entities
    all_entities = set([normalize_entity(e['text']) for e in entities if len(e['text']) >= 4 and e['text'] not in stopwords]) | set([normalize_entity(e) for e in extra_entities])
    # Remove overly generic substrings (e.g., length < 3 or common stopwords)
    stopwords = set(['the', 'and', 'for', 'with', 'from', 'that', 'this', 'are', 'was', 'but', 'not', 'all', 'any', 'can', 'has', 'have', 'had', 'you', 'her', 'his', 'she', 'him', 'who', 'how', 'why', 'use', 'our', 'out', 'get', 'got', 'let', 'may', 'one', 'two', 'job', 'dev', 'rat', 'man', 'son', 'jan', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    all_entities = set([e for e in all_entities if len(e) >= 3 and e not in stopwords])
    # Generate summary
    summary_prompt = (
    "Summarize the following text in 1-2 sentences. "
    "Be extremely concise and do not include bullets or extra details. "
    "Text: {text}"
    )
    summary = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": summary_prompt.format(text=text[:2000])}],
        temperature=0.3,
        max_tokens=256
    ).choices[0].message.content.strip()
    metadata = {
        "keywords": keywords,
        "intent": detected_intent,
        "intent_confidence": intent_confidence,
        "entities": sorted(list(all_entities)),
        "entity_types": list(set(e['type'] for e in entities)),
        "entity_details": entities,
        "summary": summary,
        "embedding": doc_emb,
        "text": text
    }
    if document_name:
        metadata["document_name"] = document_name
    return metadata