import json
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from keybert import KeyBERT
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import numpy as np
import re

keyword_model = KeyBERT()
ner_pipeline = pipeline(
    "ner",
    model=AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english"),
    tokenizer=AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english"),
    aggregation_strategy="simple",
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
intent_texts = list(intent_examples.values())

def get_openai_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

intent_embs = [get_openai_embedding(t) for t in intent_texts]

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

def extract_metadata(text, document_name=None):
    doc_emb = get_openai_embedding(text)
    # Compute cosine similarity with all intent embeddings
    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    intent_scores = [cosine_sim(doc_emb, emb) for emb in intent_embs]
    best_idx = int(np.argmax(intent_scores))
    intent_confidence = float(intent_scores[best_idx])
    # Lowered threshold for more permissive intent detection
    detected_intent = intent_labels[best_idx] if intent_confidence > 0.3 else "general_info"
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
    # --- Expanded substring/variant generation for all entities ---
    extra_entities = set()
    for ent in entities:
        ent_text = ent['text']
        # Split on space, underscore, dot, and camel case
        parts = re.split(r'[ _\.]', ent_text)
        camel_parts = re.findall(r'[A-Za-z][^A-Z]*', ent_text)
        all_parts = set(parts + camel_parts)
        for part in all_parts:
            part = part.lower().strip()
            if part and len(part) > 1:
                extra_entities.add(part)
            # Add all substrings (prefixes of length >= 3) for each part
            for i in range(3, len(part)+1):
                substr = part[:i]
                if len(substr) >= 3:
                    extra_entities.add(substr)
        # Add sliding substrings for the full entity (length >= 3)
        for i in range(len(ent_text)):
            for j in range(i+3, len(ent_text)+1):
                substr = ent_text[i:j].lower().strip()
                if substr and len(substr) >= 3:
                    extra_entities.add(substr)
    # Add split name parts and substrings for PERSON entities (legacy logic)
    for ent in entities:
        if ent['type'] == 'PER':
            parts = re.split(r'[ _\.]', ent['text'])
            camel_parts = re.findall(r'[A-Za-z][^A-Z]*', ent['text'])
            for part in parts + camel_parts:
                part = part.lower()
                if part:
                    extra_entities.add(part)
                for i in range(3, len(part)+1):
                    substr = part[:i]
                    if len(substr) >= 3:
                        extra_entities.add(substr)
    # --- Hybrid: Regex-based name extraction ---
    regex_names = extract_names_regex(text)
    for name in regex_names:
        extra_entities.add(name.lower().strip())
    # Normalize and deduplicate all entities
    all_entities = set([e['text'].lower().strip() for e in entities]) | set([e for e in extra_entities if len(e) > 1])
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