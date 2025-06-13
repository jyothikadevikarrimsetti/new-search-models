import json
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from keybert import KeyBERT
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import numpy as np

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

def extract_metadata(text):
    doc_emb = get_openai_embedding(text)
    # Compute cosine similarity with all intent embeddings
    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    intent_scores = [cosine_sim(doc_emb, emb) for emb in intent_embs]
    best_idx = int(np.argmax(intent_scores))
    intent_confidence = float(intent_scores[best_idx])
    detected_intent = intent_labels[best_idx] if intent_confidence > 0.5 else "general_info"

    # Extract keywords
    keywords_with_scores = keyword_model.extract_keywords(text, top_n=10)
    keywords = [kw for kw, _ in keywords_with_scores]

    # Get named entities with details
    entities = []
    for ent in ner_pipeline(text[:5000]):  # Limit text length for NER
        if ent['score'] > 0.8:  # Only keep high confidence entities
            entities.append({
                'text': ent['word'],
                'type': ent['entity_group'],
                'score': float(ent['score'])
            })

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
    return {
        "keywords": keywords,
        "intent": detected_intent,
        "intent_confidence": intent_confidence,
        "entities": [e['text'] for e in entities],
        "entity_types": list(set(e['type'] for e in entities)),
        "entity_details": entities,
        "summary": summary,
        "embedding": doc_emb,
        "text": text  # <-- Store full text for BM25/hybrid search
    }