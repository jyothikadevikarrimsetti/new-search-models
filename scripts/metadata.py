import json
import re
import time
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from keybert import KeyBERT
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
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
intent_embs = embedding_model.encode(intent_texts, convert_to_tensor=True)

def extract_metadata(text):
    # 1. Get document embedding
    doc_emb = embedding_model.encode(text, convert_to_tensor=True)
    
    # 2. Improved intent classification with confidence score
    intent_scores = util.cos_sim(doc_emb, intent_embs)[0]
    best_idx = int(intent_scores.argmax())
    intent_confidence = float(intent_scores[best_idx])
    
    # Only assign intent if confidence is high enough
    detected_intent = intent_labels[best_idx] if intent_confidence > 0.5 else "general_info"
    
    # 3. Enhanced keyword extraction with scores
    keywords_with_scores = keyword_model.extract_keywords(
        text, 
        top_n=10,
        keyphrase_ngram_range=(1, 2),  # Allow 2-word phrases
        stop_words='english'
    )
    keywords = [kw for kw, score in keywords_with_scores if score > 0.2]
    
    # 4. Improved named entity recognition with type information
    entities = []
    for ent in ner_pipeline(text[:5000]):  # Limit text length for NER
        if ent['score'] > 0.8:  # Only keep high confidence entities
            entities.append({
                'text': ent['word'],
                'type': ent['entity_group'],
                'score': float(ent['score'])
            })
    
    # 5. Generate a structured summary using Azure OpenAI
    summary_prompt = """Analyze and summarize this text. Extract:
    1. Main topic/subject (1 sentence)
    2. Key points (2-3 bullets)
    3. Any dates, numbers, or identifiers mentioned

    Text: {text}"""
    
    summary = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": summary_prompt.format(text=text[:2000])}],
        temperature=0.3,
        max_tokens=256
    ).choices[0].message.content.strip()    # 6. Structure the metadata for flexible filtering
    return {
        "keywords": keywords,
        "intent": detected_intent,
        "intent_confidence": intent_confidence,
        "entities": [e['text'] for e in entities],
        "entity_types": list(set(e['type'] for e in entities)),
        "entity_details": entities,
        "summary": summary,
        "embedding": doc_emb.tolist(),
        "text": text,
        # Add text statistics for potential filtering
        "text_stats": {
            "length": len(text),
            "sentence_count": len(text.split('.')),
            "has_numbers": bool(re.search(r'\d', text)),
            "timestamp": int(time.time())  # When the document was processed
        }
    }