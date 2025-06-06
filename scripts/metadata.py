import json
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
    doc_emb = embedding_model.encode(text, convert_to_tensor=True)
    intent_scores = util.cos_sim(doc_emb, intent_embs)[0]
    best_idx = int(intent_scores.argmax())

    keywords = [kw for kw, _ in keyword_model.extract_keywords(text, top_n=10)]
    named_entities = [ent['word'] for ent in ner_pipeline(text)]

    prompt = f"Summarize this text in 2-3 lines:\n\n{text[:2000]}"
    summary = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=256
    ).choices[0].message.content.strip()

    return {
        # "filename": txt_file.name,
        "keywords": keywords,
        "intent": intent_labels[best_idx],
        "entities": named_entities,
        "summary": summary,
        "embedding": doc_emb.tolist()
    }