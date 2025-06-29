import re
import spacy
from transformers import pipeline

class EntityExtractor:
    def __init__(self, nlp, ner_pipe):
        self.nlp = nlp
        self.ner_pipe = ner_pipe
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+')
        self.date_pattern = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')

    def extract_spacy(self, text):
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def extract_transformers(self, text, score_threshold=0.8):
        return [
            (ent['word'], ent['entity_group'], ent['score'])
            for ent in self.ner_pipe(text)
            if ent['score'] > score_threshold
        ]

    def extract_regex(self, text):
        emails = self.email_pattern.findall(text)
        dates = self.date_pattern.findall(text)
        return {'emails': emails, 'dates': dates}

    def extract_entities_hybrid(self, text):
        entities = set()
        entity_types = []
        entity_details = []

        # spaCy
        for ent in self.nlp(text).ents:
            entities.add(ent.text)
            entity_types.append(ent.label_)
            entity_details.append({
                "text": ent.text,
                "type": ent.label_,
                "score": None
            })

        # Transformers
        for ent in self.ner_pipe(text):
            if ent['score'] > 0.8:
                entities.add(ent['word'])
                entity_types.append(ent['entity_group'])
                entity_details.append({
                    "text": ent['word'],
                    "type": ent['entity_group'],
                    "score": ent['score']
                })

        # Regex
        for email in self.email_pattern.findall(text):
            entities.add(email)
            entity_types.append("EMAIL")
            entity_details.append({
                "text": email,
                "type": "EMAIL",
                "score": None
            })
        for date in self.date_pattern.findall(text):
            entities.add(date)
            entity_types.append("DATE")
            entity_details.append({
                "text": date,
                "type": "DATE",
                "score": None
            })

        return {
            "entities": sorted(entities),
            "entity_types": set(entity_types),
            "entity_details": entity_details
        }
