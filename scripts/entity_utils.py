# scripts/entity_utils.py
import re

# Shared spaCy loader utility
_spacy_nlp = None

def get_spacy_nlp():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            try:
                _spacy_nlp = spacy.load("en_core_web_trf")
            except Exception:
                _spacy_nlp = spacy.load("en_core_web_sm")
        except Exception:
            _spacy_nlp = None
    return _spacy_nlp

def normalize_entity(e):
    """Lowercase, remove dots, extra spaces, collapse multiple spaces, lemmatize if possible."""
    text = re.sub(r'\s+', ' ', re.sub(r'\.', '', e.lower())).strip()
    nlp = get_spacy_nlp()
    if nlp:
        doc = nlp(text)
        text = ' '.join([token.lemma_ for token in doc])
    return text
