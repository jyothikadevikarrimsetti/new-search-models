from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
import spacy

class KeywordExtractor:
    def __init__(self, model=None, nlp_model=None):
        self.keyword_model = model if model is not None else KeyBERT()
        self.nlp = nlp_model if nlp_model is not None else spacy.load("en_core_web_trf") if spacy.util.is_package("en_core_web_trf") else spacy.load("en_core_web_sm")

    def extract_keywords_tfidf(self, text, top_n=10):
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
        tfidf = vectorizer.fit_transform([text])
        scores = zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0])
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return [w for w, s in sorted_scores[:top_n]]

    def extract_keywords_keybert(self, text, top_n=10):
        return [kw for kw, _ in self.keyword_model.extract_keywords(text, top_n=top_n)]

    def extract_keywords_spacy(self, text, top_n=10):
        doc = self.nlp(text)
        noun_chunks = list(set(chunk.text.strip().lower() for chunk in doc.noun_chunks))
        nouns = list(set(token.lemma_ for token in doc if token.pos_ == 'NOUN' and not token.is_stop))
        return (noun_chunks + nouns)[:top_n]

    def extract_all(self, text, top_n=10):
        keybert_keywords = self.extract_keywords_keybert(text, top_n)
        tfidf_keywords = self.extract_keywords_tfidf(text, top_n)
        spacy_keywords = self.extract_keywords_spacy(text, top_n)
        all_keywords = keybert_keywords + tfidf_keywords + spacy_keywords
        unique_keywords = []
        for kw in all_keywords:
            if kw not in unique_keywords:
                unique_keywords.append(kw)
        return {
            # "keybert": keybert_keywords,
            # "tfidf": tfidf_keywords,
            # "spacy": spacy_keywords,
            "keywords": unique_keywords
        }
        
        
 
