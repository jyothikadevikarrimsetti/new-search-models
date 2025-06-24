import os
import re
import glob
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import concurrent.futures
import logging
import tiktoken
import numpy as np
from openai import AzureOpenAI

class IntentClassifier:
    def __init__(self, expanded_data, project_root, model_name="distilbert-base-uncased"):
        self.expanded_data = expanded_data
        self.project_root = project_root
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.label2id = None
        self.id2label = None
        self.trainer = None
        self.train_ds = None
        self.test_ds = None

    def build_label_mappings(self, df):
        df['label_id'] = df['label'].astype('category').cat.codes
        self.label2id = {label: i for i, label in enumerate(df['label'].astype('category').cat.categories)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        return df

    def prepare_dataset(self):
        df = pd.DataFrame(self.expanded_data)
        df = self.build_label_mappings(df)
        df_for_hf = df.rename(columns={'label_id': 'labels'}).drop(columns=['label'])
        dataset = Dataset.from_pandas(df_for_hf)
        dataset = dataset.map(lambda example: self.tokenizer(example["text"], truncation=True, padding="max_length", max_length=64), batched=True)
        split = dataset.train_test_split(test_size=0.2)
        self.train_ds, self.test_ds = split["train"], split["test"]

    def setup_model_and_trainer(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.label2id), id2label=self.id2label, label2id=self.label2id)
        training_args = TrainingArguments(
            output_dir="./intent_model",
            eval_strategy="epoch",
            save_strategy="epoch",
            do_eval=True,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            logging_steps=10,
            load_best_model_at_end=True,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_ds,
            eval_dataset=self.test_ds,
            tokenizer=self.tokenizer,
        )

    def train(self):
        self.prepare_dataset()
        self.setup_model_and_trainer()
        self.trainer.train()

    def predict_intent(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits
        pred_id = logits.argmax(dim=1).item()
        intent = self.id2label[pred_id]
        confidence = logits.softmax(dim=1)[0, pred_id].item()
        return intent, confidence

    def add_labeled_from_file(self, correct_intents_path):
        chunk_dir = os.path.join(self.project_root, 'data', 'chunks')
        labeled_from_file = []
        with open(correct_intents_path, 'r', encoding='utf-8') as f:
            for line in f:
                m = re.match(r'([^:]+):.*?\'intent\': \'([^\']+)\'', line)
                if m:
                    fname, intent = m.group(1).strip(), m.group(2).strip()
                    chunk_path = os.path.join(chunk_dir, fname)
                    if os.path.exists(chunk_path):
                        with open(chunk_path, 'r', encoding='utf-8') as cf:
                            chunk_text = cf.read()
                        labeled_from_file.append({"text": chunk_text, "label": intent})
                    else:
                        print(f"⚠️ Chunk file not found: {chunk_path}")
                else:
                    print(f"⚠️ Could not parse line: {line.strip()}")
        print(f"Loaded {len(labeled_from_file)} labeled examples from correct_intents.txt.")
        self.expanded_data.extend(labeled_from_file)
        print(f"expanded_data now has {len(self.expanded_data)} examples (including those from correct_intents.txt).")

    def batch_predict_chunks(self):
        chunk_dir = os.path.join(self.project_root, 'data', 'chunks')
        chunk_files = glob.glob(os.path.join(chunk_dir, '*.txt'))
        for chunk_path in chunk_files:
            with open(chunk_path, 'r', encoding='utf-8') as f:
                chunk_text = f.read()
            print(f"{os.path.basename(chunk_path)}: {self.predict_intent(chunk_text) , self.get_embedding(chunk_text)}")

    def get_embedding(self, text):
        # Get the embedding for the input text using the model's encoder
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            if hasattr(self.model, "distilbert"):
                outputs = self.model.distilbert(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            elif hasattr(self.model, "bert"):
                outputs = self.model.bert(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]
            elif hasattr(self.model, "roberta"):
                outputs = self.model.roberta(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]
            else:
                outputs = self.model.base_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]
        return embedding.cpu().numpy().flatten()

    def get_openai_embedding(self, text, timeout=15):
        """Get embeddings using Azure OpenAI's text-embedding model with context window truncation and timeout."""
        # Truncate text to fit within model context window (e.g., 8000 tokens for text-embedding-3-small)
        max_tokens = 8000
        encoding = tiktoken.encoding_for_model("text-embedding-3-small")
        tokens = encoding.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = encoding.decode(tokens)
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable not set.")
        if not api_base:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable not set.")
        if not deployment_id:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set.")
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_base,
        )
        def call():
            return client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            ).data[0].embedding
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(call)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logging.error(f"OpenAI embedding call timed out for text: {text[:50]}")
                raise TimeoutError("OpenAI embedding call timed out.")


