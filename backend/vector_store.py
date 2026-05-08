import os
from pathlib import Path
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

from config.settings import CHROMA_PERSIST_DIR, BILLING_POLICIES_DIR, EMBEDDING_MODEL


class BillingKnowledgeBase:
    def __init__(self):
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.collection = self._client.get_or_create_collection(
            name="billing_knowledge",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        if self.collection.count() == 0:
            self._load_billing_policies()

    def _load_billing_policies(self):
        policies_dir = Path(BILLING_POLICIES_DIR)
        if not policies_dir.exists():
            return

        docs, ids, metas = [], [], []
        for file in policies_dir.glob("*.txt"):
            text = file.read_text(encoding="utf-8")
            chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 40]
            for j, chunk in enumerate(chunks):
                docs.append(chunk)
                ids.append(f"{file.stem}_{j}")
                metas.append({"source": file.name, "type": "billing_policy"})

        if docs:
            self.collection.add(documents=docs, ids=ids, metadatas=metas)

    def add_document_context(self, doc_id: str, text: str, metadata: dict = None):
        if metadata is None:
            metadata = {}
        metadata.setdefault("type", "prior_document")

        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 30]
        chunks = lines if lines else [text]

        self.collection.upsert(
            documents=chunks,
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
            metadatas=[metadata.copy() for _ in chunks],
        )

    def retrieve_context(self, query: str, n_results: int = 5) -> str:
        total = self.collection.count()
        if total == 0:
            return ""
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, total),
        )
        parts = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            src = meta.get("source", "policy")
            parts.append(f"[{src}]\n{doc}")
        return "\n\n".join(parts)

    def collection_count(self) -> int:
        return self.collection.count()


@lru_cache(maxsize=1)
def get_knowledge_base() -> BillingKnowledgeBase:
    return BillingKnowledgeBase()
