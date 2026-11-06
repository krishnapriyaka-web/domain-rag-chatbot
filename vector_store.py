import os
import json
import math
import re
from typing import List, Dict, Any, Tuple, Optional

HAVE_FAISS_LANGCHAIN = False
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.docstore.document import Document
    HAVE_FAISS_LANGCHAIN = True
except ImportError:
    try:
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.docstore.document import Document
        HAVE_FAISS_LANGCHAIN = True
    except ImportError:
        HAVE_FAISS_LANGCHAIN = False


class LightweightEmbeddingVectorStore:
   
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.documents: List[Dict[str, Any]] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[List[float]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())

    def add_documents(self, documents: List[Dict[str, Any]]):
        self.documents.extend(documents)
        self._build_index()

    def _build_index(self):
        if not self.documents:
            return

        doc_count = len(self.documents)
        doc_freqs: Dict[str, int] = {}
        tokenized_docs = []

        for doc in self.documents:
            tokens = self._tokenize(doc["page_content"])
            tokenized_docs.append(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                doc_freqs[t] = doc_freqs.get(t, 0) + 1

        sorted_terms = sorted(doc_freqs.items(), key=lambda x: x[1], reverse=True)
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(sorted_terms)}
        
        self.idf = {
            term: math.log((1 + doc_count) / (1 + freq)) + 1.0
            for term, freq in doc_freqs.items()
        }

        self.doc_vectors = []
        vocab_size = len(self.vocabulary)
        for tokens in tokenized_docs:
            vec = [0.0] * vocab_size
            for t in tokens:
                if t in self.vocabulary:
                    idx = self.vocabulary[t]
                    vec[idx] += 1.0
            
            norm_sq = 0.0
            for term, idx in self.vocabulary.items():
                if vec[idx] > 0:
                    tf = 1.0 + math.log(vec[idx])
                    val = tf * self.idf.get(term, 1.0)
                    vec[idx] = val
                    norm_sq += val * val
            
            norm = math.sqrt(norm_sq)
            if norm > 0:
                vec = [v / norm for v in vec]
            self.doc_vectors.append(vec)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        if not self.documents or not self.doc_vectors:
            return []

        q_tokens = self._tokenize(query)
        vocab_size = len(self.vocabulary)
        q_vec = [0.0] * vocab_size

        for t in q_tokens:
            if t in self.vocabulary:
                idx = self.vocabulary[t]
                q_vec[idx] += 1.0

        q_norm_sq = 0.0
        for term, idx in self.vocabulary.items():
            if q_vec[idx] > 0:
                tf = 1.0 + math.log(q_vec[idx])
                val = tf * self.idf.get(term, 1.0)
                q_vec[idx] = val
                q_norm_sq += val * val

        q_norm = math.sqrt(q_norm_sq)
        if q_norm > 0:
            q_vec = [v / q_norm for v in q_vec]
        else:
            return []

        scores = []
        for doc_idx, d_vec in enumerate(self.doc_vectors):
            dot_product = sum(q_vec[i] * d_vec[i] for i in range(vocab_size) if q_vec[i] > 0)
            scores.append((doc_idx, dot_product))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = []
        for doc_idx, score in scores[:k]:
            top_results.append((self.documents[doc_idx], float(score)))

        return top_results

    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        results_with_scores = self.similarity_search_with_score(query, k=k)
        return [doc for doc, _ in results_with_scores]

    def save_local(self, folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
        payload = {
            "model_name": self.embedding_model_name,
            "documents": self.documents,
            "vocabulary": self.vocabulary,
            "idf": self.idf,
            "doc_vectors": self.doc_vectors
        }
        with open(os.path.join(folder_path, "index.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f)
        with open(os.path.join(folder_path, "index.faiss"), "w") as f:
            f.write("FAISS_METADATA_STORE")

    @classmethod
    def load_local(cls, folder_path: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
        index_file = os.path.join(folder_path, "index.json")
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found in {folder_path}")
        with open(index_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        store = cls(embedding_model_name=payload.get("model_name", embedding_model_name))
        store.documents = payload.get("documents", [])
        store.vocabulary = payload.get("vocabulary", {})
        store.idf = payload.get("idf", {})
        store.doc_vectors = payload.get("doc_vectors", [])
        return store


class VectorStoreManager:
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", index_dir: str = "domain_rag_chatbot/vector_store/saved_index"):
        self.embedding_model_name = embedding_model_name
        self.index_dir = index_dir
        self.vector_store = None
        self.is_faiss_native = False

        if HAVE_FAISS_LANGCHAIN:
            try:
                self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
                self.is_faiss_native = True
            except Exception:
                self.embeddings = None
                self.is_faiss_native = False
        else:
            self.embeddings = None
            self.is_faiss_native = False

    def create_vector_store(self, chunked_documents: List[Dict[str, Any]]):
        if self.is_faiss_native:
            langchain_docs = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in chunked_documents
            ]
            self.vector_store = FAISS.from_documents(langchain_docs, self.embeddings)
        else:
            self.vector_store = LightweightEmbeddingVectorStore(self.embedding_model_name)
            self.vector_store.add_documents(chunked_documents)

    def save_vector_store(self, target_dir: Optional[str] = None):
        save_path = target_dir or self.index_dir
        os.makedirs(save_path, exist_ok=True)
        if self.vector_store is None:
            raise ValueError("No active vector store to save. Please process documents first.")

        if self.is_faiss_native:
            self.vector_store.save_local(save_path)
        else:
            self.vector_store.save_local(save_path)

    def load_vector_store(self, source_dir: Optional[str] = None) -> bool:
    
        load_path = source_dir or self.index_dir
        if not os.path.exists(load_path):
            return False

        try:
            if self.is_faiss_native:
                self.vector_store = FAISS.load_local(
                    load_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return True
            else:
                self.vector_store = LightweightEmbeddingVectorStore.load_local(
                    load_path,
                    self.embedding_model_name
                )
                return True
        except Exception:
            try:
                self.vector_store = LightweightEmbeddingVectorStore.load_local(
                    load_path,
                    self.embedding_model_name
                )
                self.is_faiss_native = False
                return True
            except Exception:
                return False

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized. Please load or index documents.")

        results = []
        if self.is_faiss_native:
            docs = self.vector_store.similarity_search(query, k=top_k)
            for d in docs:
                results.append({
                    "page_content": d.page_content,
                    "metadata": d.metadata
                })
        else:
            results = self.vector_store.similarity_search(query, k=top_k)

        return results
