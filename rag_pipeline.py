import os
import sys
import re
from typing import List, Dict, Any, Optional, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from document_loader import DocumentLoader, TextChunker
from vector_store import VectorStoreManager
from prompt import PromptManager, STRICT_SYSTEM_PROMPT, FALLBACK_REFUSAL_MESSAGE


class RAGPipeline:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        top_k: int = 4,
        index_dir: Optional[str] = None,
        llm_provider: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.index_dir = index_dir or os.path.join(current_dir, "vector_store/saved_index")

        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_manager = VectorStoreManager(embedding_model_name=embedding_model, index_dir=self.index_dir)
        self.prompt_manager = PromptManager()

        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "auto")
        self.api_key = api_key

    def ingest_documents(self, file_inputs: List[Tuple[Any, str]]) -> Dict[str, Any]:
        all_raw_docs = []
        validation_errors = []
        processed_files = []

        for file_source, file_name in file_inputs:
            if isinstance(file_source, bytes):
                valid, msg = self.loader.validate_file(file_name, file_source)
                if not valid:
                    validation_errors.append(f"{file_name}: {msg}")
                    continue
            elif hasattr(file_source, "getvalue"):
                b = file_source.getvalue()
                valid, msg = self.loader.validate_file(file_name, b)
                if not valid:
                    validation_errors.append(f"{file_name}: {msg}")
                    continue
            elif isinstance(file_source, str) and os.path.exists(file_source):
                with open(file_source, "rb") as f:
                    b = f.read()
                valid, msg = self.loader.validate_file(file_name, b)
                if not valid:
                    validation_errors.append(f"{file_name}: {msg}")
                    continue

            docs = self.loader.extract_text_from_pdf(file_source, file_name)
            if docs:
                all_raw_docs.extend(docs)
                processed_files.append({
                    "name": file_name,
                    "pages_extracted": len(docs)
                })

        if not all_raw_docs:
            return {
                "success": False,
                "error": "No valid text could be extracted from uploaded files.",
                "validation_errors": validation_errors,
                "chunks_count": 0
            }

        chunks = self.chunker.split_documents(all_raw_docs)
        self.vector_manager.create_vector_store(chunks)
        self.vector_manager.save_vector_store(self.index_dir)

        return {
            "success": True,
            "processed_files": processed_files,
            "chunks_count": len(chunks),
            "pages_count": len(all_raw_docs),
            "validation_errors": validation_errors
        }

    def load_existing_index(self) -> bool:
        return self.vector_manager.load_vector_store(self.index_dir)

    def retrieve_chunks(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        return self.vector_manager.search(question, top_k=k)

    def answer_question(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        k = top_k or self.top_k
        try:
            retrieved_chunks = self.retrieve_chunks(question, top_k=k)
        except Exception as e:
            return {
                "answer": "Vector store not initialized. Please upload and process documents first.",
                "sources": [],
                "refusal": True,
                "error": str(e)
            }

        if not retrieved_chunks:
            return {
                "answer": FALLBACK_REFUSAL_MESSAGE,
                "sources": [],
                "refusal": True
            }

        context = self.prompt_manager.format_context(retrieved_chunks)
        user_prompt = self.prompt_manager.build_user_prompt(context, question)

        answer, is_refusal = self._generate_llm_response(question, context, user_prompt, retrieved_chunks)

        sources = []
        seen = set()
        for c in retrieved_chunks:
            meta = c.get("metadata", {})
            doc = meta.get("document", "Unknown")
            page = meta.get("page", 1)
            key = (doc, page)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "document": doc,
                    "page": page,
                    "snippet": c.get("page_content", "")[:300] + ("..." if len(c.get("page_content", "")) > 300 else "")
                })

        return {
            "answer": answer.strip(),
            "sources": sources,
            "refusal": is_refusal,
            "retrieved_chunks": retrieved_chunks
        }

    def _generate_llm_response(
        self,
        question: str,
        context: str,
        user_prompt: str,
        chunks: List[Dict[str, Any]]
    ) -> Tuple[str, bool]:
        provider = self.llm_provider.lower()
        api_key = self.api_key

        if (provider in ["groq", "auto"]) and (api_key or os.getenv("GROQ_API_KEY")):
            key = api_key or os.getenv("GROQ_API_KEY")
            try:
                from groq import Groq
                client = Groq(api_key=key)
                completion = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[
                        {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=600
                )
                ans = completion.choices[0].message.content
                is_refusal = FALLBACK_REFUSAL_MESSAGE.lower() in ans.lower()
                return ans, is_refusal
            except Exception:
                pass

        if (provider in ["openai", "auto"]) and (api_key or os.getenv("OPENAI_API_KEY")):
            key = api_key or os.getenv("OPENAI_API_KEY")
            try:
                import openai
                client = openai.OpenAI(api_key=key)
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=600
                )
                ans = completion.choices[0].message.content
                is_refusal = FALLBACK_REFUSAL_MESSAGE.lower() in ans.lower()
                return ans, is_refusal
            except Exception:
                pass

        if (provider in ["gemini", "auto"]) and (api_key or os.getenv("GEMINI_API_KEY")):
            key = api_key or os.getenv("GEMINI_API_KEY")
            try:
                import google.generativeai as genai
                genai.configure(api_key=key)
                model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    system_instruction=STRICT_SYSTEM_PROMPT
                )
                resp = model.generate_content(user_prompt)
                ans = resp.text
                is_refusal = FALLBACK_REFUSAL_MESSAGE.lower() in ans.lower()
                return ans, is_refusal
            except Exception:
                pass

        return self._grounded_extractive_fallback(question, chunks)

    def _grounded_extractive_fallback(self, question: str, chunks: List[Dict[str, Any]]) -> Tuple[str, bool]:
        injection_patterns = [
            r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
            r"reveal\s+(system\s+prompt|secret|api\s+key)",
            r"you\s+are\s+now\s+in\s+developer\s+mode",
            r"disregard\s+the\s+system\s+rules"
        ]
        for pattern in injection_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return (
                    "Security Notice: System security guardrails prevent modifying system instructions or overriding prompt rules. "
                    + FALLBACK_REFUSAL_MESSAGE,
                    True
                )

        q_words = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9]{3,}\b", question)]
        stopwords = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "the", "and", "for", "with", "about", "are", "can", "does", "from", "tell"}
        keywords = [w for w in q_words if w not in stopwords]

        if not keywords:
            return FALLBACK_REFUSAL_MESSAGE, True

        best_sentence = None
        best_doc = None
        best_page = None
        best_match_count = 0

        for chunk in chunks:
            content = chunk.get("page_content", "")
            meta = chunk.get("metadata", {})
            doc_name = meta.get("document", "Document")
            page_num = meta.get("page", 1)

            sentences = re.split(r"(?<=[.!?])\s+", content)
            for s in sentences:
                s_lower = s.lower()
                match_count = sum(1 for kw in keywords if kw in s_lower)
                if match_count > best_match_count:
                    best_match_count = match_count
                    best_sentence = s.strip()
                    best_doc = doc_name
                    best_page = page_num

        threshold = max(2, int(len(keywords) * 0.4))
        if best_sentence and best_match_count >= threshold:
            answer = f"{best_sentence}\n\n(Source: {best_doc}, Page {best_page})"
            return answer, False

        return FALLBACK_REFUSAL_MESSAGE, True
