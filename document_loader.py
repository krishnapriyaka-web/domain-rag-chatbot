import os
import io
from typing import List, Dict, Any, Tuple, Optional
import pypdf

class DocumentLoader:
    def __init__(self, max_file_size_mb: int = 15):
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def validate_file(self, file_name: str, file_bytes: bytes) -> Tuple[bool, str]:
        if not file_name.lower().endswith(".pdf"):
            return False, f"Invalid file format: '{file_name}'. Only PDF (.pdf) documents are supported."
        
        file_size = len(file_bytes)
        if file_size > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            actual_mb = file_size / (1024 * 1024)
            return False, f"File '{file_name}' exceeds maximum permitted size of {max_mb:.1f} MB (Actual: {actual_mb:.2f} MB)."
        
        if not file_bytes.startswith(b"%PDF-"):
            return False, f"File '{file_name}' does not appear to be a valid PDF document (missing %PDF- header)."

        return True, "Valid PDF file."

    def extract_text_from_pdf(self, file_source, file_name: str) -> List[Dict[str, Any]]:
        if isinstance(file_source, bytes):
            reader = pypdf.PdfReader(io.BytesIO(file_source))
        elif isinstance(file_source, str):
            reader = pypdf.PdfReader(file_source)
        elif hasattr(file_source, "read"):
            content = file_source.read()
            if isinstance(content, str):
                content = content.encode("utf-8")
            reader = pypdf.PdfReader(io.BytesIO(content))
        else:
            raise ValueError(f"Unsupported file source type: {type(file_source)}")

        documents = []
        total_pages = len(reader.pages)

        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            cleaned_text = self._clean_text(text)
            
            if not cleaned_text.strip():
                continue

            documents.append({
                "page_content": cleaned_text,
                "metadata": {
                    "source": os.path.basename(file_name),
                    "document": os.path.basename(file_name),
                    "page": page_idx,
                    "total_pages": total_pages
                }
            })

        return documents

    def _clean_text(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        cleaned_lines = []
        for line in lines:
            if line:
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
        return "\n".join(cleaned_lines)


class TextChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        for doc in documents:
            text = doc["page_content"]
            meta = doc["metadata"]
            doc_chunks = self.split_text(text)
            
            for chunk_idx, chunk_text in enumerate(doc_chunks):
                chunk_meta = dict(meta)
                chunk_meta["chunk_index"] = chunk_idx
                chunks.append({
                    "page_content": chunk_text,
                    "metadata": chunk_meta
                })
        return chunks

    def split_text(self, text: str) -> List[str]:
        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._recursive_split(text, separators, self.chunk_size, self.chunk_overlap)

    def _recursive_split(self, text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        if len(text) <= chunk_size or not separators:
            return [text.strip()] if text.strip() else []

        sep = separators[0]
        remaining_seps = separators[1:]
        
        splits = text.split(sep)
        chunks = []
        current_chunk = ""

        for part in splits:
            candidate = f"{current_chunk}{sep}{part}" if current_chunk else part
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if len(part) > chunk_size and remaining_seps:
                    sub_chunks = self._recursive_split(part, remaining_seps, chunk_size, chunk_overlap)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk and current_chunk.strip():
            chunks.append(current_chunk.strip())

        return self._merge_with_overlap(chunks, chunk_size, chunk_overlap)

    def _merge_with_overlap(self, pieces: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        if len(pieces) <= 1:
            return pieces

        result = []
        for i, piece in enumerate(pieces):
            if i == 0:
                result.append(piece)
            else:
                prev_piece = pieces[i - 1]
                overlap_text = prev_piece[-chunk_overlap:] if len(prev_piece) >= chunk_overlap else prev_piece
                merged = f"...{overlap_text}\n{piece}"
                if len(merged) > chunk_size + 100:
                    merged = merged[:chunk_size]
                result.append(merged)
        return result
