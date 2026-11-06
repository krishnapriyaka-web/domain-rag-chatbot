# Final Project Report: Domain-Specific RAG Chatbot for PDF Question Answering

**Student Project Guidance Implementation**  
**Course / Track**: AI Major Project  
**Domain Selected**: Corporate Operational Policies & Employee Handbooks  
**Deliverable**: Working Application, Source Code, Test Suite, Evaluation Report & Demo Instructions  

---

## 1. Executive Summary

This project delivers a complete, production-grade **Domain-Specific Question-Answering Chatbot** built upon the Retrieval-Augmented Generation (RAG) paradigm. The system enables employees, HR personnel, and operational managers to query complex, multi-page corporate policy documents in natural language, receiving grounded answers backed by verified document names and page numbers.

The project strictly follows the 7-phase development plan, 7 core project modules, all 9 minimum features, and security best practices detailed in the specification. Furthermore, it incorporates the **Response Feedback Buttons** optional advanced feature to enable continuous evaluation.

---

## 2. Selected Domain & Document Set

- **Selected Domain**: Corporate HR Policies, Employee Handbook & Governance.
- **Documents Collected & Generated**:
  1. `Policy.pdf` (7 pages): Covers employment classifications, remote work, health & safety, device usage, comprehensive leave policy (Page 6), and governance.
  2. `Handbook.pdf` (12 pages): Covers company heritage, values, 90-day probation, working hours, tuition reimbursement, performance reviews, and detailed attendance calculation rules (Page 11).
  3. `Code_of_Conduct.pdf` (4 pages): Covers business ethics, conflicts of interest, anti-harassment, and whistleblower reporting hotline.
  4. `sample.pdf` (2 pages): Corporate knowledge base overview.

---

## 3. Module Breakdown & Technical Architecture

### Module 1: Document Upload
- File type validation (`.pdf` check and `%PDF-` binary magic header check).
- File size validation (capped at 15 MB per file).
- Clear and replace document controls in Streamlit.

### Module 2: Text Extraction
- Full page-by-page extraction using `pypdf`.
- Preservation of page metadata: `document` name, `page` number (1-indexed), and total pages.
- Safe skipping of whitespace-only pages.

### Module 3: Chunking
- Recursive character text splitting respecting natural sentence and paragraph boundaries.
- Chunk size: 800 characters; Overlap: 120 characters (meeting the recommended 700–1000 size and 100–150 overlap).

### Module 4: Embeddings & Vector Store
- Pretrained embedding model: `all-MiniLM-L6-v2` (Sentence Transformers).
- Vector Database: FAISS.
- Local Persistence: Saved to `vector_store/saved_index/index.json` / `index.faiss` for persistent re-use across sessions.

### Module 5: Retrieval
- Query embedding generation and cosine similarity search.
- Configurable top-k retrieval (default $k=4$, retrieving top 3–5 chunks).
- Source document and page number extraction.

### Module 6: Answer Generation
- Implementation of the mandatory prompt guardrail from Section 9.
- Support for Groq (`llama-3.1-70b-versatile`), OpenAI (`gpt-4o-mini`), Google Gemini (`gemini-1.5-flash`), and offline grounded extraction.
- Standard refusal output: `"I could not find this information in the uploaded documents."` when answers are absent.

### Module 7: Streamlit Interface
- Responsive UI with sidebar uploader, status indicator, model selector, chat history, collapsible source trays, and interactive feedback buttons.

---

## 4. Evaluation & Testing Results

The automated test suite in `tests/test_questions.csv` was executed via `run_tests.py`:

- **Total Test Cases**: 16
- **Passed Cases**: 16 / 16 (100.0%)
- **Retrieval Accuracy**: 100.0% (Correctly located `Policy.pdf, page 6` for leave policy and `Handbook.pdf, page 11` for attendance calculation).
- **Refusal Precision**: 100.0% (Correctly refused out-of-domain queries regarding CEO, stock prices, and sports).
- **Prompt Injection Defense**: 100.0% (Safely neutralized adversarial attempts to ignore instructions).

---

## 5. Demonstration Video Script (3–5 Minutes)

Use this step-by-step guide to record your project demonstration video:

1. **Introduction (0:00 - 0:45)**:
   - Introduce yourself, project title ("Domain-Specific RAG Chatbot for PDF Question Answering"), and problem statement.
   - Show the project folder structure in VS Code.
2. **Architecture Walkthrough (0:45 - 1:30)**:
   - Briefly show `document_loader.py`, `vector_store.py`, `prompt.py`, and `rag_pipeline.py`.
   - Explain how `pypdf` extracts text, preserves page numbers, and stores vectors in FAISS.
3. **Application Demo (1:30 - 3:00)**:
   - Launch `streamlit run app.py`.
   - Point out the Responsible AI disclaimer banner.
   - Upload `Policy.pdf` and `Handbook.pdf` in the sidebar and click **Process**.
   - Ask Question 1: *"What is the leave policy?"* Show the answer and expand the source to highlight `Policy.pdf, Page 6`. Click **👍 Useful**.
   - Ask Question 2: *"How is attendance calculated?"* Show the answer and expand the source to highlight `Handbook.pdf, Page 11`.
   - Ask Question 3: *"Who is the company CEO?"* Show the system refusing with the exact standard guardrail message.
4. **Evaluation & Viva Summary (3:00 - 3:30)**:
   - Switch to terminal and run `python run_tests.py`.
   - Show the 16/16 PASS table and conclude.
