from typing import Dict, Any, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.componenets.vector_store import load_vector_store, get_uploaded_documents_info
from app.componenets.load_llm import load_llm
from app.config.config import TOP_K

logger = get_logger(__name__)

GENERIC_RAG_PROMPT = """You are a document question-answering assistant.

Answer the user's question using the provided document context.

Rules:
1. Use the retrieved context as the primary source.
2. Do not invent information that is not supported by the context.
3. If the answer cannot be found in the uploaded documents, clearly say that the information was not found in the provided documents.
4. Give a concise but useful answer.
5. When possible, mention the relevant document/page/section information.

Formatting rules (MANDATORY - use GitHub-flavored Markdown):
- Use ## or ### headings to separate sections, and **bold** for key terms.
- Use "- " bullet points or "1." numbered lists for anything with multiple items.
- Whenever the answer compares items or lists values, ranges, doses, symptoms,
  parameters or any multi-attribute data, ALWAYS render it as a Markdown table
  with a header row and a |---| separator row, e.g.:
  | Column A | Column B |
  |---|---|
  | value | value |
- Keep tables to 5 columns maximum. If data does not fit, use bullets instead.
- Never output raw pipe characters (|) outside of a table.
- Do not wrap the whole answer in a code block.
- End the answer with a line: Source: <document filename>, Page <page number>

Context:
{context}

Question:
{question}

Answer:"""


def get_rag_prompt_template() -> PromptTemplate:
    """Return PromptTemplate for generic RAG QA."""
    return PromptTemplate(
        template=GENERIC_RAG_PROMPT,
        input_variables=["context", "question"]
    )


def query_rag(query: str, top_k: Optional[int] = None, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Execute end-to-end RAG retrieval and generation for a user query.

    Args:
        query: user question
        top_k: optional override for number of retrieved chunks
        document_ids: optional list of document_ids to restrict retrieval to
                      (None or empty => search all uploaded documents)

    Returns dictionary with 'answer' and 'sources' list.
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        return {
            "answer": "Please ask a question about your uploaded document(s).",
            "sources": []
        }

    try:
        logger.info(f"Processing user query: '{clean_query}'")
        db = load_vector_store()
        if db is None:
            logger.warning("Vector store is not available. Prompting user to upload documents.")
            return {
                "answer": "No documents have been uploaded yet. Please upload a PDF or DOCX document first to begin asking questions.",
                "sources": []
            }

        llm = load_llm()
        if llm is None:
            raise CustomException("LLM service is not initialized or GROQ_API_KEY is missing.")

        # Registry of currently-existing documents (deleted docs are NOT listed)
        registry = get_uploaded_documents_info()
        active_ids = {d.get("document_id") for d in registry if d.get("document_id")}
        active_names = {d.get("filename") for d in registry if d.get("filename")}

        # Optional per-question document filter (empty => all documents)
        selected_ids = set(document_ids) if document_ids else None
        if selected_ids is not None:
            selected_ids = {i for i in selected_ids if i in active_ids}
            if not selected_ids:
                selected_ids = None  # nothing valid selected => search all

        k_value = top_k if top_k is not None else TOP_K
        logger.info(f"Retrieving top {k_value} relevant chunks for query...")

        # Fetch extra candidates when filtering, so we can still fill top_k
        fetch_k = k_value * 4 if selected_ids else k_value
        retrieved_docs = db.similarity_search(clean_query, k=fetch_k)

        # ---- Metadata validation (consistency guard) ----
        # Drop any chunk that is no longer in the registry (e.g. deleted docs)
        # or that is outside the user's selected document filter.
        valid_docs = []
        for doc in retrieved_docs:
            md = doc.metadata or {}
            d_id = md.get("document_id")
            src = md.get("source")
            # A chunk must map to a live registry entry (by id, or legacy by filename)
            is_live = (d_id in active_ids) if d_id else (src in active_names)
            if not is_live:
                continue
            if selected_ids is not None:
                if d_id not in selected_ids:
                    continue
            valid_docs.append(doc)
            if len(valid_docs) >= k_value:
                break

        if not valid_docs:
            logger.info("No matching chunks found in vector store.")
            return {
                "answer": "The information was not found in the provided documents.",
                "sources": []
            }

        # Build context with clear document / page markers
        context_parts = []
        sources: List[Dict[str, Any]] = []
        seen_sources = set()

        for doc in valid_docs:
            source_file = doc.metadata.get("source", "Document")
            page_val = doc.metadata.get("page", "N/A")
            
            # Format source tag
            if page_val != "N/A" and page_val is not None:
                source_label = f"{source_file} (Page {page_val})"
                citation_key = f"{source_file}-page-{page_val}"
            else:
                source_label = f"{source_file}"
                citation_key = f"{source_file}-docx"

            context_parts.append(f"[{source_label}]:\n{doc.page_content}")

            if citation_key not in seen_sources:
                seen_sources.add(citation_key)
                sources.append({
                    "document": source_file,
                    "page": page_val,
                    "label": source_label,
                    "document_id": doc.metadata.get("document_id", ""),
                    "snippet": doc.page_content[:180].strip() + "..."
                })

        full_context = "\n\n---\n\n".join(context_parts)

        # Generate answer with LLM
        prompt = get_rag_prompt_template().format(
            context=full_context,
            question=clean_query
        )

        response = llm.invoke(prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)

        logger.info(f"Generated answer successfully ({len(answer_text)} chars, {len(sources)} sources).")

        return {
            "answer": answer_text.strip(),
            "sources": sources
        }

    except Exception as e:
        error_msg = CustomException("Failed to generate answer from documents.", e)
        logger.error(str(error_msg))
        return {
            "answer": "An error occurred while analyzing the document. Please try again or re-upload the document.",
            "sources": [],
            "error": str(e)
        }


def create_qa_chain():
    """
    Backwards compatibility QA chain factory.
    """
    try:
        db = load_vector_store()
        if db is None:
            logger.warning("Vector store is not present.")
            return None

        llm = load_llm()
        if llm is None:
            logger.warning("LLM is not present.")
            return None

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": TOP_K}),
            chain_type_kwargs={"prompt": get_rag_prompt_template()},
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        error_msg = CustomException("Failed to create QA chain.", e)
        logger.error(str(error_msg))
        return None