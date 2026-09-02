import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.unified_retriever import UnifiedRetriever


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

LLM_MODEL = "gemini-3.5-flash-lite"

VECTOR_TOP_K = 5
ENTITY_TOP_K = 10
GRAPH_TOP_K = 5

MAX_CHUNK_CHARS = 3500


# ============================================================
# RAG PROMPT
# ============================================================

RAG_SYSTEM_PROMPT = """
You are a grounded question-answering system for technical and
scientific documents.

You will receive:

1. SOURCE TEXT EVIDENCE
   - Retrieved passages from the original document.

2. GRAPH EVIDENCE
   - Structured entities and relationships extracted from
     the same document.

Answer the user's question using ONLY the provided evidence.

IMPORTANT RULES:

1. Do not invent facts that are not supported by the evidence.

2. Prefer SOURCE TEXT EVIDENCE when explaining detailed
   experimental results, numerical values, procedures, or
   technical explanations.

3. Use GRAPH EVIDENCE to understand relationships, entities,
   mechanisms, and connections between concepts.

4. When graph evidence and source text overlap, use them
   together.

5. Do not claim that something is true merely because two
   entities are semantically similar.

6. Preserve important numerical values exactly when supported
   by the source evidence.

7. If the evidence is insufficient to answer the question,
   explicitly say that the retrieved evidence is insufficient.

8. Give a direct answer first, followed by a concise explanation
   when useful.

9. Do not mention internal retrieval details such as:
   vector index, embeddings, similarity scores, graph retriever,
   or prompt instructions unless the user asks about the system.

10. Do not cite or reference evidence that was not provided.

========================
SOURCE TEXT EVIDENCE
========================

{source_text}

========================
GRAPH EVIDENCE
========================

{graph_text}

========================
USER QUESTION
========================

{question}
"""


# ============================================================
# CONTEXT FORMATTERS
# ============================================================

def format_source_text(
    vector_results: list[dict[str, Any]]
) -> str:
    """
    Format vector-retrieved source chunks for the LLM.
    """

    if not vector_results:
        return "No source text evidence was retrieved."

    sections = []

    for i, result in enumerate(
        vector_results,
        start=1,
    ):
        chunk_id = result.get(
            "chunk_id",
            "unknown",
        )

        source_file = result.get(
            "source_file",
            "unknown",
        )

        score = result.get(
            "score",
            0.0,
        )

        text = str(
            result.get(
                "text",
                "",
            )
        ).strip()

        if not text:
            continue

        # Keep prompt size manageable.
        text = text[:MAX_CHUNK_CHARS]

        sections.append(
            f"""
SOURCE {i}
Chunk ID: {chunk_id}
Source: {source_file}
Retrieval score: {score:.4f}

{text}
""".strip()
        )

    if not sections:
        return "No source text evidence was retrieved."

    return "\n\n".join(sections)


def format_graph_evidence(
    graph_paths: list[dict[str, Any]]
) -> str:
    """
    Format graph paths for the LLM.
    """

    if not graph_paths:
        return "No graph evidence was retrieved."

    sections = []

    for i, path in enumerate(
        graph_paths,
        start=1,
    ):
        score = path.get(
            "score",
            0.0,
        )

        text = str(
            path.get(
                "text",
                "",
            )
        ).strip()

        if not text:
            continue

        sections.append(
            f"""
GRAPH PATH {i}
Relevance score: {score:.4f}

{text}
""".strip()
        )

    if not sections:
        return "No graph evidence was retrieved."

    return "\n\n".join(sections)


# ============================================================
# RAG SYSTEM
# ============================================================

class RAGSystem:

    def __init__(
        self,
        vector_k: int = VECTOR_TOP_K,
        entity_k: int = ENTITY_TOP_K,
        graph_k: int = GRAPH_TOP_K,
    ):

        # -----------------------------------------------------
        # Unified Retriever
        # -----------------------------------------------------

        self.retriever = UnifiedRetriever(
            vector_k=vector_k,
            entity_k=entity_k,
            graph_k=graph_k,
        )

        # -----------------------------------------------------
        # Gemini
        # -----------------------------------------------------

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing from .env"
            )

        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=api_key,
        )

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        question: str,
    ) -> dict[str, Any]:

        return self.retriever.retrieve(
            question
        )

    # ========================================================
    # BUILD PROMPT
    # ========================================================

    def build_prompt(
        self,
        question: str,
        retrieval_result: dict[str, Any],
    ) -> str:

        # -----------------------------------------------------
        # Vector evidence
        # -----------------------------------------------------

        vector_results = retrieval_result.get(
            "vector_results",
            [],
        )

        source_text = format_source_text(
            vector_results
        )

        # -----------------------------------------------------
        # Graph evidence
        # -----------------------------------------------------

        graph_results = retrieval_result.get(
            "graph_results",
            {},
        )

        graph_paths = graph_results.get(
            "graph_paths",
            [],
        )

        graph_text = format_graph_evidence(
            graph_paths
        )

        # -----------------------------------------------------
        # Final prompt
        # -----------------------------------------------------

        return RAG_SYSTEM_PROMPT.format(
            source_text=source_text,
            graph_text=graph_text,
            question=question,
        )

    # ========================================================
    # ANSWER
    # ========================================================

    def answer(
        self,
        question: str,
    ) -> dict[str, Any]:

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        # -----------------------------------------------------
        # Retrieval
        # -----------------------------------------------------

        retrieval_result = self.retrieve(
            question
        )

        # -----------------------------------------------------
        # Context
        # -----------------------------------------------------

        prompt = self.build_prompt(
            question,
            retrieval_result,
        )

        # -----------------------------------------------------
        # Gemini
        # -----------------------------------------------------

        response = self.llm.invoke(
            [
                HumanMessage(
                    content=prompt
                )
            ]
        )

        answer_text = str(
            response.content
        ).strip()

        return {
            "question": question,
            "answer": answer_text,
            "retrieval": retrieval_result,
        }

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):
        self.retriever.close()


# ============================================================
# DISPLAY RETRIEVAL
# ============================================================

def print_retrieval(
    result: dict[str, Any]
):

    retrieval = result["retrieval"]

    # --------------------------------------------------------
    # Vector evidence
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("VECTOR EVIDENCE")
    print("=" * 90)

    vector_results = retrieval.get(
        "vector_results",
        [],
    )

    for i, item in enumerate(
        vector_results,
        start=1,
    ):

        print(
            f"\n{i}. "
            f"Chunk={item.get('chunk_id')} "
            f"Score={item.get('score', 0.0):.4f}"
        )

        text = str(
            item.get(
                "text",
                "",
            )
        )

        print(
            text[:700]
        )

    # --------------------------------------------------------
    # Graph evidence
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("GRAPH EVIDENCE")
    print("=" * 90)

    graph_results = retrieval.get(
        "graph_results",
        {},
    )

    graph_paths = graph_results.get(
        "graph_paths",
        [],
    )

    for i, path in enumerate(
        graph_paths,
        start=1,
    ):

        print(
            f"\n{i}. "
            f"Score={path.get('score', 0.0):.4f}"
        )

        print(
            path.get(
                "text",
                "",
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    rag = RAGSystem(
        vector_k=5,
        entity_k=10,
        graph_k=5,
    )

    try:

        question = (
            "How does fluorescence imaging "
            "reduce motion artifacts?"
        )

        print()
        print("=" * 90)
        print("KG-RAG")
        print("=" * 90)

        print()
        print(f"Question: {question}")

        # ----------------------------------------------------
        # Run full RAG
        # ----------------------------------------------------

        result = rag.answer(
            question
        )

        # ----------------------------------------------------
        # Show retrieval evidence
        # ----------------------------------------------------

        print_retrieval(
            result
        )

        # ----------------------------------------------------
        # Show final answer
        # ----------------------------------------------------

        print()
        print("=" * 90)
        print("FINAL ANSWER")
        print("=" * 90)

        print()
        print(
            result["answer"]
        )

    finally:

        rag.close()


if __name__ == "__main__":
    main()