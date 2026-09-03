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

# If the first answer contains one of these phrases,
# we perform a validation/correction pass.
SUSPICIOUS_ANSWER_PHRASES = [
    "insufficient evidence",
    "insufficient information",
    "not enough information",
    "not enough evidence",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "can't be determined",
    "not mentioned",
    "not provided",
    "does not contain information",
    "do not contain information",
    "no information",
    "unable to answer",
]


# ============================================================
# RAG PROMPT
# ============================================================

RAG_SYSTEM_PROMPT = """
You are a factual question-answering system for technical and
scientific documents.

You will receive:

1. SOURCE TEXT EVIDENCE
   Retrieved passages from the original document.

2. GRAPH EVIDENCE
   Structured entities and relationships extracted from
   the same document.

Your task is to answer the USER QUESTION using ONLY the
provided evidence.

IMPORTANT RULES:

1. EVIDENCE-FIRST
   Carefully inspect ALL source-text evidence and ALL graph
   evidence before deciding what the answer is.

2. NEVER MISS AN EXPLICIT ANSWER
   If the requested information appears anywhere in the
   retrieved evidence, answer it.
   Do NOT say "insufficient evidence", "not mentioned",
   "not provided", or "cannot be determined" when the
   requested information is present.

3. MULTI-CHUNK REASONING
   The answer may be distributed across multiple retrieved
   chunks. Combine complementary evidence from different
   chunks when necessary.

4. ADJACENT / INCOMPLETE INFORMATION
   A retrieved chunk may end in an incomplete sentence or
   incomplete fact. Another retrieved chunk may contain the
   continuation or missing value. Combine them before
   deciding that information is unavailable.

5. NUMERICAL QUESTIONS
   For questions asking about values, concentrations,
   percentages, factors, counts, dimensions, wavelengths,
   timings, resolutions, or other numbers:
   - Find every relevant number in the evidence.
   - Determine what condition each number belongs to.
   - Preserve the units.
   - Do not confuse values from different experiments.

6. COMPARISON QUESTIONS
   When the question asks "how did X compare with Y",
   explicitly identify both X and Y and compare them.

7. ARITHMETIC
   If the requested comparison requires simple arithmetic,
   calculate it from the retrieved values.
   For example, calculate differences, percentage changes,
   or ratios when the needed values are explicitly provided.

8. CONDITIONS MATTER
   Carefully distinguish between:
   - solution vs pig skin
   - DC / steady-state vs pulsed illumination
   - fluorescence sensitivity vs motion-artifact reduction
   - stationary vs motion tests
   - phantom vs biological imaging
   - experimental results vs proposed future improvements

9. DIRECT EVIDENCE
   Prefer direct source-text statements for numerical values,
   experimental results, procedures, and technical details.

10. GRAPH EVIDENCE
    Use graph evidence to connect entities, relationships,
    mechanisms, and multi-hop facts.
    Do not treat semantic similarity between two entities
    as proof of a relationship.

11. CONFLICTING EVIDENCE
    If retrieved evidence contains genuinely conflicting
    numerical or factual claims, report the conflict
    explicitly rather than silently choosing one.

12. DO NOT INVENT
    Never add facts that are not supported by the retrieved
    evidence.

13. INSUFFICIENT EVIDENCE
    Only say the evidence is insufficient after checking:
    - all source chunks
    - all graph paths
    - numerical values
    - possible multi-chunk combinations

14. ANSWER STYLE
    Give the direct answer first.
    Then provide a short explanation when useful.
    Keep the answer concise and factual.

15. DO NOT DISCUSS INTERNAL RAG DETAILS
    Do not mention embeddings, vector indexes, retrieval
    scores, graph retrievers, prompts, or internal pipeline
    details unless the user explicitly asks about the RAG system.

16. SOURCE BOUNDARY
    Use only the evidence supplied below.
    Do not rely on outside knowledge.

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

Now answer the user's question using the evidence above.
"""


# ============================================================
# VALIDATION PROMPT
# ============================================================

VALIDATION_PROMPT = """
You are validating an answer generated from retrieved document
evidence.

Your job is NOT to generate a new answer yet.

Check whether the DRAFT ANSWER is fully supported by the
retrieved evidence.

Return exactly one of:

SUPPORTED
UNSUPPORTED
NEEDS_CORRECTION

Use NEEDS_CORRECTION when any of the following is true:

- The draft says evidence is insufficient, but the answer
  appears explicitly in the evidence.
- The draft misses an explicit numerical value.
- The draft fails to combine information from multiple chunks.
- The draft confuses experimental conditions.
- The draft gives an incorrect numerical comparison.
- The draft contradicts the evidence.
- The draft ignores a directly relevant graph relationship.

Use UNSUPPORTED when the draft contains a factual claim that
cannot be supported by the retrieved evidence and the answer
cannot be repaired using the provided evidence.

Use SUPPORTED when the draft correctly answers the question
using the retrieved evidence.

========================
USER QUESTION
========================

{question}

========================
SOURCE TEXT EVIDENCE
========================

{source_text}

========================
GRAPH EVIDENCE
========================

{graph_text}

========================
DRAFT ANSWER
========================

{draft_answer}
"""


# ============================================================
# CORRECTION PROMPT
# ============================================================

CORRECTION_PROMPT = """
You are correcting a factual answer using retrieved evidence.

Answer the USER QUESTION again using ONLY the evidence below.

IMPORTANT:

1. If the answer is explicitly present, do not say that
   evidence is insufficient.

2. Combine multiple chunks when necessary.

3. Carefully extract all relevant numerical values.

4. Preserve units.

5. Keep experimental conditions separate.

6. If the question asks for a comparison, explicitly compare
   the requested values.

7. If simple arithmetic is needed, calculate it from the
   provided values.

8. If the evidence contains genuinely conflicting values,
   report the conflict explicitly.

9. Do not invent information.

10. Return ONLY the corrected answer.
    Do not discuss the validation process.

========================
USER QUESTION
========================

{question}

========================
SOURCE TEXT EVIDENCE
========================

{source_text}

========================
GRAPH EVIDENCE
========================

{graph_text}

========================
DRAFT ANSWER
========================

{draft_answer}
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
# GEMINI RESPONSE TEXT EXTRACTION
# ============================================================

def extract_response_text(response) -> str:
    """
    Safely extract plain text from a Gemini/LangChain response.

    Gemini responses may expose content as:
    - a plain string
    - a list of content blocks
    - dictionaries containing {"type": "text", "text": "..."}
    """

    content = getattr(response, "content", response)

    # --------------------------------------------------------
    # Case 1: plain string
    # --------------------------------------------------------

    if isinstance(content, str):
        return content.strip()

    # --------------------------------------------------------
    # Case 2: list of content blocks
    # --------------------------------------------------------

    if isinstance(content, list):

        text_parts = []

        for block in content:

            # Dictionary block
            if isinstance(block, dict):

                if block.get("type") == "text":
                    text = block.get(
                        "text",
                        "",
                    )

                    if text:
                        text_parts.append(
                            str(text)
                        )

                # Some versions may not provide type
                elif "text" in block:

                    text = block.get(
                        "text",
                        "",
                    )

                    if text:
                        text_parts.append(
                            str(text)
                        )

            # String block
            elif isinstance(block, str):

                text_parts.append(block)

        return "\n".join(
            text_parts
        ).strip()

    # --------------------------------------------------------
    # Case 3: fallback
    # --------------------------------------------------------

    return str(content).strip()


# ============================================================
# ANSWER SAFETY / VALIDATION HELPERS
# ============================================================

def is_suspicious_answer(
    answer: str,
) -> bool:
    """
    Detect answers that may have incorrectly declared the
    evidence insufficient.
    """

    normalized = answer.lower().strip()

    return any(
        phrase in normalized
        for phrase in SUSPICIOUS_ANSWER_PHRASES
    )


def normalize_validation_result(
    validation_text: str,
) -> str:
    """
    Convert the validator output into one of the expected
    statuses.
    """

    normalized = validation_text.strip().upper()

    if "NEEDS_CORRECTION" in normalized:
        return "NEEDS_CORRECTION"

    if "UNSUPPORTED" in normalized:
        return "UNSUPPORTED"

    if "SUPPORTED" in normalized:
        return "SUPPORTED"

    # If validator gives unexpected output, be conservative.
    return "NEEDS_CORRECTION"


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
    # BUILD CONTEXT
    # ========================================================

    def build_context(
        self,
        retrieval_result: dict[str, Any],
    ) -> tuple[str, str]:

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

        return source_text, graph_text

    # ========================================================
    # BUILD MAIN PROMPT
    # ========================================================

    def build_prompt(
        self,
        question: str,
        source_text: str,
        graph_text: str,
    ) -> str:

        return RAG_SYSTEM_PROMPT.format(
            source_text=source_text,
            graph_text=graph_text,
            question=question,
        )

    # ========================================================
    # BUILD VALIDATION PROMPT
    # ========================================================

    def build_validation_prompt(
        self,
        question: str,
        source_text: str,
        graph_text: str,
        draft_answer: str,
    ) -> str:

        return VALIDATION_PROMPT.format(
            question=question,
            source_text=source_text,
            graph_text=graph_text,
            draft_answer=draft_answer,
        )

    # ========================================================
    # BUILD CORRECTION PROMPT
    # ========================================================

    def build_correction_prompt(
        self,
        question: str,
        source_text: str,
        graph_text: str,
        draft_answer: str,
    ) -> str:

        return CORRECTION_PROMPT.format(
            question=question,
            source_text=source_text,
            graph_text=graph_text,
            draft_answer=draft_answer,
        )

    # ========================================================
    # GENERATE
    # ========================================================

    def generate_answer(
        self,
        prompt: str,
    ) -> str:

        response = self.llm.invoke(
            [
                HumanMessage(
                    content=prompt
                )
            ]
        )

        return extract_response_text(
            response
        )

    # ========================================================
    # VALIDATE
    # ========================================================

    def validate_answer(
        self,
        question: str,
        source_text: str,
        graph_text: str,
        draft_answer: str,
    ) -> str:

        validation_prompt = self.build_validation_prompt(
            question=question,
            source_text=source_text,
            graph_text=graph_text,
            draft_answer=draft_answer,
        )

        validation_response = self.generate_answer(
            validation_prompt
        )

        return normalize_validation_result(
            validation_response
        )

    # ========================================================
    # CORRECT
    # ========================================================

    def correct_answer(
        self,
        question: str,
        source_text: str,
        graph_text: str,
        draft_answer: str,
    ) -> str:

        correction_prompt = self.build_correction_prompt(
            question=question,
            source_text=source_text,
            graph_text=graph_text,
            draft_answer=draft_answer,
        )

        return self.generate_answer(
            correction_prompt
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

        source_text, graph_text = self.build_context(
            retrieval_result
        )

        # -----------------------------------------------------
        # Main generation
        # -----------------------------------------------------

        prompt = self.build_prompt(
            question=question,
            source_text=source_text,
            graph_text=graph_text,
        )

        draft_answer = self.generate_answer(
            prompt
        )

        final_answer = draft_answer
        validation_status = "NOT_RUN"

        # -----------------------------------------------------
        # Suspicious-answer validation
        #
        # Only run the second Gemini call when the first answer
        # looks suspicious.
        # -----------------------------------------------------

        if is_suspicious_answer(
            draft_answer
        ):

            validation_status = self.validate_answer(
                question=question,
                source_text=source_text,
                graph_text=graph_text,
                draft_answer=draft_answer,
            )

            # -------------------------------------------------
            # Correct only when necessary.
            # -------------------------------------------------

            if validation_status in {
                "NEEDS_CORRECTION",
                "UNSUPPORTED",
            }:

                final_answer = self.correct_answer(
                    question=question,
                    source_text=source_text,
                    graph_text=graph_text,
                    draft_answer=draft_answer,
                )

        # -----------------------------------------------------
        # Return
        # -----------------------------------------------------

        return {
            "question": question,
            "answer": final_answer,
            "draft_answer": draft_answer,
            "validation_status": validation_status,
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
        print(
            f"Question: {question}"
        )

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
        # Show draft / validation
        # ----------------------------------------------------

        print()
        print("=" * 90)
        print("DRAFT ANSWER")
        print("=" * 90)

        print()
        print(
            result.get(
                "draft_answer",
                "",
            )
        )

        print()
        print("=" * 90)
        print("VALIDATION STATUS")
        print("=" * 90)

        print()
        print(
            result.get(
                "validation_status",
                "NOT_RUN",
            )
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