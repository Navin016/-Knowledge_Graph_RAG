"""
Runtime KG-RAG question answering service.

Keeps the existing retrieval architecture unchanged:

    question
        ↓
    UnifiedRetriever
        ↓
    vector + graph evidence
        ↓
    Gemini answer
        ↓
    optional validation/correction
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.unified_retriever import UnifiedRetriever


# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env",
    )
)


# ============================================================
# CONFIGURATION
# ============================================================

ANSWER_MODEL = os.getenv(
    "GEMINI_ANSWER_MODEL",
    "gemini-3.5-flash-lite",
)


VECTOR_TOP_K = 5
ENTITY_TOP_K = 10
GRAPH_TOP_K = 5


# Maximum amount of text included from each vector result
# in the Gemini context.
VECTOR_CONTEXT_LIMIT = 1800


# Maximum amount of graph-path text included in the prompt.
GRAPH_CONTEXT_LIMIT = 1200


# ============================================================
# SUSPICIOUS ANSWER DETECTION
# ============================================================

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
    "unable to determine",
]


# ============================================================
# MAIN ANSWER PROMPT
# ============================================================

ANSWER_PROMPT = """
You are the final answer-generation component of a grounded
KG-RAG system for technical and scientific documents.

You must answer the user's question using ONLY the retrieved
evidence provided below.

IMPORTANT RULES:

1. EVIDENCE-FIRST
   Carefully inspect ALL vector evidence and ALL graph evidence
   before answering.

2. USE EXPLICIT FACTS
   If the answer appears anywhere in the retrieved evidence,
   you MUST answer it.

3. DO NOT FALSELY CLAIM MISSING INFORMATION
   Do not say:
   - "insufficient evidence"
   - "insufficient information"
   - "not mentioned"
   - "not provided"
   - "cannot be determined"

   when the requested information is actually present
   anywhere in the retrieved evidence.

4. MULTI-CHUNK ANSWERS
   A complete answer may be distributed across multiple chunks.
   Combine information from multiple retrieved chunks when
   necessary.

5. INCOMPLETE CHUNKS
   A chunk may end in the middle of a sentence or fact.
   Another retrieved chunk may contain the continuation.
   Combine the available evidence before concluding that
   information is missing.

6. NUMERICAL QUESTIONS
   For questions involving numbers, percentages, factors,
   concentrations, counts, wavelengths, timings, resolutions,
   or other measurements:
   - identify every relevant number
   - identify the condition associated with each number
   - preserve the original units
   - do not mix values from different experiments

7. COMPARISON QUESTIONS
   When the question asks how two values compare:
   - identify both values
   - identify their respective conditions
   - explicitly compare them

8. SIMPLE CALCULATIONS
   When the required values are explicitly present, calculate
   simple differences, ratios, percentage changes, or factors
   when needed to answer the question.

9. PRESERVE EXPERIMENTAL CONDITIONS
   Distinguish carefully between:
   - solution vs pig skin
   - DC / steady-state vs pulsed light
   - stationary vs motion experiments
   - imaging phantom vs biological tissue
   - fluorescence sensitivity vs motion-artifact reduction

10. SOURCE TEXT
    Prefer direct source text for detailed technical explanations,
    numerical results, procedures, and experimental findings.

11. GRAPH EVIDENCE
    Use graph evidence to understand relationships, entities,
    mechanisms, and connections.

12. DO NOT INVENT
    Never add facts that are not supported by the retrieved evidence.

13. CONFLICTING EVIDENCE
    If the retrieved evidence contains genuinely conflicting
    numerical or factual claims, explicitly mention the conflict
    instead of silently selecting one.

14. INSUFFICIENT EVIDENCE
    Only state that evidence is insufficient after checking ALL
    provided vector evidence and graph evidence.

15. DIRECT ANSWER
    Answer the question directly first.
    Add a concise explanation when useful.

16. DO NOT DISCUSS INTERNAL SYSTEM DETAILS
    Do not mention embeddings, vector indexes, similarity scores,
    retrieval algorithms, graph retrievers, prompts, or internal
    implementation details unless the user explicitly asks.

17. SOURCE BOUNDARY
    Use ONLY the evidence supplied below.
    Do not use outside knowledge.

============================================================
USER QUESTION
============================================================

{question}

============================================================
VECTOR EVIDENCE
============================================================

{vector_context}

============================================================
GRAPH EVIDENCE
============================================================

{graph_context}

============================================================

Return only the final answer in normal readable text.
""".strip()


# ============================================================
# VALIDATION PROMPT
# ============================================================

VALIDATION_PROMPT = """
You are validating an answer produced from retrieved document
evidence.

Determine whether the draft answer correctly uses the provided
evidence.

Return exactly ONE of:

SUPPORTED
UNSUPPORTED
NEEDS_CORRECTION

Use NEEDS_CORRECTION when:

- the draft says evidence is insufficient but the answer is
  present in the evidence
- the draft misses an explicit number
- the draft fails to combine multiple relevant chunks
- the draft confuses experimental conditions
- the draft gives an incorrect numerical comparison
- the draft contradicts the evidence
- the draft ignores a directly relevant graph relationship

Use UNSUPPORTED when the draft contains factual claims that
cannot be supported by the provided evidence.

Use SUPPORTED when the draft correctly answers the question
using the provided evidence.

============================================================
USER QUESTION
============================================================

{question}

============================================================
VECTOR EVIDENCE
============================================================

{vector_context}

============================================================
GRAPH EVIDENCE
============================================================

{graph_context}

============================================================
DRAFT ANSWER
============================================================

{draft_answer}

Return only:
SUPPORTED
UNSUPPORTED
or
NEEDS_CORRECTION
""".strip()


# ============================================================
# CORRECTION PROMPT
# ============================================================

CORRECTION_PROMPT = """
Correct the draft answer using ONLY the retrieved evidence.

USER QUESTION:
{question}

VECTOR EVIDENCE:
{vector_context}

GRAPH EVIDENCE:
{graph_context}

DRAFT ANSWER:
{draft_answer}

IMPORTANT:

- If the answer is explicitly present in the evidence, answer it.
- Do not claim that evidence is insufficient when the answer
  appears in the evidence.
- Combine multiple chunks when necessary.
- Extract all relevant numerical values.
- Preserve units.
- Preserve experimental conditions.
- For comparisons, explicitly compare the requested values.
- Perform simple arithmetic when required.
- If the evidence contains a genuine conflict, report it.
- Do not invent unsupported facts.

Return ONLY the corrected final answer.
""".strip()


# ============================================================
# RESPONSE TEXT EXTRACTION
# ============================================================

def extract_response_text(
    response: Any,
) -> str:
    """
    Convert LangChain/Gemini response content into plain text.
    """

    content = getattr(
        response,
        "content",
        response,
    )

    # --------------------------------------------------------
    # Plain string
    # --------------------------------------------------------

    if isinstance(
        content,
        str,
    ):

        return content.strip()


    # --------------------------------------------------------
    # List of content blocks
    # --------------------------------------------------------

    if isinstance(
        content,
        list,
    ):

        text_parts: list[str] = []


        for block in content:

            if isinstance(
                block,
                dict,
            ):

                if block.get(
                    "type"
                ) == "text":

                    text = block.get(
                        "text",
                        "",
                    )

                    if text:

                        text_parts.append(
                            str(text)
                        )

                elif "text" in block:

                    text = block.get(
                        "text",
                        "",
                    )

                    if text:

                        text_parts.append(
                            str(text)
                        )


            elif isinstance(
                block,
                str,
            ):

                text_parts.append(
                    block
                )


        return "\n".join(
            text_parts
        ).strip()


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return str(
        content
    ).strip()


# ============================================================
# TEXT TRIMMING
# ============================================================

def _trim_text(
    value: Any,
    limit: int = 1200,
) -> str:

    text = str(
        value or ""
    ).strip()


    if len(text) <= limit:

        return text


    return (
        text[:limit]
        .rstrip()
        + "..."
    )


# ============================================================
# SUSPICIOUS ANSWER CHECK
# ============================================================

def _is_suspicious_answer(
    answer: str,
) -> bool:
    """
    Detect answers that may have incorrectly rejected available
    evidence.
    """

    normalized = (
        answer
        .lower()
        .strip()
    )


    return any(
        phrase in normalized
        for phrase in SUSPICIOUS_ANSWER_PHRASES
    )


# ============================================================
# VALIDATION STATUS
# ============================================================

def _normalize_validation_result(
    text: str,
) -> str:
    """
    Normalize Gemini's validation response.
    """

    normalized = (
        text
        .strip()
        .upper()
    )


    if "NEEDS_CORRECTION" in normalized:

        return "NEEDS_CORRECTION"


    if "UNSUPPORTED" in normalized:

        return "UNSUPPORTED"


    if "SUPPORTED" in normalized:

        return "SUPPORTED"


    # Be conservative if Gemini returns something unexpected.
    return "NEEDS_CORRECTION"


# ============================================================
# RAG SERVICE
# ============================================================

class RAGService:
    """
    Reusable query-time service for FastAPI.

    Existing architecture remains:

        question
            ↓
        UnifiedRetriever
            ↓
        vector + graph evidence
            ↓
        Gemini
            ↓
        optional validation/correction
    """

    def __init__(
        self,
        vector_k: int = VECTOR_TOP_K,
        entity_k: int = ENTITY_TOP_K,
        graph_k: int = GRAPH_TOP_K,
    ) -> None:

        # ----------------------------------------------------
        # Existing unified retriever
        # ----------------------------------------------------

        self.retriever = UnifiedRetriever(
            vector_k=vector_k,
            entity_k=entity_k,
            graph_k=graph_k,
        )


        # ----------------------------------------------------
        # Gemini answer model
        # ----------------------------------------------------

        self.llm = ChatGoogleGenerativeAI(
            model=ANSWER_MODEL,
            temperature=0,
        )


    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        question: str,
    ) -> dict[str, Any]:

        return self.retriever.retrieve(
            question
        )


    # ========================================================
    # CONTEXT BUILDING
    # ========================================================

    def build_context(
        self,
        retrieval: dict[str, Any],
    ) -> tuple[
        str,
        str,
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """
        Build LLM context while preserving the original evidence
        returned to the frontend.
        """

        # ----------------------------------------------------
        # Vector results
        # ----------------------------------------------------

        vector_results = retrieval.get(
            "vector_results",
            [],
        )


        vector_context_parts: list[str] = []


        for index, item in enumerate(
            vector_results,
            start=1,
        ):

            vector_context_parts.append(
                (
                    "VECTOR EVIDENCE {index}\n"
                    "chunk_id: {chunk_id}\n"
                    "source_file: {source_file}\n"
                    "score: {score:.4f}\n"
                    "text:\n{text}"
                ).format(
                    index=index,
                    chunk_id=item.get(
                        "chunk_id",
                        "",
                    ),
                    source_file=item.get(
                        "source_file",
                        "",
                    ),
                    score=float(
                        item.get(
                            "score",
                            0.0,
                        )
                    ),
                    text=_trim_text(
                        item.get(
                            "text",
                            "",
                        ),
                        VECTOR_CONTEXT_LIMIT,
                    ),
                )
            )


        # ----------------------------------------------------
        # Graph results
        # ----------------------------------------------------

        graph_results = retrieval.get(
            "graph_results",
            {},
        )


        graph_paths = graph_results.get(
            "graph_paths",
            [],
        )


        entities = graph_results.get(
            "entities",
            [],
        )


        graph_context_parts: list[str] = []


        for index, path in enumerate(
            graph_paths,
            start=1,
        ):

            graph_context_parts.append(
                (
                    "GRAPH PATH {index}\n"
                    "score: {score:.4f}\n"
                    "semantic_score: {semantic_score:.4f}\n"
                    "path: {text}"
                ).format(
                    index=index,
                    score=float(
                        path.get(
                            "score",
                            0.0,
                        )
                    ),
                    semantic_score=float(
                        path.get(
                            "semantic_score",
                            0.0,
                        )
                    ),
                    text=_trim_text(
                        path.get(
                            "text",
                            "",
                        ),
                        GRAPH_CONTEXT_LIMIT,
                    ),
                )
            )


        vector_context = (
            "\n\n".join(
                vector_context_parts
            )
            or "No vector evidence retrieved."
        )


        graph_context = (
            "\n\n".join(
                graph_context_parts
            )
            or "No graph evidence retrieved."
        )


        return (
            vector_context,
            graph_context,
            vector_results,
            graph_paths,
            entities,
        )


    # ========================================================
    # GENERATE
    # ========================================================

    def _generate(
        self,
        prompt: str,
    ) -> str:
        """
        Run Gemini and convert response to plain text.
        """

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

    def _validate(
        self,
        question: str,
        vector_context: str,
        graph_context: str,
        draft_answer: str,
    ) -> str:
        """
        Validate a suspicious draft answer.
        """

        prompt = VALIDATION_PROMPT.format(
            question=question,
            vector_context=vector_context,
            graph_context=graph_context,
            draft_answer=draft_answer,
        )


        validation_text = self._generate(
            prompt
        )


        return _normalize_validation_result(
            validation_text
        )


    # ========================================================
    # CORRECT
    # ========================================================

    def _correct(
        self,
        question: str,
        vector_context: str,
        graph_context: str,
        draft_answer: str,
    ) -> str:
        """
        Correct a draft answer using the retrieved evidence.
        """

        prompt = CORRECTION_PROMPT.format(
            question=question,
            vector_context=vector_context,
            graph_context=graph_context,
            draft_answer=draft_answer,
        )


        return self._generate(
            prompt
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


        # ----------------------------------------------------
        # 1. Retrieve
        # ----------------------------------------------------

        retrieval = self.retrieve(
            question
        )


        # ----------------------------------------------------
        # 2. Build context
        # ----------------------------------------------------

        (
            vector_context,
            graph_context,
            vector_results,
            graph_paths,
            entities,
        ) = self.build_context(
            retrieval
        )


        # ----------------------------------------------------
        # 3. Main Gemini answer
        # ----------------------------------------------------

        prompt = ANSWER_PROMPT.format(
            question=question,
            vector_context=vector_context,
            graph_context=graph_context,
        )


        draft_answer = self._generate(
            prompt
        )


        final_answer = draft_answer

        validation_status = "NOT_RUN"


        # ----------------------------------------------------
        # 4. Suspicious answer detection
        #
        # Normal answers use ONE Gemini call.
        #
        # If Gemini incorrectly says evidence is missing,
        # we perform validation and correction.
        # ----------------------------------------------------

        if _is_suspicious_answer(
            draft_answer
        ):

            validation_status = self._validate(
                question=question,
                vector_context=vector_context,
                graph_context=graph_context,
                draft_answer=draft_answer,
            )


            # ------------------------------------------------
            # Correct when validation detects a problem
            # ------------------------------------------------

            if validation_status in {
                "NEEDS_CORRECTION",
                "UNSUPPORTED",
            }:

                corrected_answer = self._correct(
                    question=question,
                    vector_context=vector_context,
                    graph_context=graph_context,
                    draft_answer=draft_answer,
                )


                if corrected_answer.strip():

                    final_answer = (
                        corrected_answer.strip()
                    )


        # ----------------------------------------------------
        # 5. Response payload
        # ----------------------------------------------------

        return {

            "question": question,

            "answer": final_answer,

            "draft_answer": draft_answer,

            "validation_status": validation_status,

            "vector_evidence": [

                {
                    "chunk_id": item.get(
                        "chunk_id"
                    ),

                    "source_file": item.get(
                        "source_file"
                    ),

                    "score": float(
                        item.get(
                            "score",
                            0.0,
                        )
                    ),

                    "text": _trim_text(
                        item.get(
                            "text",
                            "",
                        ),
                        1800,
                    ),
                }

                for item in vector_results

            ],

            "graph_evidence": [

                {
                    "entities": list(
                        path.get(
                            "entities",
                            []
                        )
                    ),

                    "relationships": list(
                        path.get(
                            "relationships",
                            []
                        )
                    ),

                    "text": path.get(
                        "text",
                        "",
                    ),

                    "score": float(
                        path.get(
                            "score",
                            0.0,
                        )
                    ),

                    "semantic_score": float(
                        path.get(
                            "semantic_score",
                            0.0,
                        )
                    ),
                }

                for path in graph_paths

            ],

            "entity_candidates": [

                {
                    "name": item.get(
                        "name"
                    ),

                    "source_file": item.get(
                        "source_file"
                    ),

                    "score": float(
                        item.get(
                            "score",
                            0.0,
                        )
                    ),
                }

                for item in entities

            ],

            "graph": build_graph_payload(
                graph_paths
            ),
        }


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:

        self.retriever.close()


# ============================================================
# D3 GRAPH PAYLOAD
# ============================================================

def build_graph_payload(
    graph_paths: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Convert retrieved graph paths into D3-friendly nodes
    and links.
    """

    nodes_by_id: dict[
        str,
        dict[str, Any],
    ] = {}


    links: list[
        dict[str, Any]
    ] = []


    seen_links: set[
        tuple[str, str, str]
    ] = set()


    for path in graph_paths:

        entities = [

            str(item)

            for item in path.get(
                "entities",
                [],
            )

            if str(item).strip()

        ]


        relationships = [

            str(item)

            for item in path.get(
                "relationships",
                [],
            )

        ]


        path_score = float(
            path.get(
                "score",
                0.0,
            )
        )


        # ----------------------------------------------------
        # Nodes
        # ----------------------------------------------------

        for entity in entities:

            nodes_by_id.setdefault(
                entity,
                {
                    "id": entity,
                    "label": entity,
                },
            )


        # ----------------------------------------------------
        # Links
        # ----------------------------------------------------

        for index, relationship in enumerate(
            relationships
        ):

            if (
                index + 1
                >= len(entities)
            ):

                break


            source = entities[
                index
            ]


            target = entities[
                index + 1
            ]


            key = (
                source,
                target,
                relationship,
            )


            if key in seen_links:

                continue


            seen_links.add(
                key
            )


            links.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relationship,
                    "score": path_score,
                }
            )


    return {

        "nodes": list(
            nodes_by_id.values()
        ),

        "links": links,

    }