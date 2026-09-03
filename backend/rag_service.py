"""Runtime KG-RAG question answering service.

Keeps the existing retrieval architecture unchanged:
    question -> UnifiedRetriever -> vector + graph evidence -> Gemini answer
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.unified_retriever import UnifiedRetriever


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


ANSWER_MODEL = os.getenv(
    "GEMINI_ANSWER_MODEL",
    "gemini-3.5-flash-lite",
)


def extract_response_text(response: Any) -> str:
    """Convert LangChain/AFC response content into plain text."""
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        text_parts.append(str(text))
                elif "text" in block:
                    text = block.get("text", "")
                    if text:
                        text_parts.append(str(text))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts).strip()

    return str(content).strip()


def _trim_text(value: Any, limit: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


class RAGService:
    """Reusable query-time service for FastAPI."""

    def __init__(
        self,
        vector_k: int = 5,
        entity_k: int = 10,
        graph_k: int = 5,
    ) -> None:
        self.retriever = UnifiedRetriever(
            vector_k=vector_k,
            entity_k=entity_k,
            graph_k=graph_k,
        )

        self.llm = ChatGoogleGenerativeAI(
            model=ANSWER_MODEL,
            temperature=0,
        )

    def answer(self, question: str) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        retrieval = self.retriever.retrieve(question)
        vector_results = retrieval.get("vector_results", [])
        graph_results = retrieval.get("graph_results", {})
        graph_paths = graph_results.get("graph_paths", [])
        entities = graph_results.get("entities", [])

        vector_context_parts: list[str] = []
        for index, item in enumerate(vector_results, start=1):
            vector_context_parts.append(
                "VECTOR EVIDENCE {index}\n"
                "chunk_id: {chunk_id}\n"
                "source_file: {source_file}\n"
                "score: {score:.4f}\n"
                "text:\n{text}".format(
                    index=index,
                    chunk_id=item.get("chunk_id", ""),
                    source_file=item.get("source_file", ""),
                    score=float(item.get("score", 0.0)),
                    text=_trim_text(item.get("text", "")),
                )
            )

        graph_context_parts: list[str] = []
        for index, path in enumerate(graph_paths, start=1):
            graph_context_parts.append(
                "GRAPH PATH {index}\n"
                "score: {score:.4f}\n"
                "semantic_score: {semantic_score:.4f}\n"
                "path: {text}".format(
                    index=index,
                    score=float(path.get("score", 0.0)),
                    semantic_score=float(path.get("semantic_score", 0.0)),
                    text=_trim_text(path.get("text", ""), 1000),
                )
            )

        vector_context = "\n\n".join(vector_context_parts) or "No vector evidence retrieved."
        graph_context = "\n\n".join(graph_context_parts) or "No graph evidence retrieved."

        prompt = f"""
You are the answer-generation component of a KG-RAG system.
Answer the user's question using ONLY the retrieved evidence below.
Do not invent facts that are not supported by the evidence.
Prefer concise, technically accurate explanations.
When evidence is insufficient, say that the retrieved document evidence is insufficient.

USER QUESTION:
{question}

VECTOR EVIDENCE:
{vector_context}

GRAPH EVIDENCE:
{graph_context}

Write the final answer in normal readable text.
""".strip()

        response = self.llm.invoke([HumanMessage(content=prompt)])
        answer_text = extract_response_text(response)

        return {
            "question": question,
            "answer": answer_text,
            "vector_evidence": [
                {
                    "chunk_id": item.get("chunk_id"),
                    "source_file": item.get("source_file"),
                    "score": float(item.get("score", 0.0)),
                    "text": _trim_text(item.get("text", ""), 1800),
                }
                for item in vector_results
            ],
            "graph_evidence": [
                {
                    "entities": list(path.get("entities", [])),
                    "relationships": list(path.get("relationships", [])),
                    "text": path.get("text", ""),
                    "score": float(path.get("score", 0.0)),
                    "semantic_score": float(path.get("semantic_score", 0.0)),
                }
                for path in graph_paths
            ],
            "entity_candidates": [
                {
                    "name": item.get("name"),
                    "source_file": item.get("source_file"),
                    "score": float(item.get("score", 0.0)),
                }
                for item in entities
            ],
            "graph": build_graph_payload(graph_paths),
        }

    def close(self) -> None:
        self.retriever.close()


def build_graph_payload(graph_paths: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Convert retrieved graph paths into D3-friendly nodes and links."""
    nodes_by_id: dict[str, dict[str, Any]] = {}
    links: list[dict[str, Any]] = []
    seen_links: set[tuple[str, str, str]] = set()

    for path in graph_paths:
        entities = [str(item) for item in path.get("entities", []) if str(item).strip()]
        relationships = [str(item) for item in path.get("relationships", [])]
        path_score = float(path.get("score", 0.0))

        for entity in entities:
            nodes_by_id.setdefault(
                entity,
                {
                    "id": entity,
                    "label": entity,
                },
            )

        for index, relationship in enumerate(relationships):
            if index + 1 >= len(entities):
                break

            source = entities[index]
            target = entities[index + 1]
            key = (source, target, relationship)

            if key in seen_links:
                continue

            seen_links.add(key)
            links.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relationship,
                    "score": path_score,
                }
            )

    return {
        "nodes": list(nodes_by_id.values()),
        "links": links,
    }
