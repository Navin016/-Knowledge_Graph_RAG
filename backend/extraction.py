"""
backend/extraction.py

Phase 3:
Extract subject-relation-object triples from text
using Gemini 3.6 Flash.

Pipeline:

    Text chunk
        ↓
    Gemini 3.6 Flash
        ↓
    Structured JSON
        ↓
    Pydantic validation
        ↓
    Raw Triple objects
        ↓
    Relation normalization happens separately
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field


# =========================================================
# Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found in .env"
    )

GEMINI_MODEL = "gemini-3.6-flash"


# =========================================================
# Gemini Client
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# Pydantic Schemas
# =========================================================

class Triple(BaseModel):
    """
    Represents one raw knowledge-graph relationship.

    The relation is intentionally a string rather than
    a Literal because the system should support different
    types of unstructured documents.
    """

    subject: str = Field(
        description=(
            "A meaningful entity, person, organization, "
            "place, object, concept, or other identifiable "
            "thing involved in the relationship."
        )
    )

    relation: str = Field(
        description=(
            "The semantic relationship between the subject "
            "and object. Use a concise relation name that "
            "accurately represents the relationship expressed "
            "in the text."
        )
    )

    object: str = Field(
        description=(
            "A meaningful entity, person, organization, "
            "place, object, concept, value, or other identifiable "
            "thing related to the subject."
        )
    )


class ExtractionResult(BaseModel):
    """
    Represents all triples extracted from one text chunk.
    """

    triples: list[Triple]


# =========================================================
# Extraction Instructions
# =========================================================

SYSTEM_INSTRUCTION = """
You are an information extraction system for a general-purpose
knowledge graph.

Your task is to extract factual subject-relation-object triples
from the provided text.

Each triple must contain:

1. subject
2. relation
3. object


GENERAL RULES
-------------

- Extract only information explicitly supported by the text.
- Do not invent facts.
- Do not use information from outside the text.
- Do not make assumptions that are not stated or clearly implied.
- Extract meaningful entities and concepts.
- Keep subject and object names concise and understandable.
- Preserve important entity names and terminology from the text.
- Do not include explanations outside the structured output.


RELATION RULES
--------------

- The relation should describe the actual semantic relationship
  expressed by the text.
- Use a short, meaningful relation name.
- Prefer lowercase snake_case relation names.
- Do not use complete sentences as relation names.
- Do not add unnecessary words.
- Do not force a relationship into a predefined category when
  the text expresses a different relationship.

Examples:

    "Microsoft acquired GitHub."
        Microsoft --[acquired]--> GitHub

    "Alice works at Google."
        Alice --[works_at]--> Google

    "Paris is located in France."
        Paris --[located_in]--> France

    "Pydantic is used for data validation."
        Pydantic --[used_for]--> data validation

    "Drug A treats Disease B."
        Drug A --[treats]--> Disease B

    "Company A manufactures Product B."
        Company A --[manufactures]--> Product B


RELATION CONSISTENCY
--------------------

When the same relationship is expressed multiple times within
the same text, use the same concise relation name where possible.

For example:

    "bought"
    "purchased"

may represent the same semantic relationship, but do not
invent information that the text does not support.

Relation normalization into the project's canonical vocabulary
will be performed in a separate processing stage.


EXTRACTION QUALITY
------------------

- Prefer meaningful factual relationships.
- Avoid extracting trivial grammatical relationships.
- Avoid duplicate triples within the same chunk.
- Do not create a triple simply because two entities appear
  near each other.
- If no meaningful relationships are present, return:

    {
        "triples": []
    }
"""


# =========================================================
# Extract Triples
# =========================================================

def extract_triples(
    text: str,
) -> list[Triple]:
    """
    Extract raw knowledge-graph triples from one text chunk.

    Args:
        text:
            A text chunk extracted from a PDF.

    Returns:
        List of validated Triple objects.
    """

    if not text or not text.strip():
        return []

    prompt = f"""
{SYSTEM_INSTRUCTION}

TEXT TO ANALYZE
---------------

{text}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ExtractionResult,
        },
    )

    result = ExtractionResult.model_validate_json(
        response.text
    )

    return result.triples