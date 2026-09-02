"""
backend/extraction.py

Phase 3:
Extract subject-relation-object triples from unstructured text.

Pipeline:

    Text chunk
        ↓
    Gemini 3.5 Flash-Lite
        ↓
    Structured JSON
        ↓
    Pydantic validation
        ↓
    Raw Triple objects
        ↓
    Relation normalization
        ↓
    Entity resolution
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

GEMINI_MODEL = "gemini-3.5-flash-lite"


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

    subject: str = Field(
        description=(
            "A concise entity, object, method, technology, "
            "person, organization, scientific concept, "
            "measurement, material, dataset, model, or other "
            "meaningful concept explicitly supported by the text."
        )
    )

    relation: str = Field(
        description=(
            "A concise semantic relationship expressed by the text. "
            "Prefer lowercase snake_case."
        )
    )

    object: str = Field(
        description=(
            "A concise entity, object, method, technology, "
            "person, organization, scientific concept, "
            "measurement, material, dataset, model, value, "
            "or other meaningful concept explicitly supported "
            "by the text."
        )
    )


class ExtractionResult(BaseModel):
    """
    All triples extracted from one text chunk.
    """

    triples: list[Triple]


# =========================================================
# Extraction Instructions
# =========================================================

SYSTEM_INSTRUCTION = """
You are an expert information extraction system for constructing
a high-quality knowledge graph from arbitrary unstructured documents.

Your task is to extract factual, atomic, and useful
subject-relation-object triples from the provided text.

The extraction objective is:

    HIGH RECALL
    +
    HIGH PRECISION
    +
    ATOMIC REPRESENTATION
    +
    NO HALLUCINATION


============================================================
1. CORE RULE
============================================================

Extract every DISTINCT and MEANINGFUL factual relationship
that is explicitly stated or directly expressed in the text.

Do NOT extract a relationship merely because two entities
appear in the same sentence or paragraph.

Do NOT use outside knowledge.

Do NOT invent facts.

Do NOT guess missing relationships.

However, do NOT be unnecessarily conservative.

If the text explicitly states five independent facts,
extract five triples.

A paragraph may contain many valid relationships.


============================================================
2. ENTITY REPRESENTATION
============================================================

Subjects and objects should be concise, meaningful entities
or concepts.

They may represent:

- person
- organization
- institution
- company
- place
- product
- software
- hardware
- technology
- scientific concept
- method
- algorithm
- model
- dataset
- material
- chemical
- disease
- measurement
- experimental condition
- document
- publication
- application
- component
- meaningful technical concept


Use the shortest expression that preserves the identity
and meaning of the entity.

Examples:

"University of Iowa Technology Institute"

GOOD:
University of Iowa Technology Institute

BAD:
Institute


"Intel Core i7-6770HQ processor"

GOOD:
Intel Core i7-6770HQ processor

BAD:
processor


"Farneback dense flow optical tracking method"

GOOD:
Farneback dense flow optical tracking method

BAD:
method


"pulsed light fluorescence imaging"

GOOD:
pulsed light fluorescence imaging

BAD:
imaging


Do not remove important model names, versions, acronyms,
technical qualifiers, or distinguishing terms.


============================================================
3. ATOMIC ENTITIES
============================================================

Do NOT put an entire action, clause, sentence, or explanation
inside subject or object.

BAD:

"pixel tracking and fluorescence pixel identification
with pulsed light imaging"

BETTER:

pixel tracking
fluorescence pixel identification
pulsed light imaging


BAD:

"synchronizing camera captures with excitation ON and OFF states"

BETTER:

camera captures
excitation ON and OFF states


BAD:

"45-fold reduction of motion artifacts"

BETTER:

motion artifacts

The relationship should carry the meaning:

pixel tracking --[reduces]--> motion artifacts


============================================================
4. CORE
FERENCE RESOLUTION
============================================================

Technical documents frequently use references such as:

    the system
    the method
    the approach
    this method
    this approach
    this study
    this work
    the technique
    the framework
    the model
    it
    they
    these systems
    the proposed system
    the proposed method

When such a reference clearly refers to a previously named
entity in the provided text, resolve the reference to that
named entity.

Example:

"FastAPI is a Python web framework. The framework uses
Pydantic for data validation."

Return:

FastAPI --[is_a]--> Python web framework
FastAPI --[uses]--> Pydantic
Pydantic --[used_for]--> data validation

Do NOT return:

framework --[uses]--> Pydantic


Another example:

"The proposed imaging system uses two CCD cameras.
The system captures fluorescence images."

Return:

imaging system --[uses]--> CCD cameras
imaging system --[captures]--> fluorescence images

NOT:

system --[captures]--> fluorescence images


IMPORTANT:

Only resolve a generic reference when the antecedent is
clear from the provided text.

If the reference cannot be resolved confidently,
DO NOT create the triple.


============================================================
5. GENERIC PLACEHOLDERS
============================================================

Do NOT use the following as entities when they merely refer
to another entity:

    this work
    this study
    this paper
    this article
    this manuscript
    the authors
    authors
    the system
    system
    the method
    method
    the approach
    approach
    proposed method
    proposed approach
    proposed system
    the technique
    technique
    the framework
    framework
    the model
    model
    the algorithm
    algorithm
    the experiment
    experiment
    the application
    application
    the document
    document
    the process
    process

Resolve them to a named entity when the reference is clear.

Otherwise omit the relationship.


============================================================
6. ATOMIC TRIPLE EXTRACTION
============================================================

If one sentence contains multiple factual relationships,
extract them separately.

Example:

"The imaging system uses Python and OpenCV for image
processing and uses a GPU for acceleration."

Extract the independently supported relationships:

imaging system --[uses]--> Python
imaging system --[uses]--> OpenCV
imaging system --[used_for]--> image processing
imaging system --[uses]--> GPU

Do NOT create unsupported relationships.

For example, do NOT automatically create:

GPU --[used_for]--> image processing

unless the text explicitly supports that relationship.


============================================================
7. RELATION FORMAT
============================================================

Use concise semantic relations.

Prefer lowercase snake_case.

Examples:

    is_a
    uses
    used_for
    developed_by
    created_by
    affiliated_with
    contains
    includes
    consists_of
    component_of
    depends_on
    implemented_with
    implemented_on
    applied_to
    tested_on
    evaluated_on
    evaluates
    measures
    improves
    reduces
    increases
    compares_with
    compared_with
    supports
    provides
    enables
    produces
    generates
    located_in
    published_by
    authored_by
    funded_by
    associated_with

Do not force a relationship into this list.

If another concise relation accurately represents the text,
use it.


============================================================
8. RELATION DIRECTION
============================================================

Preserve the direction expressed in the text.

Text:

"FastAPI uses Pydantic."

Return:

FastAPI --[uses]--> Pydantic

Text:

"Pydantic is used by FastAPI."

Return:

Pydantic --[used_by]--> FastAPI

Do not reverse relationships unless the text supports
the reverse direction.


============================================================
9. DEFINITIONS AND CLASSIFICATIONS
============================================================

Definitions are important KG relationships.

Example:

"FastAPI is a Python web framework."

Extract:

FastAPI --[is_a]--> Python web framework


Example:

"Arduino Uno is a microcontroller development board."

Extract:

Arduino Uno --[is_a]--> microcontroller development board


Do not omit simple classification relationships.


============================================================
10. COMPONENTS AND DEPENDENCIES
============================================================

Extract explicit component and dependency relationships.

Example:

"The system consists of a camera, light source, and computer."

Extract:

system --[includes]--> camera
system --[includes]--> light source
system --[includes]--> computer

If "system" can be resolved to a named entity, use the
named entity instead.


Example:

"FastAPI is built on Starlette and uses Pydantic."

Extract:

FastAPI --[built_on]--> Starlette
FastAPI --[uses]--> Pydantic


============================================================
11. METHODS AND APPLICATIONS
============================================================

Extract relationships involving methods, algorithms,
technologies, and applications.

Example:

"Farneback optical flow is used for pixel tracking."

Extract:

Farneback optical flow --[used_for]--> pixel tracking


Example:

"Fluorescence imaging is used in surgical oncology."

Extract:

fluorescence imaging --[used_for]--> surgical oncology


============================================================
12. COMPARISONS
============================================================

Extract explicit comparisons.

Example:

"Pulsed light imaging was compared with conventional
steady-state fluorescence imaging."

Extract:

pulsed light imaging --[compared_with]--> conventional
steady-state fluorescence imaging


Do not turn a comparison into a causal relationship unless
the text explicitly states causality.


============================================================
13. CAUSALITY
============================================================

Be careful with causal relations.

Do NOT assume:

    A appears before B
    therefore A causes B

Only use relations such as:

    causes
    leads_to
    results_in
    reduces
    improves
    increases

when the text explicitly expresses that relationship.


For example:

BAD:

pulsed light imaging --[causes]--> motion artifacts

if the text only states that motion artifacts can arise
during frame subtraction.

Prefer a relation that accurately reflects the text, such as:

frame subtraction --[can_introduce]--> motion artifacts

only when that relationship is explicitly supported.


============================================================
14. MEASUREMENTS AND RESULTS
============================================================

Preserve important quantitative findings.

Example:

"Pixel tracking reduced motion artifacts from 98.60% to 1.54%."

Extract:

pixel tracking --[reduces]--> motion artifacts

If the numerical result is explicitly important, it may also
be represented as:

motion artifacts --[reduced_from]--> 98.60%
motion artifacts --[reduced_to]--> 1.54%

Do not create triples for every isolated number.

Numbers should only be represented when they carry meaningful
factual information.


============================================================
15. ACRONYMS AND ALIASES
============================================================

When an acronym is explicitly defined:

"Retrieval-Augmented Generation (RAG)"

preserve the meaningful entity:

Retrieval-Augmented Generation

and preserve the acronym:

RAG

Do not assume that semantically similar names are aliases.

Alias resolution is handled by the downstream entity
resolution stage.


============================================================
16. TABLES AND FIGURE CAPTIONS
============================================================

Extract meaningful factual relationships explicitly stated
in tables and figure captions when the text provides enough
context.

Do not infer facts from visual layout alone.


============================================================
17. REFERENCES
============================================================

If text is clearly a bibliography/reference list, do not
invent relationships between cited works.

For example:

"[1] Smith et al., Some Paper"

should not automatically produce:

Smith --[authored]--> Some Paper

unless the reference text itself explicitly provides enough
information to support that relationship.


============================================================
18. DUPLICATES
============================================================

Do not repeat an identical triple within the same chunk.

However, retain distinct relationships involving the same
entity.

For example:

FastAPI --[uses]--> Python
FastAPI --[uses]--> Pydantic
FastAPI --[provides]--> API documentation

are three different triples.


Do NOT aggressively remove duplicates across chunks.

The downstream normalization and entity-resolution stages
will handle cross-chunk duplication.


============================================================
19. SELF-CHECK BEFORE OUTPUT
============================================================

Before returning each triple, verify:

1. Is the subject meaningful?
2. Is the object meaningful?
3. Is the relation factual?
4. Is the relationship explicitly supported?
5. Is the direction correct?
6. Is either entity actually a generic reference?
7. Can a generic reference be resolved to a named entity?
8. Is the subject/object unnecessarily long?
9. Is the triple atomic?
10. Did the surrounding sentence contain another
    independent relationship that should also be extracted?


============================================================
20. FINAL OBJECTIVE
============================================================

Maximize the number of CORRECT and USEFUL triples.

Do not maximize triple count blindly.

The desired output is:

    high recall
    +
    factual correctness
    +
    concise entities
    +
    correct relation direction
    +
    atomic relationships
    +
    minimal generic placeholders


Return ONLY the structured JSON response required by the schema.

If no meaningful factual relationships exist:

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
            Text chunk extracted from a PDF.

    Returns:
        Validated list of Triple objects.
    """

    if not text or not text.strip():
        return []

    prompt = f"""
{SYSTEM_INSTRUCTION}

TEXT TO ANALYZE
===============

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