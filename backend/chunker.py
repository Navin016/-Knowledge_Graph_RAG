"""
backend/chunker.py

Phase 2:
Split extracted PDF text into overlapping chunks.

Two approaches are provided:

1. chunk_text_smart()
   -------------------
   REAL DEFAULT FOR THE PROJECT.

   Uses LangChain's RecursiveCharacterTextSplitter.
   It tries to preserve natural text boundaries:

       Paragraph
           ↓
       Line break
           ↓
       Sentence
           ↓
       Word
           ↓
       Character

   This generally produces better chunks for the Gemini
   knowledge-graph extraction stage.

2. chunk_text()
   -------------
   Simple character-based fallback.

   This version is useful for:
       - understanding how chunking works
       - debugging
       - comparing against the smart splitter
       - experimental evaluation

Later, both approaches can be evaluated using:
    Precision
    Recall
    F1 score

and the better-performing approach can be selected.
"""

from pathlib import Path

from backend.pdf_processor import extract_text


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEFAULT_CHUNK_SIZE = 2500
DEFAULT_OVERLAP = 300


# ---------------------------------------------------------
# Smart Chunking - REAL PROJECT DEFAULT
# ---------------------------------------------------------

def chunk_text_smart(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """
    Split text using LangChain's
    RecursiveCharacterTextSplitter.

    The splitter tries separators in this order:

        1. Paragraph breaks
        2. Line breaks
        3. Sentence endings
        4. Spaces
        5. Individual characters

    This helps avoid cutting sentences in the middle.

    Args:
        text:
            Full text extracted from the PDF.

        chunk_size:
            Maximum number of characters per chunk.

        overlap:
            Number of characters shared between
            consecutive chunks.

    Returns:
        List of overlapping text chunks.

    Raises:
        ValueError:
            If chunk_size or overlap is invalid.
    """

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    if not text or not text.strip():
        return []

    # -----------------------------------------------------
    # Import LangChain splitter
    # -----------------------------------------------------

    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter
    )

    # -----------------------------------------------------
    # Create splitter
    # -----------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,

        # Try natural boundaries first.
        separators=[
            "\n\n",   # Paragraph
            "\n",     # Line
            ". ",     # Sentence
            "! ",     # Sentence
            "? ",     # Sentence
            " ",      # Word
            "",       # Character fallback
        ],

        # Keep the separator with the preceding text
        # where possible.
        keep_separator=True,
    )

    # -----------------------------------------------------
    # Split text
    # -----------------------------------------------------

    chunks = splitter.split_text(text)

    return chunks


# ---------------------------------------------------------
# Simple Character-Based Chunking - FALLBACK
# ---------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """
    Split text into fixed-size overlapping chunks.

    This is the dependency-free fallback implementation.

    Example:

        chunk_size = 1000
        overlap = 150

        Chunk 1:
        0 -------------------- 999

        Chunk 2:
        850 ------------------- 1849

        Therefore, 150 characters are shared.

    Args:
        text:
            Full text extracted from the PDF.

        chunk_size:
            Maximum number of characters per chunk.

        overlap:
            Number of overlapping characters.

    Returns:
        List of text chunks.

    Raises:
        ValueError:
            If chunk_size or overlap is invalid.
    """

    # -----------------------------------------------------
    # Validate chunk size
    # -----------------------------------------------------

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    # -----------------------------------------------------
    # Validate overlap
    # -----------------------------------------------------

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    # -----------------------------------------------------
    # Handle empty text
    # -----------------------------------------------------

    if not text or not text.strip():
        return []

    chunks = []

    start = 0
    text_length = len(text)

    # Number of new characters added per chunk.
    step = chunk_size - overlap

    # -----------------------------------------------------
    # Create chunks
    # -----------------------------------------------------

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += step

    return chunks


# ---------------------------------------------------------
# Manual Test
# ---------------------------------------------------------

if __name__ == "__main__":

    import sys

    # -----------------------------------------------------
    # PDF path
    # -----------------------------------------------------

    if len(sys.argv) > 1:

        pdf_path = Path(sys.argv[1])

    else:

        pdf_path = Path(
            "data/pdfs/sample.pdf"
        )

    # -----------------------------------------------------
    # Check PDF
    # -----------------------------------------------------

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    # -----------------------------------------------------
    # Phase 1:
    # PDF → Text
    # -----------------------------------------------------

    print("\nExtracting PDF text...")

    extracted_text = extract_text(
        pdf_path
    )

    print(
        f"Extracted characters: "
        f"{len(extracted_text)}"
    )

    # -----------------------------------------------------
    # Phase 2:
    # Text → Smart Chunks
    # -----------------------------------------------------

    print(
        "\nCreating smart chunks..."
    )

    chunks = chunk_text_smart(
        extracted_text
    )

    # -----------------------------------------------------
    # Display information
    # -----------------------------------------------------

    print(
        f"\nNumber of chunks: {len(chunks)}"
    )

    print(
        f"Chunk size: "
        f"{DEFAULT_CHUNK_SIZE} characters"
    )

    print(
        f"Overlap: "
        f"{DEFAULT_OVERLAP} characters"
    )

    print(
        "\nChunking method: "
        "RecursiveCharacterTextSplitter"
    )

    # -----------------------------------------------------
    # Display chunks
    # -----------------------------------------------------

    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        preview = (
            chunk[:300]
            .replace("\n", " ")
        )

        print("\n" + "=" * 70)

        print(
            f"Chunk {i}"
        )

        print("=" * 70)

        print(
            f"Length: {len(chunk)} characters"
        )

        print(
            f"Preview: {preview}"
        )