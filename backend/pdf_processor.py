from pathlib import Path

import pdfplumber


def extract_text(pdf_path: str | Path) -> str:
    """
    Extract text from all pages of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Complete extracted text as a single string.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If no extractable text is found.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    pages_text = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_number, page in enumerate(
            pdf.pages,
            start=1
        ):

            page_text = page.extract_text()

            if page_text:
                pages_text.append(page_text)

            else:
                print(
                    f"[pdf_processor] Warning: no text found "
                    f"on page {page_number} of {pdf_path.name}"
                )

    full_text = "\n".join(pages_text)

    if not full_text.strip():
        raise ValueError(
            f"No extractable text found in '{pdf_path.name}'. "
            "This may be a scanned PDF without a text layer. "
            "OCR can be added later if required."
        )

    return full_text


def extract_text_per_page(
    pdf_path: str | Path
) -> list[str]:
    """
    Extract text from each PDF page separately.

    Empty pages are represented by an empty string.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    with pdfplumber.open(pdf_path) as pdf:

        return [
            page.extract_text() or ""
            for page in pdf.pages
        ]


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python pdf_processor.py <path-to-pdf>"
        )
        sys.exit(1)

    pdf_path = sys.argv[1]

    extracted = extract_text(pdf_path)

    print(
        f"Extracted {len(extracted)} characters."
    )

    print("\n--- First 500 characters ---")
    print(extracted[:500])