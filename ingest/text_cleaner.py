"""Text cleaning utilities."""

from langchain_core.documents import Document
from typing import List
import re
from bs4 import BeautifulSoup


# Common boilerplate patterns to remove
BOILERPLATE_PATTERNS = [
    r"skip to (main )?content",
    r"cookie (consent|notice|policy)",
    r"accept (all )?cookies?",
    r"privacy policy",
    r"terms of service",
    r"© \d{4}",
    r"all rights reserved",
    r"follow us on",
    r"share on (facebook|twitter|linkedin)",
    r"subscribe to (our )?newsletter",
    r"click here",
    r"read more",
    r"continue reading",
    r"loading\.\.\.",
    r"javascript (is )?required",
    r"enable javascript",
]


def _remove_html_tags(text: str) -> str:
    """Remove HTML tags from text.

    Args:
        text: Text potentially containing HTML

    Returns:
        Text with HTML tags removed
    """
    # Parse HTML and extract text
    soup = BeautifulSoup(text, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style", "meta", "link", "noscript"]):
        script.decompose()

    # Get text content
    text = soup.get_text(separator=" ", strip=True)
    return text


def _remove_boilerplate(text: str) -> str:
    """Remove common boilerplate text patterns.

    Args:
        text: Text to clean

    Returns:
        Text with boilerplate removed
    """
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line_lower = line.lower().strip()

        # Skip empty lines
        if not line_lower:
            continue

        # Check if line matches boilerplate patterns
        is_boilerplate = False
        for pattern in BOILERPLATE_PATTERNS:
            if re.search(pattern, line_lower):
                is_boilerplate = True
                break

        # Skip very short lines that are likely navigation/UI elements
        if len(line_lower) < 3:
            continue

        if not is_boilerplate:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _remove_duplicate_sentences(text: str) -> str:
    """Remove duplicate sentences within a document.

    Args:
        text: Text to deduplicate

    Returns:
        Text with duplicate sentences removed
    """
    # Split into sentences (simple approach using periods, exclamation, question marks)
    sentences = re.split(r"[.!?]+\s+", text)

    seen = set()
    unique_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Normalize sentence for comparison (lowercase, remove extra spaces)
        normalized = re.sub(r"\s+", " ", sentence.lower())

        # Skip if we've seen this sentence before
        if (
            normalized not in seen and len(normalized) > 10
        ):  # Ignore very short sentences
            seen.add(normalized)
            unique_sentences.append(sentence)

    return ". ".join(unique_sentences)


def clean_text(documents: List[Document]) -> List[Document]:
    """Clean and normalize text in documents.

    Removes HTML tags, boilerplate content, and duplicate sentences.

    Args:
        documents: List of Document objects

    Returns:
        List of cleaned Document objects
    """
    cleaned = []
    seen_content = set()  # Track duplicate documents across the list

    for doc in documents:
        text = doc.page_content

        # Step 1: Remove HTML tags
        text = _remove_html_tags(text)

        # Step 2: Remove boilerplate
        text = _remove_boilerplate(text)

        # Step 3: Remove duplicate sentences within document
        text = _remove_duplicate_sentences(text)

        # Step 4: Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        # Step 5: Skip empty or very short documents
        if len(text) < 10:
            continue

        # Step 6: Remove duplicate documents (normalize and compare)
        normalized_content = re.sub(r"\s+", " ", text.lower())
        if normalized_content in seen_content:
            continue
        seen_content.add(normalized_content)

        # Update document with cleaned content
        doc.page_content = text
        cleaned.append(doc)

    return cleaned
