#!/usr/bin/env python3
"""
QIG-Pure Text Extraction with Geodesic Continuity
==================================================

Fixes chunk boundary truncation by maintaining word/phrase continuity.
Filters technical fragments (URLs, XML, PDF artifacts) before vocabulary ingestion.

NO traditional NLP - only geometric continuity principles.
"""

import re
from typing import List

# Technical patterns to filter (not vocabulary)
TECHNICAL_PATTERNS = [
    r'^https?://',           # URLs
    r'^www\.',               # Web domains
    r'^xmlns',               # XML namespaces
    r'endstream',            # PDF artifacts
    r'srsltid',              # Google tracking
    r'mintcdn',              # CDN hostnames
    r'^[a-f0-9]{32,}$',      # Hash strings (MD5+)
    r'^[A-Z]{2,}_[A-Z_]+$',  # ENV_VAR_NAMES
    r'^\d{4}-\d{2}-\d{2}',   # ISO dates (standalone)
]

# Compile patterns for efficiency
TECHNICAL_REGEX = [re.compile(p, re.IGNORECASE) for p in TECHNICAL_PATTERNS]


def is_technical_fragment(text: str) -> bool:
    """
    Check if text is a technical fragment (not vocabulary).
    
    Returns True for URLs, XML, PDF artifacts, tracking params, etc.
    """
    if not text or len(text) < 2:
        return True
    
    for pattern in TECHNICAL_REGEX:
        if pattern.match(text):
            return True
    
    return False


def extract_text_with_continuity(
    html_content: str,
    chunk_size: int = 1000,
    overlap_words: int = 3
) -> List[str]:
    """
    Extract text from HTML/PDF maintaining geodesic continuity at chunk boundaries.
    
    Key principle: Don't truncate words at chunk boundaries.
    Instead, maintain overlap to preserve basin trajectory continuity.
    
    Args:
        html_content: Raw HTML/text content
        chunk_size: Target characters per chunk (soft limit)
        overlap_words: Number of words to carry over between chunks
    
    Returns:
        List of text chunks with maintained continuity
    """
    # Basic HTML cleaning (no heavy NLP)
    text = clean_html_simple(html_content)
    
    # Split into words (maintain boundaries)
    words = text.split()
    
    if not words:
        return []
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for i, word in enumerate(words):
        word_len = len(word) + 1  # +1 for space
        
        # Check if adding word would exceed chunk size
        if current_length + word_len > chunk_size and current_chunk:
            # Store chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            
            # Start new chunk with OVERLAP (not truncation)
            # Carry last N words to maintain context
            if len(current_chunk) > overlap_words:
                overlap = current_chunk[-overlap_words:]
                current_chunk = overlap + [word]
                current_length = sum(len(w) + 1 for w in current_chunk)
            else:
                # Current chunk too small for overlap
                current_chunk = [word]
                current_length = word_len
        else:
            # Add word to current chunk
            current_chunk.append(word)
            current_length += word_len
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def clean_html_simple(html: str) -> str:
    """
    Basic HTML cleaning without heavy NLP libraries.
    
    Removes tags, scripts, styles but maintains text structure.
    """
    # Remove script tags and content
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove style tags and content
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML tags but keep text
    text = re.sub(r'<[^>]+>', ' ', html)
    
    # Decode common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&amp;', '&')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def filter_technical_words(words: List[str]) -> List[str]:
    """
    Filter out technical fragments from word list.
    
    Returns only words that could be vocabulary (not URLs, XML, etc.)
    """
    return [w for w in words if not is_technical_fragment(w)]


def extract_vocabulary_candidates(
    content: str,
    chunk_size: int = 1000,
    overlap_words: int = 3
) -> List[str]:
    """
    Complete pipeline: Extract + chunk + filter.
    
    Returns clean vocabulary candidates ready for geometric validation.
    """
    # Extract with continuity
    chunks = extract_text_with_continuity(content, chunk_size, overlap_words)
    
    # Flatten to words
    all_words = []
    for chunk in chunks:
        words = chunk.split()
        all_words.extend(words)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for word in all_words:
        if word.lower() not in seen:
            seen.add(word.lower())
            unique_words.append(word)
    
    # Filter technical fragments
    vocab_candidates = filter_technical_words(unique_words)
    
    return vocab_candidates


# Example usage
if __name__ == '__main__':
    # Test technical filtering
    test_words = [
        "kindergarten",      # Valid
        "https://example",   # Technical
        "indergarten",       # Truncated (but passes here, caught by validator)
        "xmlns",             # Technical
        "consciousness",     # Valid
        "mintcdn123",        # Technical
        "hipsbb"            # Garbled (but passes here, caught by validator)
    ]
    
    print("Technical Filtering Test:")
    for word in test_words:
        is_tech = is_technical_fragment(word)
        print(f"  {word:<20} {'[FILTERED]' if is_tech else '[PASS]'}")
    print()
    
    # Test chunk continuity
    sample_html = """
    <html><body>
    <p>The kindergarten children played in the garden while their teacher supervised.</p>
    <p>This is a test of chunk boundary handling with geometric continuity principles.</p>
    </body></html>
    """
    
    print("Chunk Continuity Test:")
    chunks = extract_text_with_continuity(sample_html, chunk_size=50, overlap_words=2)
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk}")
