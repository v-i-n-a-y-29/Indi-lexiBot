import re
import unicodedata

def normalize_unicode(text: str) -> str:
    """Normalizes unicode characters (e.g. smart quotes, ligatures)."""
    return unicodedata.normalize('NFKC', text)

def remove_standalone_page_numbers(text: str) -> str:
    """Removes standalone page numbers and basic repetitive headers/footers."""
    # Match standalone numbers or "Page X of Y" on their own lines
    text = re.sub(r'^\s*Page\s+\d+\s*(of\s+\d+)?\s*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^\s*-\s*\d+\s*-\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    return text

def fix_hard_wraps(text: str) -> str:
    """
    Joins sentences that have been broken by PDF hard wraps.
    Matches lines ending without a terminal punctuation and followed by a lowercase letter.
    """
    # Join line if it ends in a word char/comma/hyphen and next line starts with lowercase
    text = re.sub(r'([^\.?!;:\n])\n([a-z])', r'\1 \2', text)
    # Handle hyphenated words broken across lines
    text = re.sub(r'([a-zA-Z]+)-\n([a-zA-Z]+)', r'\1\2', text)
    return text

def normalize_whitespace(text: str) -> str:
    """Normalizes spaces and removes repeated blank lines."""
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace 3 or more newlines with 2 newlines (preserve paragraphs)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing whitespace
    return text.strip()

def clean_text(raw_text: str) -> str:
    """Applies all cleaning steps to the raw text extracted from a PDF."""
    if not raw_text:
        return ""
    
    text = normalize_unicode(raw_text)
    text = remove_standalone_page_numbers(text)
    text = fix_hard_wraps(text)
    text = normalize_whitespace(text)
    
    return text