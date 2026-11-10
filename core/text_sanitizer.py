"""
Text Sanitizer for Text-to-Speech (TTS).
Strips Markdown, code blocks, technical syntax, URLs, and formatting artifacts
so that synthesized speech sounds natural and conversational.
"""
import re


def clean_text_for_tts(text: str) -> str:
    """
    Transform raw LLM markdown output into natural spoken text.
    """
    if not text:
        return ""

    s = text

    # Remove code blocks with backticks (replace with verbal mention if needed)
    s = re.sub(r'```(?:[a-zA-Z0-9_\-]+)?\n([\s\S]*?)```', r' [code block omitted] ', s)
    
    # Inline code: remove backticks
    s = re.sub(r'`([^`]+)`', r'\1', s)

    # Markdown links: keep text, drop url [Text](url) -> Text
    s = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', s)

    # HTML tags: remove <tag>...</tag> or <tag/>
    s = re.sub(r'<[^>]*>', '', s)

    # Bold/Italics: ***text***, **text**, *text*, ___text___, __text__, _text_
    s = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', s)
    s = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', s)

    # Headers: # Header -> Header
    s = re.sub(r'^#{1,6}\s*([^\n]+)', r'\1', s, flags=re.MULTILINE)

    # Markdown tables: remove table rows / pipes
    s = re.sub(r'\|[^\n]+\|', '', s)

    # Bullet points / list markers: - item, * item, 1. item -> item
    s = re.sub(r'^\s*[-*•+]\s+', '', s, flags=re.MULTILINE)
    s = re.sub(r'^\s*\d+\.\s+', '', s, flags=re.MULTILINE)

    # Blockquotes: > quote -> quote
    s = re.sub(r'^\s*>\s+', '', s, flags=re.MULTILINE)

    # URLs: replace http(s)://... with "link"
    s = re.sub(r'https?://\S+', 'link', s)

    # Multiple exclamation/question marks: !??! -> !
    s = re.sub(r'[!]{2,}', '!', s)
    s = re.sub(r'[?]{2,}', '?', s)

    # Multiple dots / ellipses: ... -> .
    s = re.sub(r'\.{2,}', '.', s)

    # Normalize whitespace & newlines
    s = re.sub(r'\n\s*\n', '. ', s)
    s = re.sub(r'\n', ', ', s)
    s = re.sub(r'\s+', ' ', s)

    # Strip leading/trailing whitespace
    s = s.strip()

    # Ensure ending punctuation for TTS prosody
    if s and s[-1] not in '.!?':
        s += '.'

    return s


_META_PREFIXES = (
    "note:",
    "disclaimer:",
    "warning:",
    "based on ",
    "since the ",
    "since i ",
    "it seems",
    "it appears",
    "the user ",
    "let me ",
    "i'll ",
    "i will ",
    "i am ",
    "i'm going to ",
    "here is a possible",
    "this response ",
    "i'll look",
    "i will look",
)

_TRACE_LINE = re.compile(
    r"(?i)^(?:\s*(?:\[[^\]]+\]|"
    r"running |dispatching |spawn_|tool call|using tool|subagent|"
    r"searching |reading |wrote |edited |http request|"
    r"processing audio|active assistant|matched target|"
    r"i(?:'ll| will) (?:look|check|read|search|spawn)|"
    r"heard you\.|the transcript came through).*)"
)


def extract_speakable_text(raw_text: str, max_sentences: int = 3) -> str:
    """
    Keep only the conversational reply for TTS.

    Guaardvark's extractSpeakableText: drop tool traces, agent-team chatter,
    meta preambles, then cap to a few spoken sentences.
    """
    if not raw_text or not str(raw_text).strip():
        return "Done."

    text = str(raw_text)
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"/home/\S+", "", text)

    kept_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _TRACE_LINE.match(stripped):
            continue
        lower = stripped.lower()
        if any(lower.startswith(p) for p in _META_PREFIXES):
            continue
        kept_lines.append(stripped)

    text = " ".join(kept_lines) if kept_lines else text
    text = clean_text_for_tts(text)
    sentences = split_into_sentences(text) or ([text] if text else [])
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
    tts_text = " ".join(sentences).strip()
    # Short answers ("4.", "Yes.", "No.") are answers; only text with nothing
    # speakable in it becomes "Done."
    if not re.search(r"[A-Za-z0-9]", tts_text):
        return "Done."
    return tts_text


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into clean sentences for sentence-level streaming TTS.
    """
    if not text:
        return []
    
    # Split on sentence boundaries (. ! ?) while preserving punctuation
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    
    for sent in raw_sentences:
        s = clean_text_for_tts(sent)
        if s and len(s) > 1:
            cleaned.append(s)
            
    return cleaned
