import os
import re
from typing import List, Dict, Optional, Tuple

INPUT_FILE = "../data/computer_fundamental_clean.txt"
OUTPUT_DIR = "../data/chunks_by_metadata_idea"

TOPIC = "Introduction to ICT"

# Target size of each chunk (tune later if needed)
TARGET_WORDS = 180          # aim ~120–250 words
MIN_WORDS = 80              # don't flush tiny chunks unless it's the last one
OVERLAP_UNITS = 1           # overlap small amount to reduce boundary misses

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Your original structural patterns
UNIT_PATTERN = re.compile(r"^UNIT\s*:\s*(.*)", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"^\d+\.\s*.*")

# Common “noise” patterns found in PDF-to-text dumps
PAGE_MARKER = re.compile(r"^---\s*Page\s*\d+\s*---$", re.IGNORECASE)
PREPARED_BY = re.compile(r"^Prepared\s+By\s*:", re.IGNORECASE)

# Bullet patterns (your file uses "", but handle common variants)
BULLET_LINE = re.compile(r"^\s*(?:[\u2022\u2023\u25E6\u2043\u2219\-\*•])\s+.*")

def word_count(s: str) -> int:
    return len(re.findall(r"\w+", s))

def clean_line(line: str) -> str:
    """Light cleaning to reduce junk that creates empty/low-signal chunks."""
    l = line.strip()
    if not l:
        return ""
    if PAGE_MARKER.match(l):
        return ""
    if PREPARED_BY.match(l):
        return ""
    # Drop standalone page numbers
    if re.fullmatch(r"\d+", l):
        return ""
    # Normalize weird spaces
    l = re.sub(r"\s+", " ", l)
    return l

def split_into_idea_units(lines: List[str]) -> List[str]:
    """
    Split section content into 'idea units'.
    We treat blank lines as paragraph breaks, and also treat bullet lines as separate units.
    """
    units: List[str] = []
    buf: List[str] = []

    def flush_buf():
        nonlocal buf
        if buf:
            text = "\n".join(buf).strip()
            if text:
                units.append(text)
        buf = []

    for raw in lines:
        line = clean_line(raw)
        if not line:
            flush_buf()
            continue

        # If it's a bullet line, flush current paragraph and store bullet as its own unit
        if BULLET_LINE.match(line):
            flush_buf()
            units.append(line)
            continue

        buf.append(line)

    flush_buf()
    return units

def pack_units(units: List[str]) -> List[Tuple[str, int]]:
    """
    Pack idea units into chunks around TARGET_WORDS with optional overlap.
    Returns list of (chunk_text, start_unit_index).
    """
    chunks: List[Tuple[str, int]] = []
    buf: List[str] = []
    buf_words = 0
    start_idx = 0
    i = 0

    while i < len(units):
        u = units[i]
        w = word_count(u)

        if not buf:
            start_idx = i

        # If adding u would exceed target and buffer is big enough, flush
        if buf and (buf_words + w) > TARGET_WORDS and buf_words >= MIN_WORDS:
            chunks.append(("\n\n".join(buf).strip(), start_idx))

            # overlap: keep last OVERLAP_UNITS units
            if OVERLAP_UNITS > 0:
                buf = buf[-OVERLAP_UNITS:]
            else:
                buf = []
            buf_words = sum(word_count(x) for x in buf)
            # don't increment i; try adding current unit again after flush/overlap
            continue

        buf.append(u)
        buf_words += w
        i += 1

    if buf:
        chunks.append(("\n\n".join(buf).strip(), start_idx))

    return chunks

def flush_section(topic: str,
                  unit: Optional[str],
                  section: Optional[str],
                  content_lines: List[str],
                  out_chunks: List[Dict]):
    if not section:
        return
    # Remove very low-signal sections (headers with no body)
    cleaned = [clean_line(x) for x in content_lines]
    cleaned = [x for x in cleaned if x]
    if not cleaned:
        return

    idea_units = split_into_idea_units(cleaned)
    if not idea_units:
        return

    packed = pack_units(idea_units)
    for (chunk_text, start_unit_idx) in packed:
        if not chunk_text.strip():
            continue
        out_chunks.append({
            "topic": topic,
            "unit": unit,
            "section": section,
            "start_unit_idx": start_unit_idx,
            "content": chunk_text.strip()
        })

# ---- Main parse loop (keeps your Unit/Section logic) ----
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_lines = [line.rstrip("\n") for line in f.readlines()]

all_chunks: List[Dict] = []
current_unit: Optional[str] = None
current_section: Optional[str] = None
current_content: List[str] = []

for line in raw_lines:
    unit_match = UNIT_PATTERN.match(line.strip())
    if unit_match:
        # flush any pending section before switching units (defensive)
        flush_section(TOPIC, current_unit, current_section, current_content, all_chunks)
        current_unit = unit_match.group(1).strip()
        current_section = None
        current_content = []
        continue

    if SECTION_PATTERN.match(line.strip()):
        # flush previous section
        flush_section(TOPIC, current_unit, current_section, current_content, all_chunks)
        current_section = line.strip()
        current_content = []
        continue

    if current_section is not None:
        current_content.append(line)

# flush last
flush_section(TOPIC, current_unit, current_section, current_content, all_chunks)

# Write chunk files
for i, ch in enumerate(all_chunks):
    fname = f"chunk_{i:04d}.txt"
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Topic: {ch['topic']}\n")
        if ch["unit"]:
            f.write(f"Unit: {ch['unit']}\n")
        f.write(f"Section: {ch['section']}\n")
        f.write(f"IdeaStartIndex: {ch['start_unit_idx']}\n\n")
        f.write(ch["content"])

print(f"Created {len(all_chunks)} idea-sized chunks in {OUTPUT_DIR}")
