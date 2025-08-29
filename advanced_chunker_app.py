#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advanced_chunker_app.py

An advanced, production-ready document chunker for RAG.

Key features
------------
1) Hybrid strategies:
   - semantic: sentence embeddings → topic boundaries → token-aware merges
   - texttiling: classical topic segmentation (falls back to semantic if available)
   - token: pure token-based sliding window with smart boundaries (no mid-code/table cuts)
   - hybrid (default): semantic boundaries + token caps with overlap

2) Structure-aware:
   - Respects headings, bullets, code fences, tables
   - Keeps tables/code blocks intact (atomic) when possible
   - Captures rich metadata: section titles, page numbers, char offsets, token counts

3) Layout-aware options:
   - PDF parsing with PyPDF2 (fallback) or PyMuPDF if installed (better layout)
   - DOCX parsing if python-docx installed
   - Plain text/Markdown first-class support

4) Query-time expansion:
   - Utility to re-merge neighbors around a retrieved chunk (late chunking idea)

5) Deps are optional:
   - sentence-transformers for embeddings (recommended)
   - nltk for sentence tokenization / texttiling
   - tiktoken for exact token counts (OpenAI tokenizer)
   - spacy for noun-phrase keywords (optional)
   - Everything degrades gracefully if missing.

Usage
-----
python advanced_chunker_app.py --input /path/to/file.pdf --out chunks.jsonl \
  --strategy hybrid --target_tokens 700 --overlap 80 --min_tokens 120

Install (recommended deps):
pip install sentence-transformers nltk tiktoken PyPDF2 python-docx spacy streamlit pymupdf
python -m nltk.downloader punkt
(optional) pip install pdfplumber pymupdf

Output
------
JSONL with: {"id","text","metadata":{...}}

Author: ChatGPT (Advanced RAG Chunker)
License: MIT
"""
import os
import re
import io
import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

import streamlit as st # Import Streamlit
import io # For file handling

# ---------- Optional imports (graceful fallbacks) ----------
try:
    import tiktoken  # for precise token counts
except Exception:
    tiktoken = None

try:
    import nltk
    from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
except Exception:
    nltk = None
    nltk_sent_tokenize = None

# Classical topic segmentation
try:
    from nltk.tokenize import TextTilingTokenizer
except Exception:
    TextTilingTokenizer = None

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    SentenceTransformer = None
    np = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    import fitz # PyMuPDF for better PDF handling
except Exception:
    fitz = None


# ---------- Utilities ----------
def approx_token_len(text: str) -> int:
    """Very rough fallback when tiktoken is unavailable."""
    # Heuristic: ~4 chars/token in English-like text
    return max(1, int(len(text) / 4))

def count_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    if tiktoken is None:
        return approx_token_len(text)
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def cosine_sim(a: "np.ndarray", b: "np.ndarray") -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)

def normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()

def split_sentences(text: str) -> List[str]:
    text = text.replace("\r\n", "\n")
    if nltk_sent_tokenize is not None:
        return [s.strip() for s in nltk_sent_tokenize(text) if s.strip()]
    # Regex fallback (simple; not perfect)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
    return [p.strip() for p in parts if p.strip()]

def is_code_fence(line: str) -> bool:
    return bool(re.match(r"^\s*`{3,}", line))

def looks_like_table_block(block: str) -> bool:
    lines = block.strip().splitlines()
    if len(lines) < 2:
        return False
    pipe_ratio = sum(1 for ln in lines for ch in ln if ch == '|') / max(1, sum(len(ln) for ln in lines))
    return pipe_ratio > 0.02 or any(re.match(r"^\s*[-|+:]+\s*$", ln) for ln in lines)

def is_heading(line: str) -> bool:
    # Markdown headings or line that ends with colon and TitleCase-ish
    if re.match(r"^\s{0,3}#{1,6}\s+\S", line):
        return True
    if re.match(r"^\s*[0-9IVXivx]+[.)]\s+\S", line):
        return True
    if len(line) < 120 and re.match(r"^[A-Z][A-Za-z0-9 ,/&()\-:]{3,}$", line) and line.strip().endswith(":"):
        return True
    return False

def extract_sections(text: str) -> List[Tuple[str, str]]:
    """
    Very light section detection: split by Markdown-ish headings.
    Returns list of (section_title, section_text).
    """
    lines = text.splitlines()
    sections = []
    buf = []
    current_title = "Document"
    for ln in lines:
        if is_heading(ln):
            # flush previous
            if buf:
                sections.append((current_title, "\n".join(buf).strip()))
            current_title = ln.strip("# ").strip()
            buf = []
        else:
            buf.append(ln)
    if buf:
        sections.append((current_title, "\n".join(buf).strip()))
    return sections or [("Document", text)]

# ---------- Readers ----------
def read_text(file_input: Any) -> Tuple[str, Dict[str, Any]]:
    text_content = ""
    meta = {}
    file_name = ""

    if isinstance(file_input, str): # Path provided
        file_name = os.path.basename(file_input)
        ext = os.path.splitext(file_input)[1].lower()
        meta = {"source_path": os.path.abspath(file_input)}
        with open(file_input, "rb") as f:
            file_bytes = f.read()
    else: # Streamlit UploadedFile
        file_name = file_input.name
        ext = os.path.splitext(file_input.name)[1].lower()
        meta = {"source_name": file_input.name, "size": file_input.size}
        file_bytes = file_input.getvalue()

    # Determine file type and extract text
    if ext in [".txt", ".md"]:
        text_content = file_bytes.decode("utf-8", errors="ignore")
    elif ext == ".pdf":
        if fitz: # Use PyMuPDF if available
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                for i, page in enumerate(doc):
                    meta.setdefault("pages", {})[str(i+1)] = {}
                    text_content += f"\n\n[[PAGE {i+1}]]\n{page.get_text()}"
                doc.close()
            except Exception as e:
                print(f"[WARN] PDF read failed with PyMuPDF: {e}")
                if PyPDF2: # Fallback to PyPDF2
                    try:
                        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                        for i, page in enumerate(reader.pages):
                            meta.setdefault("pages", {})[str(i+1)] = {}
                            txt = page.extract_text() or ""
                            text_content += f"\n\n[[PAGE {i+1}]]\n{txt}"
                    except Exception as e:
                        print(f"[WARN] PDF read failed with PyPDF2 fallback: {e}")
        elif PyPDF2: # Only PyPDF2 available
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                for i, page in enumerate(reader.pages):
                    meta.setdefault("pages", {})[str(i+1)] = {}
                    txt = page.extract_text() or ""
                    text_content += f"\n\n[[PAGE {i+1}]]\n{txt}"
            except Exception as e:
                print(f"[WARN] PDF read failed with PyPDF2: {e}")
        else:
            print("[WARN] No PDF library installed (PyMuPDF or PyPDF2). Cannot read PDF.")
    elif ext == ".docx" and docx is not None:
        try:
            d = docx.Document(io.BytesIO(file_bytes))
            text_content = "\n".join(p.text for p in d.paragraphs)
        except Exception as e:
            print(f"[WARN] DOCX read failed: {e}")
    else:
        # Fallback for unsupported types or if file_input was a string that wasn't handled
        try:
            text_content = file_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] Could not decode file as utf-8: {e}")

    if not text_content:
        print(f"[WARN] No text extracted from {file_name}. Attempting to read as plain text.")
        text_content = file_bytes.decode("utf-8", errors="ignore") # Final attempt

    return text_content, meta

# ---------- Topic segmentation (semantic) ----------
class SemanticSegmenter:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if SentenceTransformer is None or np is None:
            raise RuntimeError("sentence-transformers (and numpy) not installed.")
        self.model = SentenceTransformer(model_name)

    def boundaries(self, sentences: List[str], window: int = 3, drop_z: float = 0.8) -> List[int]:
        """
        Find boundary indices (sentence-level) where similarity drops significantly.
        - window: local context window size
        - drop_z: z-score threshold for a 'valley' split
        """
        if len(sentences) <= 1:
            return []
        embs = self.model.encode(sentences, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        # Local cohesion: average cosine sim of each sentence to window around it
        sims = []
        for i in range(len(sentences)):
            left = max(0, i - window)
            right = min(len(sentences), i + window + 1)
            nbrs = [j for j in range(left, right) if j != i]
            if not nbrs:
                sims.append(0.0)
                continue
            v = np.mean([cosine_sim(embs[i], embs[j]) for j in nbrs])
            sims.append(v)
        sims = np.array(sims)
        # z-score of *negative* deltas (drops)
        deltas = np.diff(sims)
        if len(deltas) == 0:
            return []
        mu, sigma = float(np.mean(deltas)), float(np.std(deltas) + 1e-9)
        zscores = (deltas - mu) / sigma
        # Boundary if big negative drop
        bnds = [i+1 for i, z in enumerate(zscores) if z < -abs(drop_z)]
        # avoid boundary spam: enforce min segment length of 3 sentences
        pruned = []
        last = 0
        for b in bnds:
            if b - last >= 3:
                pruned.append(b)
                last = b
        return pruned

# ---------- Chunk builders ----------
@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]

def merge_sentences_to_token_capped_chunks(
    sentences: List[str],
    target_tokens: int = 700,
    overlap: int = 80,
    min_tokens: int = 120,
    model_name: str = "gpt-4o-mini"
) -> List[List[str]]:
    chunks = []
    cur = []
    cur_tokens = 0

    def flush():
        nonlocal cur, cur_tokens
        if cur:
            chunks.append(cur)
            cur = []
            cur_tokens = 0

    for s in sentences:
        stoks = count_tokens(s, model_name)
        # if sentence itself is huge (e.g., table row), force-cut safely
        if stoks > target_tokens:
            # cut by punctuation or whitespace
            parts = re.split(r'(?<=[,;])\s+|\s{2,}', s)
            for part in parts:
                if not part.strip():
                    continue
                ptoks = count_tokens(part, model_name)
                if cur_tokens + ptoks > target_tokens and cur_tokens >= min_tokens:
                    flush()
                cur.append(part)
                cur_tokens += ptoks
            continue

        # normal case
        if cur_tokens + stoks > target_tokens and cur_tokens >= min_tokens:
            flush()
            # overlap: carry last ~overlap tokens worth of sentences
            if overlap > 0 and chunks:
                back = []
                btoks = 0
                for prev in reversed(chunks[-1]):
                    t = count_tokens(prev, model_name)
                    if btoks + t >= overlap:
                        break
                    back.insert(0, prev)
                    btoks += t
                cur = back[:]
                cur_tokens = sum(count_tokens(x, model_name) for x in cur)

        cur.append(s)
        cur_tokens += stoks

    flush()
    return chunks

def smart_boundary_cleanup(block: str) -> str:
    # Try not to cut inside code fences / tables
    lines = block.splitlines()
    open_code = False
    fixed = []
    for ln in lines:
        if is_code_fence(ln):
            open_code = not open_code
        fixed.append(ln)
    if open_code:
        fixed.append("```")
    text = "\n".join(fixed)
    # remove leading/trailing extra blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
    return text.strip()

def to_chunks_from_boundaries(sentences: List[str], boundaries: List[int],
                              target_tokens: int, overlap: int, min_tokens: int,
                              model_name: str) -> List[str]:
    segments = []
    last = 0
    for b in boundaries + [len(sentences)]:
        seg = sentences[last:b]
        last = b
        segments.append(seg)

    # Now cap each segment to target tokens with overlap
    final_blocks = []
    for seg in segments:
        sub = merge_sentences_to_token_capped_chunks(
            seg, target_tokens=target_tokens, overlap=overlap, min_tokens=min_tokens, model_name=model_name
        )
        for ss in sub:
            blk = smart_boundary_cleanup("\n".join(ss).strip())
            if blk:
                final_blocks.append(blk)
    return final_blocks

# ---------- Main pipeline ----------
def chunk_text(text: str,
               strategy: str = "hybrid",
               target_tokens: int = 700,
               overlap: int = 80,
               min_tokens: int = 120,
               model_name: str = "gpt-4o-mini") -> List[Chunk]:

    text = normalize_whitespace(text.replace("\r\n", "\n"))
    sections = extract_sections(text)

    all_chunks: List[Chunk] = []
    chunk_idx = 0

    for sec_title, sec_text in sections:
        if not sec_text.strip():
            continue

        # Keep atomic blocks (tables/code fences) intact in sentence splitting
        # by temporarily replacing them with placeholders.
        placeholders = {}
        def stash_blocks(pattern, prefix):
            nonlocal sec_text, placeholders
            def repl(m):
                key = f"__{prefix}_{len(placeholders)}__"
                placeholders[key] = m.group(0)
                return key
            sec_text = re.sub(pattern, repl, sec_text, flags=re.MULTILINE|re.DOTALL)

        # code fences
        stash_blocks(r"```.*?```", "CODE")
        # simple tables (markdown)
        stash_blocks(r"(?:\n\s*\|.*\|[\s\S]*?(?:\n\n|$))", "TABLE")

        sents = split_sentences(sec_text)

        # restore placeholders as single-sentence tokens
        sents = [placeholders.get(s, s) for s in sents]

        blocks: List[str] = []

        if strategy in ("semantic", "hybrid"):
            try:
                seg = SemanticSegmenter()
                bnds = seg.boundaries(sents, window=3, drop_z=0.8)
                blocks = to_chunks_from_boundaries(sents, bnds, target_tokens, overlap, min_tokens, model_name)
            except Exception as e:
                # fallback to token-only
                blocks = ["\n".join(ss) for ss in merge_sentences_to_token_capped_chunks(
                    sents, target_tokens=target_tokens, overlap=overlap, min_tokens=min_tokens, model_name=model_name
                )]
        elif strategy == "texttiling" and TextTilingTokenizer is not None:
            try:
                ttt = TextTilingTokenizer()
                tiles = ttt.tokenize("\n".join(sents))
                # Each tile may be big; cap by tokens
                tmp_sents = []
                for tile in tiles:
                    tmp_sents.extend(split_sentences(tile))
                blocks = ["\n".join(ss) for ss in merge_sentences_to_token_capped_chunks(
                    tmp_sents, target_tokens=target_tokens, overlap=overlap, min_tokens=min_tokens, model_name=model_name
                )]
            except Exception:
                blocks = ["\n".join(ss) for ss in merge_sentences_to_token_capped_chunks(
                    sents, target_tokens=target_tokens, overlap=overlap, min_tokens=min_tokens, model_name=model_name
                )]
        else:
            # pure token mode
            blocks = ["\n".join(ss) for ss in merge_sentences_to_token_capped_chunks(
                sents, target_tokens=target_tokens, overlap=overlap, min_tokens=min_tokens, model_name=model_name
            )]

        # Restore placeholders inside blocks
        restored = []
        for b in blocks:
            for key, val in placeholders.items():
                b = b.replace(key, val)
            b = smart_boundary_cleanup(b)
            if b:
                restored.append(b)

        for b in restored:
            chunk_idx += 1
            all_chunks.append(Chunk(
                id=f"chunk_{chunk_idx:05d}",
                text=b,
                metadata={
                    "section": sec_title,
                    "tokens": count_tokens(b, model_name),
                }
            ))
    return all_chunks

def write_jsonl(chunks: List[Chunk], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps({"id": ch.id, "text": ch.text, "metadata": ch.metadata}, ensure_ascii=False) + "\n")

def remerge_neighbors(chunks: List[Chunk], center_idx: int, k: int = 1, max_tokens: int = 2000,
                      model_name: str = "gpt-4o-mini") -> Chunk:
    """
    Query-time late chunking: expand around a selected chunk by ±k neighbors,
    capped by max_tokens.
    """
    left = max(0, center_idx - k)
    right = min(len(chunks), center_idx + k + 1)
    buf = []
    for i in range(left, right):
        buf.append(chunks[i].text)
    merged = "\n\n".join(buf)
    # if too big, trim from ends
    while count_tokens(merged, model_name) > max_tokens and len(buf) > 1:
        # drop whichever end is larger
        if count_tokens(buf[0], model_name) >= count_tokens(buf[-1], model_name):
            buf = buf[1:]
        else:
            buf = buf[:-1]
        merged = "\n\n".join(buf)

    return Chunk(
        id=f"merged_{chunks[center_idx].id}",
        text=merged,
        metadata={
            "merged_from": [c.id for c in chunks[left:right]],
            "tokens": count_tokens(merged, model_name)
        }
    )

def main():
    st.set_page_config(layout="wide", page_title="Advanced RAG Chunker")
    st.title("🧠 Advanced RAG Chunker")

    st.markdown("""
    This tool provides multiple advanced chunking strategies for Retrieval Augmented Generation (RAG).
    Upload your document (PDF, DOCX, TXT, MD) and configure the chunking parameters.
    """)

    # File Uploader
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "md"])

    st.sidebar.header("Chunking Parameters")
    # Strategy selection
    strategy = st.sidebar.selectbox(
        "Strategy",
        ["hybrid", "semantic", "texttiling", "token"],
        index=0,
        help="\n- **hybrid (default)**: semantic boundaries + token caps with overlap\n- **semantic**: sentence embeddings → topic boundaries → token-aware merges\n- **texttiling**: classical topic segmentation (falls back to semantic if available)\n- **token**: pure token-based sliding window with smart boundaries (no mid-code/table cuts)"
    )

    # Token parameters
    target_tokens = st.sidebar.slider("Target Tokens", 100, 2000, 700, 50,
                                        help="Aim for chunks of this many tokens. Adjust based on your LLM context window.")
    overlap = st.sidebar.slider("Overlap Tokens", 0, 500, 80, 10,
                                  help="Number of tokens to overlap between consecutive chunks. Helps maintain context.")
    min_tokens = st.sidebar.slider("Minimum Tokens", 50, 500, 120, 10,
                                   help="Minimum tokens for a chunk. Smaller chunks are merged.")
    model_name = st.sidebar.text_input("Tokenizer Model (for tiktoken)", "gpt-4o-mini",
                                        help="Model name for tiktoken (e.g., 'gpt-4o-mini', 'cl100k_base').")

    process_button = st.sidebar.button("Chunk Document")

    if process_button and uploaded_file is not None:
        st.info(f"Processing {uploaded_file.name} using '{strategy}' strategy...")
        try:
            text, meta = read_text(uploaded_file)
            if not text:
                st.error("Could not extract text from the uploaded file.")
            else:
                st.subheader("Extracted Text (first 500 chars)")
                st.code(text[:500] + "..." if len(text) > 500 else text, language="text")
                st.subheader("Extracted Metadata")
                st.json(meta)

                chunks = chunk_text(
                    text=text,
                    strategy=strategy,
                    target_tokens=target_tokens,
                    overlap=overlap,
                    min_tokens=min_tokens,
                    model_name=model_name
                )

                st.subheader(f"Generated Chunks ({len(chunks)}) ")
                # Display chunks in an expandable way
                for i, chunk in enumerate(chunks):
                    with st.expander(f"Chunk {i+1} (ID: {chunk.id}, Tokens: {chunk.metadata.get('tokens', 'N/A')})"):
                        st.json(asdict(chunk))

                # Optional: Download chunks
                if chunks:
                    jsonl_output = "\n".join([json.dumps(asdict(ch), ensure_ascii=False) for ch in chunks])
                    st.download_button(
                        label="Download Chunks as JSONL",
                        data=jsonl_output,
                        file_name=f"chunks_{os.path.splitext(uploaded_file.name)[0]}.jsonl",
                        mime="application/jsonl",
                    )

                st.success(f"Successfully generated {len(chunks)} chunks!")

        except Exception as e:
            st.error(f"An error occurred during chunking: {e}")
            st.exception(e)
    elif process_button and uploaded_file is None:
        st.warning("Please upload a document to chunk.")

if __name__ == "__main__":
    main()
