import os
import sys
import time
import hashlib
import sqlite3
import argparse
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import pymupdf
from tqdm import tqdm
from dotenv import load_dotenv

# Load file .env jika ada
load_dotenv()

# ==============================================================================
# KONFIGURASI UMUM (Dapat disesuaikan lewat Environment Variable)
# ==============================================================================
PDF_SOURCE_DIR = os.getenv("PDF_SOURCE_DIR", "source_pdfs")
DB_PATH = os.getenv("DB_PATH", "data/rag_metadata.db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "50"))
BATCH_COMMIT_SIZE = int(os.getenv("BATCH_COMMIT_SIZE", "5000"))

INDONESIAN_STOPWORDS = {
    "apa", "apakah", "siapa", "mengapa", "kenapa", "bagaimana", "dimana",
    "kapan", "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk",
    "pada", "adalah", "sebagai", "dengan", "atau", "oleh", "dalam", "bisa",
    "dapat", "jelaskan", "sebutkan", "tentang", "tersebut", "ada"
}

INDEXING_LOCK = threading.Lock()

# ==============================================================================
# DATABASE MANAGEMENT
# ==============================================================================
def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")
    cursor.execute("PRAGMA cache_size = -128000;")
    cursor.execute("PRAGMA temp_store = MEMORY;")
    return conn

def init_db(db_path: str = DB_PATH):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        file_path TEXT UNIQUE,
        checksum TEXT,
        total_pages INTEGER,
        status TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER,
        file_name TEXT,
        page_number INTEGER,
        content TEXT,
        FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
    )
    """)

    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
        chunk_id UNINDEXED,
        file_name,
        page_number UNINDEXED,
        content
    )
    """)

    conn.commit()
    conn.close()

def get_db_stats(db_path: str = DB_PATH) -> Dict[str, int]:
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'Indexed'")
        total_docs = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]
        conn.close()
        return {"total_docs": total_docs, "total_chunks": total_chunks}
    except Exception:
        return {"total_docs": 0, "total_chunks": 0}

# ==============================================================================
# EKSTRAKSI & INDEXING
# ==============================================================================
def get_checksum(filepath: str, chunk_size: int = 65536) -> str:
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if end < text_len:
            last_space = chunk.rfind(" ")
            if last_space > chunk_size * 0.7:
                end = start + last_space
                chunk = text[start:end]
        chunk_clean = chunk.strip()
        if len(chunk_clean) >= MIN_CHUNK_CHARS:
            chunks.append(chunk_clean)
        start = end - overlap
        if start >= text_len - overlap:
            break
    return chunks

def index_pdfs(source_dir: str = PDF_SOURCE_DIR, db_path: str = DB_PATH) -> Dict[str, Any]:
    with INDEXING_LOCK:
        init_db(db_path)
        conn = get_db_connection(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT file_path, checksum FROM documents WHERE status = 'Indexed'")
        processed_records = {row[0]: row[1] for row in cursor.fetchall()}

        path_obj = Path(source_dir)
        if not path_obj.exists():
            conn.close()
            return {"status": "dir_not_found", "message": f"Direktori '{source_dir}' tidak ditemukan."}

        pdf_files = list(path_obj.rglob("*.pdf"))
        if not pdf_files:
            conn.close()
            return {"status": "empty", "message": f"Tidak ada file PDF ditemukan di '{source_dir}'."}

        chunk_buffer = []

        def flush_buffer(buffer):
            if not buffer:
                return
            cursor.execute("SELECT COALESCE(MAX(chunk_id), 0) FROM chunks")
            max_id_before = cursor.fetchone()[0]

            cursor.executemany("""
            INSERT INTO chunks (doc_id, file_name, page_number, content)
            VALUES (?, ?, ?, ?)
            """, buffer)
            
            cursor.execute("""
            INSERT INTO fts_chunks (chunk_id, file_name, page_number, content)
            SELECT chunk_id, file_name, page_number, content
            FROM chunks
            WHERE chunk_id > ?
            """, (max_id_before,))
            
            conn.commit()
            buffer.clear()

        indexed_count = 0
        for pdf_path in tqdm(pdf_files, desc="Memproses PDF"):
            str_path = str(pdf_path)
            checksum = get_checksum(str_path)

            if str_path in processed_records and processed_records[str_path] == checksum:
                continue

            file_name = pdf_path.name
            try:
                doc = pymupdf.open(str_path)
                total_pages = len(doc)

                cursor.execute("""
                INSERT INTO documents (file_name, file_path, checksum, total_pages, status)
                VALUES (?, ?, ?, ?, 'Processing')
                ON CONFLICT(file_path) DO UPDATE SET checksum = ?, total_pages = ?, status = 'Processing'
                """, (file_name, str_path, checksum, total_pages, checksum, total_pages))
                doc_id = cursor.lastrowid

                for page_idx in range(total_pages):
                    page = doc.load_page(page_idx)
                    text = page.get_text("text").strip()
                    if not text:
                        continue

                    page_num = page_idx + 1
                    page_chunks = split_text(text)

                    for chunk in page_chunks:
                        chunk_buffer.append((doc_id, file_name, page_num, chunk))
                        if len(chunk_buffer) >= BATCH_COMMIT_SIZE:
                            flush_buffer(chunk_buffer)

                doc.close()
                cursor.execute("UPDATE documents SET status = 'Indexed' WHERE doc_id = ?", (doc_id,))
                conn.commit()
                indexed_count += 1

            except Exception as e:
                print(f"\n[Gagal Index] {file_name}: {e}")
                cursor.execute("""
                INSERT INTO documents (file_name, file_path, checksum, total_pages, status)
                VALUES (?, ?, ?, 0, 'Error')
                ON CONFLICT(file_path) DO UPDATE SET status = 'Error'
                """, (file_name, str_path, checksum))
                conn.commit()

        flush_buffer(chunk_buffer)
        conn.close()

        stats = get_db_stats(db_path)
        return {
            "status": "success",
            "new_indexed": indexed_count,
            "total_docs": stats["total_docs"],
            "total_chunks": stats["total_chunks"]
        }

# ==============================================================================
# RETRIEVAL & GEMINI LLM RAG
# ==============================================================================
def clean_and_build_fts_query(user_query: str) -> str:
    raw_tokens = "".join([c if c.isalnum() or c.isspace() else " " for c in user_query.lower()]).split()
    meaningful = [t for t in raw_tokens if t not in INDONESIAN_STOPWORDS and len(t) > 2]
    if not meaningful:
        meaningful = [t for t in raw_tokens if len(t) > 2]
    return " OR ".join(meaningful) if meaningful else ""

def retrieve_relevant_chunks(query: str, db_path: str = DB_PATH, top_k: int = 8) -> List[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    fts_query = clean_and_build_fts_query(query)
    if not fts_query:
        conn.close()
        return []
        
    sql = """
    SELECT c.file_name, c.page_number, c.content, bm25(fts_chunks) AS rank
    FROM fts_chunks f
    JOIN chunks c ON f.chunk_id = c.chunk_id
    WHERE fts_chunks MATCH ?
    ORDER BY rank
    LIMIT ?
    """
    
    try:
        cursor.execute(sql, (fts_query, top_k))
        rows = cursor.fetchall()
    except Exception as e:
        print(f"[ERROR SQL] Query database gagal: {e}")
        rows = []
    finally:
        conn.close()
        
    results = []
    for row in rows:
        results.append({
            "file_name": row[0],
            "page_number": row[1],
            "text": row[2].strip(),
            "score": round(row[3], 4)
        })
    return results

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY belum dikonfigurasi di file .env!")
    from google import genai
    return genai.Client(api_key=api_key)

def ask_gemini_rag(query: str, db_path: str = DB_PATH, top_k: int = 8) -> tuple[str, List[Dict[str, Any]]]:
    chunks = retrieve_relevant_chunks(query, db_path=db_path, top_k=top_k)
    if not chunks:
        return "Tidak ditemukan potongan teks yang relevan di database lokal untuk kata kunci tersebut.", []
        
    context_blocks = []
    for idx, c in enumerate(chunks, 1):
        context_blocks.append(
            f"[Sumber {idx}] Dokumen: {c['file_name']} (Halaman: {c['page_number']})\n{c['text']}"
        )
    context_str = "\n\n".join(context_blocks)
    
    prompt = f"""Kamu adalah analis dokumen akademik dan medis yang presisi.
Jawab pertanyaan berikut HANYA berdasarkan KONTEKS DOKUMEN yang tersedia di bawah ini.

Pedoman Jawaban:
1. Utamakan ketepatan faktual. Jangan berasumsi atau mengarang data di luar teks.
2. Jelaskan fakta penting secara runut, padat, dan sertakan rujukan nama file serta nomor halaman.
3. Gunakan formatting Markdown yang rapi (headings, bullet points, dan penekanan tebal).
4. Jika dokumen hanya memuat sebagian informasi, jelaskan fakta yang ada terlebih dahulu dan sebutkan keterbatasannya secara jujur.

KONTEKS DOKUMEN:
{context_str}

PERTANYAAN:
{query}

JAWABAN ANALITIS:"""

    models_to_try = [
        "gemini-flash-latest",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
    ]

    try:
        client = get_gemini_client()
    except Exception as e:
        return f"Konfigurasi API Key bermasalah: {e}", chunks

    last_error = ""
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if response.text:
                return response.text.strip(), chunks
        except Exception as e:
            last_error = str(e)
            time.sleep(1)
            continue
            
    return f"Gagal mendapatkan respon dari Gemini API: {last_error}", chunks

# ==============================================================================
# FASTAPI WEB SERVER & ENTERPRISE UI
# ==============================================================================
HTML_CONTENT = """<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DocuRAG Enterprise &mdash; Intelligent Document Analysis</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-body: #0b0f19;
            --bg-surface: #111827;
            --bg-card: #151f32;
            --bg-input: #0f172a;
            --border: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(255, 255, 255, 0.18);
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --accent-cyan: #38bdf8;
            --accent-glow: rgba(37, 99, 235, 0.16);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --success: #10b981;
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 16px;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif; }
        body { background: var(--bg-body); color: var(--text-primary); display: flex; flex-direction: column; height: 100vh; overflow: hidden; -webkit-font-smoothing: antialiased; }

        /* Header Navigation */
        header {
            background: rgba(17, 24, 39, 0.85);
            backdrop-filter: blur(14px);
            border-bottom: 1px solid var(--border);
            padding: 0.85rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 20;
        }
        .header-brand { display: flex; align-items: center; gap: 12px; }
        .brand-badge {
            width: 36px; height: 36px; border-radius: 9px;
            background: linear-gradient(135deg, #2563eb 0%, #0284c7 100%);
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3); color: #fff;
        }
        .brand-info h1 { font-size: 1.05rem; font-weight: 700; color: #fff; letter-spacing: -0.01em; }
        .brand-info p { font-size: 0.74rem; color: var(--text-muted); font-weight: 500; }

        .header-actions { display: flex; align-items: center; gap: 10px; }
        .status-pill {
            display: flex; align-items: center; gap: 7px;
            background: var(--bg-card); border: 1px solid var(--border);
            padding: 6px 14px; border-radius: 999px; font-size: 0.78rem; color: var(--text-secondary);
        }
        .status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--success); box-shadow: 0 0 8px var(--success); }
        .btn-action {
            display: flex; align-items: center; gap: 7px;
            background: rgba(255, 255, 255, 0.05); color: var(--text-primary);
            border: 1px solid var(--border); padding: 7px 14px;
            border-radius: var(--radius-sm); font-size: 0.8rem; font-weight: 600;
            cursor: pointer; transition: all 0.2s ease;
        }
        .btn-action:hover { background: rgba(255, 255, 255, 0.1); border-color: var(--border-hover); }
        .btn-action:disabled { opacity: 0.5; cursor: not-allowed; }

        .spin { animation: spin 0.9s linear infinite; }
        @keyframes spin { 100% { transform: rotate(360deg); } }

        /* Main Chat Viewport */
        main {
            flex: 1; overflow-y: auto; padding: 2rem 1.5rem;
            display: flex; flex-direction: column; gap: 1.5rem;
            max-width: 880px; margin: 0 auto; width: 100%;
            scroll-behavior: smooth;
        }
        main::-webkit-scrollbar { width: 6px; }
        main::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.12); border-radius: 4px; }

        /* Welcome Blank State */
        .hero-state {
            margin: auto 0; text-align: center;
            display: flex; flex-direction: column; align-items: center; gap: 14px;
            padding: 2.5rem 1rem;
        }
        .hero-icon {
            width: 58px; height: 58px; border-radius: 16px;
            background: var(--accent-glow); border: 1px solid rgba(37, 99, 235, 0.35);
            display: flex; align-items: center; justify-content: center;
            margin-bottom: 4px; color: var(--accent-cyan);
        }
        .hero-state h2 { font-size: 1.45rem; font-weight: 700; color: #fff; letter-spacing: -0.02em; }
        .hero-state p { font-size: 0.92rem; color: var(--text-secondary); max-width: 540px; line-height: 1.55; }
        .suggestions { display: flex; flex-direction: column; gap: 8px; width: 100%; max-width: 600px; margin-top: 1rem; }
        .suggestion-card {
            background: var(--bg-surface); border: 1px solid var(--border);
            padding: 12px 16px; border-radius: var(--radius-md); font-size: 0.87rem; color: var(--text-secondary);
            text-align: left; cursor: pointer; transition: all 0.2s ease;
            display: flex; align-items: center; justify-content: space-between;
        }
        .suggestion-card:hover {
            background: var(--bg-card); border-color: rgba(37, 99, 235, 0.4);
            color: #fff; transform: translateY(-1px);
        }
        .suggestion-card svg { opacity: 0; transition: opacity 0.2s; color: var(--accent-cyan); }
        .suggestion-card:hover svg { opacity: 1; }

        /* Chat Messages */
        .msg-row { display: flex; gap: 14px; width: 100%; }
        .msg-row.user { justify-content: flex-end; }
        .avatar {
            width: 34px; height: 34px; border-radius: 9px; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.9rem; font-weight: 700;
        }
        .msg-row.ai .avatar { background: linear-gradient(135deg, #2563eb, #0284c7); color: #fff; box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25); }
        .msg-row.user .avatar { display: none; }

        .bubble {
            max-width: 85%; padding: 1.15rem 1.4rem;
            border-radius: var(--radius-lg); font-size: 0.94rem; line-height: 1.68;
            position: relative;
        }
        .msg-row.user .bubble {
            background: var(--primary); color: #fff;
            border-bottom-right-radius: 4px; box-shadow: 0 4px 14px rgba(37, 99, 235, 0.22);
        }
        .msg-row.ai .bubble {
            background: var(--bg-surface); border: 1px solid var(--border);
            color: var(--text-primary); border-top-left-radius: 4px; flex: 1;
        }

        /* Typography inside AI bubbles */
        .bubble h1, .bubble h2, .bubble h3, .bubble h4 { margin: 1.1rem 0 0.45rem; font-weight: 700; color: #fff; font-size: 1.06rem; }
        .bubble h1:first-child, .bubble h2:first-child, .bubble h3:first-child { margin-top: 0; }
        .bubble p { margin-bottom: 0.7rem; }
        .bubble p:last-child { margin-bottom: 0; }
        .bubble ul, .bubble ol { margin: 0.5rem 0 0.75rem 1.35rem; }
        .bubble li { margin-bottom: 0.35rem; }
        .bubble strong { color: #fff; font-weight: 600; }
        .bubble code { background: rgba(0, 0, 0, 0.4); padding: 2px 6px; border-radius: 4px; font-size: 0.88em; font-family: monospace; color: #38bdf8; }

        /* Actions on AI response */
        .bubble-actions {
            display: flex; justify-content: flex-end; margin-top: 0.75rem;
            padding-top: 0.6rem; border-top: 1px solid rgba(255, 255, 255, 0.05);
        }
        .btn-copy {
            background: transparent; border: none; color: var(--text-muted);
            font-size: 0.75rem; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 6px;
            padding: 4px 8px; border-radius: 4px; transition: all 0.2s;
        }
        .btn-copy:hover { color: #fff; background: rgba(255, 255, 255, 0.06); }

        /* Citation Section */
        .citation-box {
            margin-top: 1rem; padding-top: 0.85rem;
            border-top: 1px solid var(--border);
        }
        .citation-title {
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
            color: var(--text-muted); font-weight: 700; margin-bottom: 8px;
            display: flex; align-items: center; gap: 6px;
        }
        .citation-tags { display: flex; flex-wrap: wrap; gap: 7px; }
        .citation-tag {
            display: inline-flex; align-items: center; gap: 6px;
            background: var(--bg-card); border: 1px solid var(--border);
            padding: 5px 11px; border-radius: 6px; font-size: 0.78rem; color: #cbd5e1;
            cursor: pointer; transition: all 0.2s ease;
        }
        .citation-tag:hover {
            border-color: rgba(56, 189, 248, 0.4); background: rgba(56, 189, 248, 0.08); color: #fff;
        }
        .citation-page {
            background: rgba(56, 189, 248, 0.15); color: var(--accent-cyan);
            padding: 1px 6px; border-radius: 4px; font-weight: 600; font-size: 0.72rem;
        }

        /* Bottom Floating Bar */
        footer {
            background: rgba(17, 24, 39, 0.92);
            backdrop-filter: blur(14px);
            border-top: 1px solid var(--border);
            padding: 1rem 1.5rem 1.25rem;
        }
        .footer-wrap { max-width: 880px; margin: 0 auto; display: flex; flex-direction: column; gap: 8px; }
        .input-bar {
            background: var(--bg-input); border: 1px solid var(--border);
            border-radius: var(--radius-lg); padding: 6px 8px 6px 18px;
            display: flex; align-items: center; gap: 10px; transition: all 0.2s ease;
        }
        .input-bar:focus-within { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15); }
        .input-field {
            flex: 1; background: transparent; border: none; outline: none;
            color: #fff; font-size: 0.95rem; line-height: 1.4; resize: none; max-height: 120px;
        }
        .input-field::placeholder { color: var(--text-muted); }
        .btn-send {
            background: var(--primary); color: #fff; border: none;
            width: 38px; height: 38px; border-radius: 10px;
            display: flex; align-items: center; justify-content: center;
            cursor: pointer; transition: all 0.2s ease; flex-shrink: 0;
        }
        .btn-send:hover { background: var(--primary-hover); transform: scale(1.03); }
        .btn-send:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

        .helper-text { text-align: center; font-size: 0.73rem; color: var(--text-muted); }

        /* Loader */
        .typing-loader {
            display: none; align-items: center; gap: 8px;
            padding: 9px 15px; border-radius: var(--radius-md); background: var(--bg-surface);
            border: 1px solid var(--border); width: fit-content; font-size: 0.82rem; color: var(--text-secondary);
        }
        .dots { display: inline-flex; gap: 4px; }
        .dot-pulse { width: 5px; height: 5px; border-radius: 50%; background: var(--accent-cyan); animation: pulse 1.2s infinite ease-in-out; }
        .dot-pulse:nth-child(2) { animation-delay: 0.2s; }
        .dot-pulse:nth-child(3) { animation-delay: 0.4s; }
        @keyframes pulse { 0%, 100% { opacity: 0.3; transform: scale(0.8); } 50% { opacity: 1; transform: scale(1.1); } }

        /* Modal for Inspecting Source Excerpt */
        .modal-overlay {
            position: fixed; inset: 0; background: rgba(0, 0, 0, 0.7); backdrop-filter: blur(4px);
            display: none; align-items: center; justify-content: center; z-index: 50; padding: 1.5rem;
        }
        .modal-overlay.open { display: flex; }
        .modal-box {
            background: var(--bg-surface); border: 1px solid var(--border);
            border-radius: var(--radius-lg); width: 100%; max-width: 650px;
            max-height: 80vh; display: flex; flex-direction: column; overflow: hidden;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.6);
        }
        .modal-header {
            padding: 1.1rem 1.4rem; border-bottom: 1px solid var(--border);
            display: flex; justify-content: space-between; align-items: center; background: var(--bg-card);
        }
        .modal-header h3 { font-size: 1rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }
        .modal-close { background: transparent; border: none; color: var(--text-muted); font-size: 1.3rem; cursor: pointer; padding: 2px 6px; }
        .modal-close:hover { color: #fff; }
        .modal-body { padding: 1.4rem; overflow-y: auto; font-size: 0.9rem; line-height: 1.65; color: var(--text-secondary); }
        .modal-body pre { background: var(--bg-input); padding: 1rem; border-radius: var(--radius-sm); border: 1px solid var(--border); white-space: pre-wrap; font-family: inherit; color: #f1f5f9; }

        /* Toast Notifications */
        .toast {
            position: fixed; top: 20px; right: 20px; z-index: 100;
            background: var(--bg-card); border: 1px solid var(--border);
            color: #fff; padding: 12px 18px; border-radius: var(--radius-md);
            box-shadow: 0 12px 24px -4px rgba(0,0,0,0.5); font-size: 0.85rem;
            opacity: 0; transform: translateY(-10px); transition: all 0.3s ease; pointer-events: none;
        }
        .toast.show { opacity: 1; transform: translateY(0); pointer-events: auto; }
    </style>
</head>
<body>
    <header>
        <div class="header-brand">
            <div class="brand-badge">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                </svg>
            </div>
            <div class="brand-info">
                <h1>DocuRAG Enterprise</h1>
                <p>Private Document Intelligence</p>
            </div>
        </div>
        <div class="header-actions">
            <div id="statsBadge" class="status-pill">
                <span class="status-dot"></span>
                <span id="statsText">Menghubungkan...</span>
            </div>
            <button class="btn-action" onclick="resetChat()" title="Mulai obrolan baru">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="12" y1="5" x2="12" y2="19"></line>
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
                <span>Percakapan Baru</span>
            </button>
            <button id="btnReindex" class="btn-action" onclick="triggerIndex()" title="Pindai dokumen PDF baru">
                <svg id="syncSvg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"></path>
                </svg>
                <span>Sinkronisasi Dokumen</span>
            </button>
        </div>
    </header>

    <main id="chatBox">
        <div id="heroState" class="hero-state">
            <div class="hero-icon">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <circle cx="11" cy="14" r="3"></circle>
                    <line x1="13.2" y1="16.2" x2="16" y2="19"></line>
                </svg>
            </div>
            <h2>Asisten Analisis Dokumen</h2>
            <p>Tanyakan informasi apa pun dari 2.500+ dokumen PDF Anda. Jawaban diekstrak secara lokal dan diverifikasi menggunakan AI dengan sitasi halaman transparan.</p>
            
            <div class="suggestions">
                <div class="suggestion-card" onclick="useStarter(this)">
                    <span>Dari semua dokumen yang ada, kebanyakan membahas apa saja?</span>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
                </div>
                <div class="suggestion-card" onclick="useStarter(this)">
                    <span>Faktor-faktor apa saja yang berhubungan dengan preeklampsia pada ibu hamil?</span>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
                </div>
                <div class="suggestion-card" onclick="useStarter(this)">
                    <span>Bagaimana konsep manajemen penyakit berbasis wilayah?</span>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
                </div>
            </div>
        </div>
    </main>

    <footer>
        <div class="footer-wrap">
            <div id="loadingStatus" class="typing-loader">
                <div class="dots">
                    <span class="dot-pulse"></span>
                    <span class="dot-pulse"></span>
                    <span class="dot-pulse"></span>
                </div>
                <span>Mencari rujukan dokumen & merangkum analisis...</span>
            </div>
            <div class="input-bar">
                <input type="text" id="queryInput" class="input-field" placeholder="Ketik pertanyaan terkait isi dokumen di sini... (Enter untuk kirim)" onkeypress="handleKey(event)" autofocus>
                <button id="btnSend" class="btn-send" onclick="sendQuery()" title="Kirim">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="22" y1="2" x2="11" y2="13"></line>
                        <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                    </svg>
                </button>
            </div>
            <div class="helper-text">
                Jawaban disajikan secara faktual berdasarkan database lokal & Google Gemini AI. Klik sitasi untuk melihat kutipan teks asli.
            </div>
        </div>
    </footer>

    <!-- Modal Kutipan Teks Dokumen -->
    <div id="citationModal" class="modal-overlay" onclick="closeModalOnOverlay(event)">
        <div class="modal-box">
            <div class="modal-header">
                <h3>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                    </svg>
                    <span id="modalDocName">Kutipan Dokumen</span>
                </h3>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <div class="modal-body">
                <div style="margin-bottom: 10px; font-size: 0.8rem; color: #38bdf8;" id="modalMeta"></div>
                <pre id="modalContent"></pre>
            </div>
        </div>
    </div>

    <div id="toast" class="toast"></div>

    <script>
        let currentSourcesMap = {};

        function showToast(msg) {
            const toast = document.getElementById('toast');
            toast.innerText = msg;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 3500);
        }

        async function loadStats() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('statsText').innerText = 
                    `${data.total_docs} Dokumen • ${data.total_chunks} Chunks`;
            } catch(e) {
                document.getElementById('statsText').innerText = 'Database siap';
            }
        }

        async function triggerIndex() {
            const btn = document.getElementById('btnReindex');
            const svg = document.getElementById('syncSvg');
            btn.disabled = true;
            if (svg) svg.classList.add('spin');
            showToast('Memulai sinkronisasi folder PDF...');
            try {
                const res = await fetch('/api/index', { method: 'POST' });
                const data = await res.json();
                showToast(data.message || `Sinkronisasi selesai! ${data.new_indexed || 0} file baru diindeks.`);
                loadStats();
            } catch(e) {
                showToast('Gagal menjalankan sinkronisasi: ' + e);
            } finally {
                btn.disabled = false;
                if (svg) svg.classList.remove('spin');
            }
        }

        function resetChat() {
            const chatBox = document.getElementById('chatBox');
            chatBox.innerHTML = `
                <div id="heroState" class="hero-state">
                    <div class="hero-icon">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                            <circle cx="11" cy="14" r="3"></circle>
                            <line x1="13.2" y1="16.2" x2="16" y2="19"></line>
                        </svg>
                    </div>
                    <h2>Asisten Analisis Dokumen</h2>
                    <p>Tanyakan informasi apa pun dari 2.500+ dokumen PDF Anda. Jawaban diekstrak secara lokal dan diverifikasi menggunakan AI dengan sitasi halaman transparan.</p>
                    
                    <div class="suggestions">
                        <div class="suggestion-card" onclick="useStarter(this)">
                            <span>Dari semua dokumen yang ada, kebanyakan membahas apa saja?</span>
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
                        </div>
                        <div class="suggestion-card" onclick="useStarter(this)">
                            <span>Faktor-faktor apa saja yang berhubungan dengan preeklampsia pada ibu hamil?</span>
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
                        </div>
                        <div class="suggestion-card" onclick="useStarter(this)">
                            <span>Bagaimana konsep manajemen penyakit berbasis wilayah?</span>
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
                        </div>
                    </div>
                </div>
            `;
            document.getElementById('queryInput').value = '';
            document.getElementById('queryInput').focus();
        }

        function useStarter(el) {
            const text = el.querySelector('span').innerText;
            document.getElementById('queryInput').value = text;
            sendQuery();
        }

        function handleKey(e) {
            if (e.key === 'Enter') sendQuery();
        }

        async function sendQuery() {
            const input = document.getElementById('queryInput');
            const q = input.value.trim();
            if (!q) return;

            input.value = '';
            const chatBox = document.getElementById('chatBox');
            const loader = document.getElementById('loadingStatus');
            const hero = document.getElementById('heroState');
            if (hero) hero.style.display = 'none';

            // User Message
            const userRow = document.createElement('div');
            userRow.className = 'msg-row user';
            userRow.innerHTML = `<div class="bubble">${escapeHtml(q)}</div>`;
            chatBox.appendChild(userRow);
            chatBox.scrollTop = chatBox.scrollHeight;

            loader.style.display = 'inline-flex';
            document.getElementById('btnSend').disabled = true;

            try {
                const res = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: q, top_k: 8 })
                });
                const data = await res.json();

                // Simpan sitasi untuk modal inspector
                const msgId = 'msg_' + Date.now();
                currentSourcesMap[msgId] = data.sources || [];

                let sourcesHtml = '';
                if (data.sources && data.sources.length > 0) {
                    sourcesHtml = '<div class="citation-box"><div class="citation-title">' +
                        '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg>' +
                        '<span>Rujukan Dokumen Terverifikasi (Klik untuk lihat kutipan asli):</span></div><div class="citation-tags">' +
                        data.sources.slice(0, 5).map((s, idx) => 
                            `<span class="citation-tag" onclick="inspectCitation('${msgId}', ${idx})">` +
                            `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>` +
                            `<span>${escapeHtml(s.file_name)}</span>` +
                            `<span class="citation-page">Hal. ${s.page_number}</span></span>`
                        ).join('') +
                        '</div></div>';
                }

                const parsedAnswer = typeof marked !== 'undefined' ? marked.parse(data.answer) : escapeHtml(data.answer);

                const aiRow = document.createElement('div');
                aiRow.className = 'msg-row ai';
                aiRow.innerHTML = `
                    <div class="avatar">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
                    </div>
                    <div class="bubble">
                        ${parsedAnswer}
                        ${sourcesHtml}
                        <div class="bubble-actions">
                            <button class="btn-copy" onclick="copyToClipboard(this, ${JSON.stringify(data.answer)})">
                                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                                <span>Salin Jawaban</span>
                            </button>
                        </div>
                    </div>
                `;
                chatBox.appendChild(aiRow);
            } catch(e) {
                const errRow = document.createElement('div');
                errRow.className = 'msg-row ai';
                errRow.innerHTML = `
                    <div class="avatar" style="background:#ef4444;">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                    </div>
                    <div class="bubble" style="border-color: rgba(239,68,68,0.3);">
                        <strong>Error:</strong> Gagal mendapatkan respon (${escapeHtml(e.message)}).
                    </div>
                `;
                chatBox.appendChild(errRow);
            } finally {
                loader.style.display = 'none';
                document.getElementById('btnSend').disabled = false;
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        }

        function inspectCitation(msgId, idx) {
            const list = currentSourcesMap[msgId];
            if (!list || !list[idx]) return;
            const item = list[idx];

            document.getElementById('modalDocName').innerText = item.file_name;
            document.getElementById('modalMeta').innerText = `Halaman: ${item.page_number} | Nilai Relevansi (BM25): ${item.score}`;
            document.getElementById('modalContent').innerText = item.text;
            document.getElementById('citationModal').classList.add('open');
        }

        function closeModal() {
            document.getElementById('citationModal').classList.remove('open');
        }

        function closeModalOnOverlay(e) {
            if (e.target.id === 'citationModal') closeModal();
        }

        function copyToClipboard(btn, text) {
            navigator.clipboard.writeText(text).then(() => {
                const oldHtml = btn.innerHTML;
                btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg> <span style="color:#10b981;">Tersalin</span>';
                setTimeout(() => btn.innerHTML = oldHtml, 2000);
            }).catch(() => {
                showToast('Gagal menyalin teks');
            });
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.innerText = text;
            return div.innerHTML;
        }

        window.onload = loadStats;
    </script>
</body>
</html>"""

def create_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    class QueryRequest(BaseModel):
        query: str
        top_k: Optional[int] = 8

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(DB_PATH)
        stats = get_db_stats(DB_PATH)
        print(f"[*] Database siap. Total Dokumen: {stats['total_docs']}, Total Chunks: {stats['total_chunks']}")
        if stats['total_docs'] == 0:
            print("[*] Database masih kosong. Menjalankan indexing awal...")
            index_pdfs(PDF_SOURCE_DIR, DB_PATH)
        yield

    app = FastAPI(title="DocuRAG Enterprise", version="1.1.0", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index_ui():
        return HTMLResponse(content=HTML_CONTENT)

    @app.get("/api/status")
    def get_status():
        stats = get_db_stats(DB_PATH)
        return {
            "status": "ok",
            "pdf_source_dir": PDF_SOURCE_DIR,
            "db_path": DB_PATH,
            "total_docs": stats["total_docs"],
            "total_chunks": stats["total_chunks"]
        }

    @app.post("/api/index")
    def trigger_index():
        result = index_pdfs(PDF_SOURCE_DIR, DB_PATH)
        return result

    @app.post("/api/ask")
    def ask_endpoint(body: QueryRequest):
        if not body.query.strip():
            raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")
        answer, sources = ask_gemini_rag(body.query.strip(), db_path=DB_PATH, top_k=body.top_k or 8)
        return {
            "query": body.query,
            "answer": answer,
            "sources": sources
        }

    return app

# ==============================================================================
# CLI INTERACTIVE RUNNER
# ==============================================================================
def run_cli_interactive():
    init_db(DB_PATH)
    print("=" * 60)
    print("MODE CLI INTERAKTIF - PDF RAG ASSISTANT")
    print("=" * 60)
    stats = get_db_stats(DB_PATH)
    print(f"Status Database: {stats['total_docs']} Dokumen | {stats['total_chunks']} Chunks")
    print("Ketik 'exit' atau 'quit' untuk keluar.")
    print("Ketik 'reindex' untuk memindai ulang folder PDF.\n")

    while True:
        try:
            q = input("\nMasukkan pertanyaan Anda: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSampai jumpa!")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            print("Sampai jumpa!")
            break
        if q.lower() == "reindex":
            print(f"Memindai direktori '{PDF_SOURCE_DIR}'...")
            res = index_pdfs(PDF_SOURCE_DIR, DB_PATH)
            print(f"Selesai! Hasil: {res}")
            continue

        print(f"\nMencari dokumen terkait: '{q}'...")
        jawaban, sources = ask_gemini_rag(q, db_path=DB_PATH, top_k=8)

        print("\n" + "=" * 60)
        print("JAWABAN AI:")
        print("=" * 60)
        print(jawaban)

        if sources:
            print("\n" + "-" * 60)
            print("REFERENSI DOKUMEN YANG DIGUNAKAN:")
            print("-" * 60)
            for idx, s in enumerate(sources[:5], 1):
                print(f"[{idx}] {s['file_name']} (Halaman {s['page_number']})")

# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction RAG Assistant")
    parser.add_argument("--cli", action="store_true", help="Jalankan dalam mode interaktif CLI terminal")
    parser.add_argument("--index", action="store_true", help="Hanya jalankan proses indexing PDF ke database")
    parser.add_argument("--ask", type=str, default=None, help="Ajukan satu pertanyaan langsung")
    parser.add_argument("--server", action="store_true", help="Jalankan server Web UI FastAPI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host untuk server web")
    parser.add_argument("--port", type=int, default=8000, help="Port untuk server web")

    args = parser.parse_args()

    if args.index:
        print(f"Menjalankan indexing pada folder: {PDF_SOURCE_DIR}")
        res = index_pdfs(PDF_SOURCE_DIR, DB_PATH)
        print(f"Hasil Indexing: {res}")
        sys.exit(0)

    if args.ask:
        init_db(DB_PATH)
        print(f"Pertanyaan: {args.ask}")
        ans, srcs = ask_gemini_rag(args.ask, db_path=DB_PATH)
        print("\nJawaban:\n", ans)
        if srcs:
            print("\nSumber Dokumen:")
            for s in srcs[:5]:
                print(f"- {s['file_name']} (Hlm {s['page_number']})")
        sys.exit(0)

    if args.cli:
        run_cli_interactive()
        sys.exit(0)

    # Mode Default (Server Web FastAPI jika uvicorn tersedia, fallback ke CLI)
    try:
        import uvicorn
        print(f"Memulai Web UI Server di http://{args.host}:{args.port} ...")
        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError:
        print("[Info] Library 'uvicorn' atau 'fastapi' belum terinstall.")
        print("[Info] Beralih otomatis ke mode CLI terminal...\n")
        run_cli_interactive()
