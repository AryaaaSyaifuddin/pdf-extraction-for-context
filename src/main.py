import os
import sys
import time
import hashlib
import sqlite3
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

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

# ==============================================================================
# DATABASE MANAGEMENT
# ==============================================================================
def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
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

        # Skip jika sudah pernah diindeks dengan checksum identik
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
    except Exception:
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
3. Jika dokumen hanya memuat sebagian informasi, jelaskan fakta yang ada terlebih dahulu dan sebutkan keterbatasannya secara jujur.

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

    client = get_gemini_client()
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
# FASTAPI WEB SERVER & UI
# ==============================================================================
def create_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    app = FastAPI(title="Local PDF RAG Pipeline", version="1.0.0")

    class QueryRequest(BaseModel):
        query: str
        top_k: Optional[int] = 8

    HTML_CONTENT = """<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF RAG Assistant</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent: #38bdf8;
            --accent-hover: #0284c7;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --code-bg: #090d16;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg-color); color: var(--text-main); display: flex; flex-direction: column; height: 100vh; }
        header { background: var(--card-bg); border-bottom: 1px solid var(--border-color); padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; }
        .logo { font-size: 1.25rem; font-weight: 700; color: var(--accent); display: flex; align-items: center; gap: 8px; }
        .stats-badge { background: #0f172a; border: 1px solid var(--border-color); padding: 6px 12px; border-radius: 999px; font-size: 0.85rem; color: var(--text-muted); }
        .btn-index { background: var(--accent); color: #0f172a; border: none; padding: 6px 14px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: 0.2s; }
        .btn-index:hover { background: var(--accent-hover); color: #fff; }
        main { flex: 1; overflow-y: auto; padding: 2rem; display: flex; flex-direction: column; gap: 1.5rem; max-width: 900px; margin: 0 auto; width: 100%; }
        .message-card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 1.25rem; }
        .message-user { border-left: 4px solid var(--accent); }
        .message-ai { border-left: 4px solid #10b981; }
        .sender-tag { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; margin-bottom: 0.5rem; color: var(--text-muted); }
        .content { line-height: 1.6; white-space: pre-wrap; font-size: 0.95rem; }
        .sources-list { margin-top: 1rem; padding-top: 0.75rem; border-top: 1px dashed var(--border-color); }
        .sources-title { font-size: 0.8rem; font-weight: 600; color: var(--accent); margin-bottom: 0.4rem; }
        .source-pill { display: inline-block; background: #0f172a; border: 1px solid var(--border-color); border-radius: 6px; padding: 4px 8px; font-size: 0.8rem; margin: 3px; color: var(--text-muted); }
        footer { background: var(--card-bg); border-top: 1px solid var(--border-color); padding: 1rem 2rem; }
        .input-group { max-width: 900px; margin: 0 auto; display: flex; gap: 10px; }
        input[type="text"] { flex: 1; background: #0f172a; border: 1px solid var(--border-color); border-radius: 8px; padding: 12px 16px; color: #fff; font-size: 1rem; outline: none; }
        input[type="text"]:focus { border-color: var(--accent); }
        button.btn-send { background: var(--accent); color: #0f172a; border: none; padding: 0 20px; border-radius: 8px; font-weight: 700; font-size: 1rem; cursor: pointer; }
        button.btn-send:hover { background: var(--accent-hover); color: #fff; }
        .loader { display: none; text-align: center; color: var(--text-muted); font-size: 0.9rem; margin-top: 10px; }
    </style>
</head>
<body>
    <header>
        <div class="logo">📄 PDF RAG Assistant</div>
        <div style="display: flex; gap: 10px; align-items: center;">
            <div id="statsBadge" class="stats-badge">Memuat status...</div>
            <button id="btnReindex" class="btn-index" onclick="triggerIndex()">⚡ Re-index PDF</button>
        </div>
    </header>

    <main id="chatBox">
        <div class="message-card message-ai">
            <div class="sender-tag">Sistem</div>
            <div class="content">Halo! Masukkan pertanyaan Anda di bawah untuk mencari informasi dari dokumen PDF yang telah diindeks.</div>
        </div>
    </main>

    <footer>
        <div class="input-group">
            <input type="text" id="queryInput" placeholder="Tanyakan isi dokumen PDF Anda di sini..." onkeypress="handleKey(event)" autofocus>
            <button class="btn-send" onclick="sendQuery()">Kirim</button>
        </div>
        <div id="loadingStatus" class="loader">Sedang mencari konteks dokumen dan memproses jawaban AI...</div>
    </footer>

    <script>
        async function loadStats() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('statsBadge').innerText = 
                    `${data.total_docs} Dokumen | ${data.total_chunks} Potongan Chunks`;
            } catch(e) {
                document.getElementById('statsBadge').innerText = 'Status tidak tersedia';
            }
        }

        async function triggerIndex() {
            const btn = document.getElementById('btnReindex');
            btn.disabled = true;
            btn.innerText = 'Mengindeks...';
            try {
                const res = await fetch('/api/index', { method: 'POST' });
                const data = await res.json();
                alert(data.message || 'Indexing selesai!');
                loadStats();
            } catch(e) {
                alert('Gagal menjalankan indexing: ' + e);
            } finally {
                btn.disabled = false;
                btn.innerText = '⚡ Re-index PDF';
            }
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

            // Tambahkan pesan user
            const userMsg = document.createElement('div');
            userMsg.className = 'message-card message-user';
            userMsg.innerHTML = `<div class="sender-tag">Pertanyaan Anda</div><div class="content">${escapeHtml(q)}</div>`;
            chatBox.appendChild(userMsg);
            chatBox.scrollTop = chatBox.scrollHeight;

            loader.style.display = 'block';

            try {
                const res = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: q, top_k: 8 })
                });
                const data = await res.json();

                let sourcesHtml = '';
                if (data.sources && data.sources.length > 0) {
                    sourcesHtml = '<div class="sources-list"><div class="sources-title">Referensi yang digunakan:</div>' +
                        data.sources.slice(0, 5).map(s => `<span class="source-pill">📌 ${s.file_name} (Halaman ${s.page_number})</span>`).join('') +
                        '</div>';
                }

                const aiMsg = document.createElement('div');
                aiMsg.className = 'message-card message-ai';
                aiMsg.innerHTML = `<div class="sender-tag">Jawaban Gemini RAG</div><div class="content">${escapeHtml(data.answer)}</div>${sourcesHtml}`;
                chatBox.appendChild(aiMsg);
            } catch(e) {
                const errBox = document.createElement('div');
                errBox.className = 'message-card message-ai';
                errBox.innerHTML = `<div class="sender-tag" style="color:#ef4444;">Error</div><div class="content">Gagal mendapatkan respon: ${escapeHtml(e.message)}</div>`;
                chatBox.appendChild(errBox);
            } finally {
                loader.style.display = 'none';
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.innerText = text;
            return div.innerHTML;
        }

        window.onload = loadStats;
    </script>
</body>
</html>
"""

    @app.on_event("startup")
    def on_startup():
        init_db(DB_PATH)
        stats = get_db_stats(DB_PATH)
        print(f"[*] Database siap. Total Dokumen: {stats['total_docs']}, Total Chunks: {stats['total_chunks']}")
        if stats['total_docs'] == 0:
            print("[*] Database masih kosong. Menjalankan indexing awal...")
            index_pdfs(PDF_SOURCE_DIR, DB_PATH)

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
