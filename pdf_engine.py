"""PDF 解析 + VectorStore 增量更新引擎（共用模組）"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

import pdfplumber
import pypdfium2 as pdfium
import pytesseract
from tqdm import tqdm
import hashlib
import json
import os
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── 設定 ──────────────────────────────────────────────────
PDF_FOLDER = "./pdfs"
PERSIST_DIR = "./chroma_db"
PROCESSED_LOG = "./processed_files.json"


# ── OCR ───────────────────────────────────────────────────

def ocr_page(pdf_path: str, page_index: int) -> str:
    """用 Tesseract OCR 辨識掃描頁面的文字。"""
    pdf_doc = pdfium.PdfDocument(pdf_path)
    page = pdf_doc[page_index]
    bitmap = page.render(scale=3)
    pil_image = bitmap.to_pil()
    text = pytesseract.image_to_string(pil_image, lang="chi_tra+chi_sim+eng")
    return text.strip()


# ── 檔案 hash ─────────────────────────────────────────────

def get_file_hash(filepath: str) -> str:
    """計算檔案 SHA256 hash。"""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── 處理記錄 ──────────────────────────────────────────────

def load_processed_log() -> dict:
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_processed_log(log: dict):
    with open(PROCESSED_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


# ── PDF 解析 ──────────────────────────────────────────────

def parse_pdf(path: str, filename: str, progress_callback=None) -> list[Document]:
    """解析 PDF 文字與表格，按頁返回 Document list。
    一般 PDF 用 pdfplumber，掃描頁面自動改用 Tesseract OCR。
    progress_callback(current, total, filename) 可用於 UI 進度更新。
    """
    documents = []
    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)
        for page in tqdm(pdf.pages, desc=f"  解析 {filename}", unit="頁"):
            page_num = page.page_number
            page_text_parts = []

            # 擷取表格
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    header = table[0]
                    md = "| " + " | ".join(str(c) if c else "" for c in header) + " |\n"
                    md += "| " + " | ".join("---" for _ in header) + " |\n"
                    for row in table[1:]:
                        md += "| " + " | ".join(str(c) if c else "" for c in row) + " |\n"
                    page_text_parts.append(md)

            # 擷取文字
            text = page.extract_text()
            if text and text.strip():
                page_text_parts.append(text)

            # 掃描頁面 fallback OCR
            if not page_text_parts:
                ocr_text = ocr_page(path, page_num - 1)
                if ocr_text.strip():
                    page_text_parts.append(ocr_text)

            if page_text_parts:
                documents.append(Document(
                    page_content="\n\n".join(page_text_parts),
                    metadata={"source": filename, "page": page_num},
                ))

            if progress_callback:
                progress_callback(page_num, total_pages, filename)

    return documents


# ── VectorStore 增量更新 ──────────────────────────────────

def sync_vectorstore(progress_callback=None) -> Chroma:
    """掃描 PDF 資料夾，增量更新 VectorStore。
    progress_callback(current, total, filename) 可用於 UI 進度更新。
    回傳 Chroma vectorstore 實例。
    """
    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)

    embedding = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    processed_log = load_processed_log()

    current_files = {
        f.name: str(f) for f in Path(PDF_FOLDER).glob("*.pdf")
    }

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding,
        collection_name="pdf_docs",
    )

    # 1) 刪除已移除的檔案
    removed = set(processed_log.keys()) - set(current_files.keys())
    for filename in removed:
        print(f"  移除: {filename}")
        vectorstore.delete(where={"source": filename})
        del processed_log[filename]

    # 2) 處理新增或修改的檔案
    added_count = 0
    for filename, filepath in current_files.items():
        file_hash = get_file_hash(filepath)

        if filename in processed_log and processed_log[filename] == file_hash:
            continue

        if filename in processed_log:
            print(f"  更新: {filename}")
            vectorstore.delete(where={"source": filename})
        else:
            print(f"  新增: {filename}")

        page_docs = parse_pdf(filepath, filename, progress_callback=progress_callback)
        if not page_docs:
            print(f"  警告: {filename} 解析後無內容，跳過。")
            continue
        docs = text_splitter.split_documents(page_docs)
        if not docs:
            print(f"  警告: {filename} 切割後無文件，跳過。")
            continue
        vectorstore.add_documents(docs)
        processed_log[filename] = file_hash
        added_count += 1

    save_processed_log(processed_log)

    if not removed and added_count == 0:
        print("  所有檔案已是最新，無需更新。")

    return vectorstore
