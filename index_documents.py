# index_documents.py (ОБНОВЛЕННАЯ ВЕРСИЯ)

import pathlib
import re
import unicodedata
import html
import pickle
import multiprocessing
import os
from typing import List, Optional

import tiktoken
from bs4 import BeautifulSoup
from charset_normalizer import from_path
from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema import Document
from langchain.storage import EncoderBackedStore
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.storage import SQLStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markdownify import markdownify as mdify

load_dotenv()

# --- Упрощенная и более корректная секция парсинга ---

def _clean(txt: str) -> str:
    """Очищает текст от лишних переносов строк и HTML-сущностей."""
    txt = unicodedata.normalize("NFC", html.unescape(txt))
    return re.sub(r"\n{2,}", "\n", txt).strip()

def _html_to_md(soup: BeautifulSoup) -> str:
    """
    Конвертирует HTML в Markdown. 
    markdownify сама корректно обработает <img>, сохранив относительные пути.
    """
    # Удаляем ненужные теги перед конвертацией
    for s in soup(['script', 'style']):
        s.decompose()
    # Конвертируем body в Markdown, heading_style="ATX" делает заголовки вида #, ##
    return mdify(str(soup), heading_style="ATX")

def parse_file(path: pathlib.Path) -> Optional[dict]:
    """
    Парсит HTML-файл, извлекает контент и метаданные.
    Важно: сохраняет абсолютный путь к самому HTML-файлу в `source`.
    """
    try:
        # Определяем кодировку и читаем файл
        results = from_path(path)
        best_match = results.best()
        raw_html = best_match.output(encoding=best_match.encoding) if best_match else path.read_text(encoding='utf-8', errors='replace')
        
        soup = BeautifulSoup(raw_html, "html.parser")
        
        # Находим заголовок
        title_tag = soup.select_one("#printheader h1, #idheader h1, h1")
        title = title_tag.get_text(strip=True) if title_tag else (soup.title.get_text(strip=True) if soup.title else path.stem)
        
        body_tag = soup.find('body')
        if not body_tag: return None
        
        # Получаем чистый Markdown с относительными путями к картинкам
        text_md = _clean(_html_to_md(body_tag))
        
        if not text_md.strip(): return None

        # Возвращаем словарь с данными. `source` содержит полный путь к файлу.
        # Это КЛЮЧЕВОЙ момент для последующего нахождения картинок.
        return {
            "title": title,
            "text": text_md,
            "source": str(path.resolve()), # Абсолютный путь к HTML файлу
            "product": path.parent.parent.name,
        }
    except Exception as e:
        print(f"⚠️  Не удалось обработать файл {path}: {e}")
        return None

# ==============================================================================
# Остальная часть файла `index_documents.py` остается БЕЗ ИЗМЕНЕНИЙ.
# Просто скопируйте и вставьте ее сюда из вашего оригинального файла.
# ... (iter_html_files, get_retriever, count_tokens_for_documents, main, etc.)
# ==============================================================================

# (начало неизмененной части)
def iter_html_files(root: pathlib.Path):
    html_files = list(root.glob("**/HTML/*.html"))
    print(f"Найдено {len(html_files)} HTML файлов для обработки.")
    return html_files

def get_retriever(use_parent_document=True, use_openai_embeddings=True):
    if use_openai_embeddings:
        print("🤖 Использование эмбеддингов OpenAI (text-embedding-3-small)...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", request_timeout=120)
    else:
        print("🤖 Использование локальной модели эмбеддингов (Hugging Face)...")
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

    if use_parent_document:
        print("🔧 Настройка ParentDocumentRetriever...")
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        vectorstore = Chroma(collection_name="docs_multiproduct_parent_v3_relative", persist_directory="chroma_db", embedding_function=embeddings)
        
        underlying_store = SQLStore(namespace="documents_v3", db_url="sqlite:///docstore.db")
        underlying_store.create_schema()
        store = EncoderBackedStore(
            store=underlying_store,
            key_encoder=lambda x: x,
            value_serializer=pickle.dumps,
            value_deserializer=pickle.loads
        )

        retriever = ParentDocumentRetriever(vectorstore=vectorstore, docstore=store, child_splitter=child_splitter)
        return retriever
    else:
        print("🔧 Настройка оптимизированного векторного retriever'а...")
        vectorstore = Chroma(collection_name="docs_multiproduct_v5_optimized", persist_directory="chroma_db", embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
        return retriever

def count_tokens_for_documents(documents: List[Document], model_name: str = "cl100k_base") -> int:
    try:
        encoding = tiktoken.get_encoding(model_name)
    except ValueError:
        encoding = tiktoken.get_encoding("gpt2")
    total_tokens = sum(len(encoding.encode(doc.page_content)) for doc in documents)
    return total_tokens

def main(use_parent_document=True, max_files=None, use_openai=True):
    mode_name = "ParentDocumentRetriever" if use_parent_document else "простой векторный поиск"
    print(f"🚀 Начало индексации документов ({mode_name})...")
    ROOT = pathlib.Path("data")

    files_to_process = iter_html_files(ROOT)
    if max_files:
        files_to_process = files_to_process[:max_files]
        print(f"🔄 Режим тестирования: обработка первых {max_files} файлов.")

    parent_docs = []
    with multiprocessing.Pool() as pool:
        results = pool.map(parse_file, files_to_process)

    for parsed_content in results:
        if parsed_content:
            parent_docs.append(Document(page_content=parsed_content["text"], metadata={
                "title": parsed_content["title"], 
                "source": parsed_content["source"], # <-- Важно, что здесь полный путь
                "product": parsed_content["product"],
            }))

    print(f"✅ Загружено уникальных документов: {len(parent_docs)}")
    if not parent_docs:
        print("❌ Не найдено ни одного документа для индексации.")
        return

    if use_openai:
        total_tokens = count_tokens_for_documents(parent_docs)
        price_per_million_tokens = 0.02
        estimated_cost = (total_tokens / 1_000_000) * price_per_million_tokens
        print("\n" + "="*50)
        print("📊 Оценка использования токенов (OpenAI)")
        print(f"   Всего токенов для эмбеддингов: {total_tokens:,}")
        print(f"   Примерная стоимость: ${estimated_cost:.6f}")
        print("="*50 + "\n")
    else:
        print("\n" + "="*50 + "\n📊 Используются бесплатные локальные эмбеддинги." + "\n" + "="*50 + "\n")

    retriever = get_retriever(use_parent_document=use_parent_document, use_openai_embeddings=use_openai)

    batch_size = 100
    total_docs = len(parent_docs)
    
    print("⏳ Добавление документов в retriever порциями...")
    for i in range(0, total_docs, batch_size):
        batch_docs = parent_docs[i:i + batch_size]
        print(f"  📦 Обработка батча {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} ({len(batch_docs)} документов)")
        
        retriever.add_documents(batch_docs, ids=None)

    print("🎉 Индексация успешно завершена!")

if __name__ == "__main__":
    import sys
    # Важно: После смены логики нужно переиндексировать документы
    # Можно поменять имя коллекции в Chroma, чтобы не было конфликтов
    print("‼️ ВАЖНО: Логика парсинга изменилась. Рекомендуется удалить старую базу Chroma (папку chroma_db) для полной переиндексации.")
    
    use_parent_document = "--simple" not in sys.argv
    use_openai_emb = "--local-embeddings" not in sys.argv
    max_files = None
    if "--test" in sys.argv: max_files = 100
    
    print(f"🎯 Режим: {'ParentDocumentRetriever' if use_parent_document else 'Простой векторный'}")
    print(f"🤖 Эмбеддинги: {'OpenAI' if use_openai_emb else 'Локальные (Hugging Face)'}")
    if use_openai_emb and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️ ВНИМАНИЕ: Не найден OPENAI_API_KEY. Убедитесь, что он есть в .env файле.")
    
    main(use_parent_document=use_parent_document, max_files=max_files, use_openai=use_openai_emb)