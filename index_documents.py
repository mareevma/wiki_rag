import pathlib
import re
import unicodedata
import html
from itertools import chain
from typing import Optional

# --- Библиотеки для парсинга ---
from charset_normalizer import from_path
from bs4 import BeautifulSoup, NavigableString
from markdownify import markdownify as mdify

# --- Библиотеки LangChain ---
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_core.stores import BaseStore
import pickle
import os
from typing import Iterator, List, Tuple, Any
from langchain.schema import Document

# ==============================================================================
# 1. Персистентный docstore
# ==============================================================================

class PickleDocStore(BaseStore):
    """Простое файловое хранилище для документов с использованием pickle"""
    
    def __init__(self, store_path="./docstore"):
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
        self.index_file = os.path.join(store_path, "index.pkl")
        self._load_index()
    
    def _load_index(self):
        """Загружает индекс документов"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'rb') as f:
                    self._index = pickle.load(f)
            except:
                self._index = {}
        else:
            self._index = {}
    
    def _save_index(self):
        """Сохраняет индекс документов"""
        with open(self.index_file, 'wb') as f:
            pickle.dump(self._index, f)
    
    def mset(self, key_value_pairs: List[Tuple[str, Any]]) -> None:
        """Сохраняет несколько документов"""
        for key, value in key_value_pairs:
            # Создаем безопасное имя файла из ключа
            safe_key = str(hash(key) % 10000000)  # Используем хеш для безопасного имени
            file_path = os.path.join(self.store_path, f"doc_{safe_key}.pkl")
            
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            
            self._index[key] = file_path
        
        self._save_index()
    
    def mget(self, keys: List[str]) -> List[Any]:
        """Получает несколько документов"""
        results = []
        for key in keys:
            if key in self._index:
                try:
                    with open(self._index[key], 'rb') as f:
                        results.append(pickle.load(f))
                except:
                    results.append(None)
            else:
                results.append(None)
        return results
    
    def mdelete(self, keys: List[str]) -> None:
        """Удаляет документы"""
        for key in keys:
            if key in self._index:
                try:
                    os.remove(self._index[key])
                except:
                    pass
                del self._index[key]
        self._save_index()
    
    def yield_keys(self, prefix: str = "") -> Iterator[str]:
        """Возвращает все ключи"""
        for key in self._index.keys():
            if key.startswith(prefix):
                yield key

# ==============================================================================
# 2. Секция парсинга HTML (с резервной логикой)
# ==============================================================================

def _clean(txt: str) -> str:
    """Очищает текст: нормализует, убирает двойные переносы строк."""
    txt = unicodedata.normalize("NFC", html.unescape(txt))
    return re.sub(r"\n{2,}", "\n", txt).strip()

def _html_to_md_with_imgs(soup: BeautifulSoup) -> str:
    """Конвертирует HTML в Markdown, сохраняя теги <img>."""
    for img in soup.find_all("img", src=True):
        img.replace_with(NavigableString(f"![{img.get('alt', '')}]({img['src']})"))
    # Удаляем скрипты и стили перед конвертацией, чтобы избежать мусора
    for s in soup(['script', 'style']):
        s.decompose()
    return mdify(str(soup), heading_style="ATX")

def load_zoom_section_v3_optimized(path: pathlib.Path) -> Optional[dict]:
    """
    Надёжно загружает и парсит HTML-файл.
    Извлекает весь контент из тега <body>.
    """
    try:
        # 1) Автоопределение кодировки
        results = from_path(path)
        best_match = results.best()
        raw_html = best_match.output(encoding=best_match.encoding) if best_match else path.read_text(encoding='utf-8', errors='replace')

        # 2) Единый разбор с помощью BeautifulSoup
        soup = BeautifulSoup(raw_html, "html.parser")

        # 3) Извлечение заголовка
        title_tag = soup.select_one("#printheader h1, #idheader h1, h1")
        title = title_tag.get_text(strip=True) if title_tag else (soup.title.get_text(strip=True) if soup.title else path.stem)

        # 4) Извлечение контента из <body>
        body_tag = soup.find('body')
        if body_tag:
            content_soup = body_tag
        else:
            content_soup = None

        if not content_soup:
            return None # Не удалось найти тег <body>

        # 5) Конвертация в Markdown
        text_md = _clean(_html_to_md_with_imgs(content_soup))

        return {
            "title":   title,
            "text":    text_md,
            "source":  str(path.resolve()),
        }
    except Exception as e:
        print(f"⚠️  Не удалось обработать файл {path}: {e}")
        return None

# ==============================================================================
# 2. Логика индексации документов (без изменений)
# ==============================================================================

def iter_html_files(root: pathlib.Path):
    """Находит все .html файлы в подпапках */HTML/ и возвращает (имя_продукта, путь_к_файлу)."""
    for product_dir in root.iterdir():
        if not product_dir.is_dir():
            continue

        product_name = product_dir.name
        html_dir = product_dir / "HTML"
        if html_dir.exists() and html_dir.is_dir():
            yield from ((product_name, p) for p in html_dir.glob("*.html"))

def get_retriever(use_parent_document=True):  # Возвращаем True по умолчанию
    """
    Создает и возвращает настроенный retriever для использования в RAG системе.
    
    Args:
        use_parent_document (bool): Если True, использует ParentDocumentRetriever
                                   Если False, использует оптимизированный простой поиск
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", request_timeout=60)
    
    if use_parent_document:
        print("🔧 Настройка ParentDocumentRetriever...")
        
        # --- Настройка ParentDocumentRetriever с InMemoryStore ---
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # маленькие чанки для точного поиска
            chunk_overlap=50
        )
        
        vectorstore = Chroma(
            collection_name="docs_multiproduct_parent_v1",
            persist_directory="chroma_db",
            embedding_function=embeddings,
        )
        
        # Используем персистентный docstore
        docstore = PickleDocStore("./docstore")
        
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=None,  # храним полные документы
        )
        return retriever
    
    else:
        print("🔧 Настройка оптимизированного векторного retriever'а...")
        
        # --- Оптимизированный простой retriever ---
        vectorstore = Chroma(
            collection_name="docs_multiproduct_v4_optimized",
            persist_directory="chroma_db",
            embedding_function=embeddings,
        )
        
        # Больше результатов для лучшего контекста
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}  # Увеличено с 6 до 8
        )
        return retriever

def main(use_parent_document=True, max_files=None):
    """
    Главная функция для выполнения индексации всех инструкций.
    
    Args:
        use_parent_document (bool): Использовать ParentDocumentRetriever
        max_files (int): Максимальное количество файлов для обработки (None = все)
    """
    mode_name = "ParentDocumentRetriever" if use_parent_document else "простой векторный поиск"
    if max_files:
        print(f"🚀 Начало индексации документов ({mode_name}) - ТЕСТ режим: {max_files} файлов...")
    else:
        print(f"🚀 Начало индексации документов ({mode_name})...")
    ROOT = pathlib.Path("data")

    # --- Шаг 1: Чтение и парсинг HTML-файлов ---
    parent_docs = []
    seen_paths = set()  # Для отслеживания уже обработанных файлов
    processed_count = 0

    for product, path in iter_html_files(ROOT):
        # Ограничение на количество файлов
        if max_files and processed_count >= max_files:
            print(f"🔄 Достигнут лимит файлов: {max_files}")
            break
            
        path_str = str(path.resolve())
        if path_str in seen_paths:
            continue
        seen_paths.add(path_str)

        parsed_content = load_zoom_section_v3_optimized(path)
        if not parsed_content or not parsed_content["text"].strip(): # Проверяем, что текст не пустой
            print(f"⏩ Пропущен пустой или некорректный файл: {path.name}")
            continue
        
        doc = Document(
            page_content=parsed_content["text"],
            metadata={
                "title": parsed_content["title"],
                "source": parsed_content["source"],
                "product": product,
                "file_path": path_str  # Сохраняем оригинальный путь в метаданных
            }
        )
        parent_docs.append(doc)
        processed_count += 1
        
        # Показываем прогресс каждые 50 файлов
        if processed_count % 50 == 0:
            print(f"📊 Обработано файлов: {processed_count}")

    print(f"✅ Загружено уникальных документов: {len(parent_docs)}")
    if max_files:
        print(f"🎯 Режим тестирования: обработано {processed_count} из {max_files} запрошенных файлов")

    if not parent_docs:
        print("❌ Не найдено ни одного документа для индексации. Проверьте структуру папки 'data/'.")
        return

    # --- Шаг 2: Получение retriever'а ---
    retriever = get_retriever(use_parent_document=use_parent_document)

    if use_parent_document:
        # --- ParentDocumentRetriever: добавляем документы напрямую ---
        print("⏳ Добавление документов в ParentDocumentRetriever...")
        
        batch_size = 100  # меньший батч для ParentDocument (больше операций)
        total_docs = len(parent_docs)
        
        for i in range(0, total_docs, batch_size):
            batch_docs = parent_docs[i:i + batch_size]
            
            print(f"📦 Обработка батча {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} ({len(batch_docs)} документов)")
            
            try:
                # Передаем ids=None согласно документации LangChain
                retriever.add_documents(batch_docs, ids=None)
            except Exception as e:
                print(f"❌ Ошибка при добавлении батча {i//batch_size + 1}: {e}")
                # Пробуем меньший батч
                smaller_batch_size = max(1, batch_size // 4)
                print(f"🔄 Повторная попытка с размером батча: {smaller_batch_size}")
                for j in range(i, min(i + batch_size, total_docs), smaller_batch_size):
                    small_batch_docs = parent_docs[j:j + smaller_batch_size]
                    try:
                        retriever.add_documents(small_batch_docs, ids=None)
                    except Exception as inner_e:
                        print(f"❌ Не удалось добавить документы: {inner_e}")
                        
    else:
        # --- Простой retriever: сначала разбиваем на чанки ---
        print("📄 Разделение документов на чанки...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # оптимальный размер для баланса качества и скорости
            chunk_overlap=80
        )
        
        all_splits = []
        for doc in parent_docs:
            splits = text_splitter.split_documents([doc])
            all_splits.extend(splits)
        
        print(f"✂️  Создано чанков: {len(all_splits)}")
        
        print("⏳ Добавление чанков в векторное хранилище...")
        vectorstore = retriever.vectorstore
        
        batch_size = 200  # больший батч для простых чанков
        total_docs = len(all_splits)
        
        for i in range(0, total_docs, batch_size):
            batch_docs = all_splits[i:i + batch_size]
            
            print(f"📦 Обработка батча {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} ({len(batch_docs)} документов)")
            
            try:
                vectorstore.add_documents(batch_docs)
            except Exception as e:
                print(f"❌ Ошибка при добавлении батча {i//batch_size + 1}: {e}")
                # Попробуем меньший размер батча
                smaller_batch_size = max(1, batch_size // 2)
                print(f"🔄 Повторная попытка с меньшим размером батча: {smaller_batch_size}")
                for j in range(i, min(i + batch_size, total_docs), smaller_batch_size):
                    small_batch_docs = all_splits[j:j + smaller_batch_size]
                    try:
                        vectorstore.add_documents(small_batch_docs)
                    except Exception as inner_e:
                        print(f"❌ Не удалось добавить документы: {inner_e}")
    
    print("🔍 Индексация успешно завершена!")


if __name__ == "__main__":
    import sys
    
    # Проверяем аргументы командной строки
    use_parent_document = "--simple" not in sys.argv  # По умолчанию ParentDocument
    
    # Проверяем лимит файлов для тестирования
    max_files = None
    if "--test" in sys.argv:
        max_files = 100  # Тест режим: только 100 файлов
    elif "--small-test" in sys.argv:
        max_files = 20   # Маленький тест: только 20 файлов
    
    if use_parent_document:
        print("🎯 Режим: ParentDocumentRetriever (по умолчанию)")
        print("💡 Для простого режима запустите: python index_documents.py --simple")
    else:
        print("🎯 Режим: Простой векторный поиск") 
        print("💡 Для ParentDocument режима запустите: python index_documents.py")
    
    if max_files:
        print(f"🧪 Тест режим: обработка {max_files} файлов")
        print("💡 Дополнительные режимы:")
        print("   --test: 100 файлов")
        print("   --small-test: 20 файлов")
        print("   (без флагов): все файлы")
    
    main(use_parent_document=use_parent_document, max_files=max_files)