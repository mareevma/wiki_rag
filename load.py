from charset_normalizer import from_path
from bs4 import BeautifulSoup, NavigableString, Comment # 👈 Добавлен Comment
import re, unicodedata, html
from pathlib import Path
from markdownify import markdownify as mdify

# — вспомогательные функции (без изменений) —
def _clean(txt: str) -> str:
    txt = unicodedata.normalize("NFC", html.unescape(txt))
    return re.sub(r"\n{2,}", "\n", txt).strip()

def _html_to_md_with_imgs(soup: BeautifulSoup) -> str:
    for img in soup.find_all("img", src=True):
        img.replace_with(NavigableString(f"![{img.get('alt', '')}]({img['src']})"))
    return mdify(str(soup), heading_style="ATX")

# — 🔥 ОПТИМИЗИРОВАННАЯ ВЕРСИЯ ПАРСЕРА —
def load_zoom_section_v3_optimized(path: str) -> dict:
    # 1) Автоопределение кодировки (без изменений)
    results = from_path(path)
    best_match = results.best()
    if not best_match:
        # Обработка случая, когда кодировка не определена
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            raw_html = f.read()
    else:
        raw_html = best_match.output(encoding=best_match.encoding)

    # 2) Единый разбор BeautifulSoup
    soup = BeautifulSoup(raw_html, "html.parser")

    # Заголовок (можно сделать чуть короче с CSS-селектором)
    title_tag = soup.select_one("#printheader h1, #idheader h1")
    title = title_tag.get_text(strip=True) if title_tag else soup.title.get_text(strip=True)

    # 3) 🔥 Извлечение контента между комментариями через BeautifulSoup
    zoom_soup = BeautifulSoup('<html><body></body></html>', 'html.parser') # Создаем пустой суп для контента
    body = zoom_soup.body

    # Ищем стартовый комментарий
    start_comment = soup.find(string=lambda text: isinstance(text, Comment) and "ZOOMRESTART" in text)
    
    if start_comment:
        # Итерируемся по всем следующим элементам
        for sibling in start_comment.find_all_next():
            # Если дошли до конечного комментария, останавливаемся
            if isinstance(sibling, Comment) and "ZOOMSTOP" in sibling:
                break
            # Копируем тег в наш новый суп
            if sibling.name: # Проверяем, что это тег, а не просто строка
                 body.append(sibling)

    text_md = _clean(_html_to_md_with_imgs(zoom_soup))

    return {
        "title":   title,
        "text":    text_md,
        "source":  str(path),
    }