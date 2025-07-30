
import re
import pathlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional 
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from index_documents import get_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document

# --- ФУНКЦИЯ ДЛЯ ДИНАМИЧЕСКОГО ПОЛУЧЕНИЯ ПРОДУКТОВ ---
def get_known_products(data_root_path: str = "data") -> dict:
    """Сканирует директорию `data` и возвращает словарь продуктов."""
    root = pathlib.Path(data_root_path)
    products = {}
    if not root.is_dir():
        return products
    for product_path in root.iterdir():
        if product_path.is_dir():
            product_name = product_path.name
            products[product_name.lower()] = product_name
    return products

# --- НАСТРОЙКА ОСНОВНЫХ КОМПОНЕНТОВ ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
retriever = get_retriever(use_parent_document=True)
embeddings_model = retriever.vectorstore._embedding_function

# --- ДИНАМИЧЕСКОЕ ОПРЕДЕЛЕНИЕ ПРОДУКТОВ ---
auto_products = get_known_products()
CANONICAL_PRODUCT_NAMES = list(auto_products.values())
print(f"✅ Загружены продукты из папок: {CANONICAL_PRODUCT_NAMES}")

# --- ФУНКЦИЯ ДЛЯ СЕМАНТИЧЕСКОГО ПОИСКА ПРОДУКТА ---
def _find_target_product_semantically(
    query: str, 
    product_names: list[str], 
    threshold=0.6
) -> Optional[str]:
    """Находит наиболее семантически близкий продукт к запросу, используя эмбеддинги."""
    if not product_names:
        return None

    query_embedding = embeddings_model.embed_query(query)
    product_embeddings = embeddings_model.embed_documents(product_names)

    similarities = cosine_similarity([query_embedding], product_embeddings)[0]

    max_similarity_index = np.argmax(similarities)
    max_similarity = similarities[max_similarity_index]

    if max_similarity > threshold:
        return product_names[max_similarity_index]
    
    return None

# --- ОБНОВЛЕННАЯ ФУНКЦИЯ ПОИСКА В БАЗЕ ЗНАНИЙ ---
def _search_knowledge_base(query: str, k: int = 7) -> str:
    target_product = _find_target_product_semantically(query, CANONICAL_PRODUCT_NAMES)
    
    if target_product:
        print(f"ℹ️  Семантически определен продукт: '{target_product}'")

    initial_results = retriever.get_relevant_documents(query, k=20)
    
    if not initial_results:
        return "Документы не найдены."

    if target_product:
        filtered_results = [
            doc for doc in initial_results if doc.metadata.get('product') == target_product
        ]
        if not filtered_results:
            return f"По вашему запросу не найдено релевантных документов для продукта '{target_product}'."
        results = filtered_results[:k]
    else:
        results = initial_results[:k]

    if not results:
        return "Документы не найдены."

    blob_parts = []
    for i, doc in enumerate(results):
        content = doc.page_content
        product = doc.metadata.get('product', 'Неизвестный продукт')
        title = doc.metadata.get('title', 'Без заголовка')
        source_path = doc.metadata.get('source', 'Неизвестный источник')
        
        source_header = (
            f"Источник {i+1}: "
            f"Product='{product}', "
            f"Title='{title}', "
            f"SourceFile='{source_path}'"
        )
        blob_parts.append(f"===== {source_header} =====\n{content}\n")
        
    return "\n".join(blob_parts)

# --- Настройка агента ---
retriever_tool = Tool(
    name="retriever",
    func=_search_knowledge_base,
    description="Используйте для поиска по технической документации."
)
TOOLS = [retriever_tool]


SYSTEM_MSG = (
    "Вы — ассистент по технической документации. Ваша задача — точно следовать инструкциям и генерировать ответы на основе документов.\n\n"
    "**САМОЕ ГЛАВНОЕ ПРАВИЛО:**\n"
    "В предоставленном контексте есть текст и изображения в формате Markdown: `![...](...)`. "
    "Изображение и текст, который находится непосредственно до и после него, являются **единым неделимым блоком**. "
    "**ЗАПРЕЩЕНО** перефразировать, сокращать или изменять текст, который окружает изображение. Ты должен скопировать этот блок (текст-картинка-текст) в свой ответ **ДОСЛОВНО**, чтобы сохранить контекст.\n\n"
    "**ПРИМЕР:**\n"
    "КОНТЕКСТ: '...в появившемся окне нажмите кнопку «Далее». ![Снимок окна с кнопкой Далее](img1.png) Это действие приведет к открытию следующего диалога...'\n"
    "ПРАВИЛЬНЫЙ ОТВЕТ: '...в появившемся окне нажмите кнопку «Далее». ![Снимок окна с кнопкой Далее](img1.png) Это действие приведет к открытию следующего диалога...'\n"
    "НЕПРАВИЛЬНЫЙ ОТВЕТ (ЗАПРЕЩЕНО): '...нажмите «Далее», и откроется новый диалог. ![Снимок окна с кнопкой Далее](img1.png)'\n\n"
    "**ДОПОЛНИТЕЛЬНЫЕ ПРАВИЛА:**\n"
    "1. Если пользователь указывает в вопросе продукт (например, 'ФПСУ Амиго'), контекст будет содержать документы ТОЛЬКО по этому продукту. Основывай ответ исключительно на них. Если контекст говорит, что документов по продукту нет, сообщи об этом пользователю.\n"
    "2. После основного ответа всегда добавляй секцию `Источники:`.\n"
    "3. В источниках для каждого использованного документа создай список следующего формата: `Продукт: [Название продукта], Документ: [Название документа]`."
)

agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
agent = create_openai_functions_agent(llm, TOOLS, agent_prompt)
executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=False, return_intermediate_steps=True)

def chat(user_input: str) -> dict:
    result = executor.invoke({"input": user_input})
    output_text = result.get("output", "Произошла ошибка.")
    tool_output = "\n".join(
        [str(step[1]) for step in result.get("intermediate_steps", [])]
    )
    return {"text": output_text, "tool_output": tool_output}

if __name__ == "__main__":
    while True:
        q = input("\n👤 Вы: ")
        if q.lower().strip() in {"exit", "quit", "выход"}: break
        answer = chat(q)
        print("\n🪄 Ответ модели для CLI:\n", answer['text'])