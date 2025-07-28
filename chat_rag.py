from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from index_documents import get_retriever
# Альтернативно можно использовать:
# from index_documents_parent import get_retriever_with_persistent_docstore

# Настройка кэширования LLM
set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))

# 1) LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# 2) Промпт для контекстуализации вопроса (переформулирование с учетом истории)
contextualize_q_system_prompt = (
    "Дан чат-история и последний вопрос пользователя, который может "
    "ссылаться на контекст из истории чата. Сформулируй самостоятельный "
    "вопрос, который можно понять без истории чата. НЕ отвечай на вопрос, "
    "просто переформулируй его при необходимости, иначе верни как есть."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 3) Получение retriever'а и создание history-aware retriever
# Для переключения на ParentDocument режим измените на: get_retriever(use_parent_document=True)
USE_PARENT_DOCUMENT = True  # 🔧 Настройка: True для ParentDocument, False для простого поиска

retriever = get_retriever(use_parent_document=USE_PARENT_DOCUMENT)
print(f"🔧 Режим retriever'а: {'ParentDocument' if USE_PARENT_DOCUMENT else 'Простой векторный'}")

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 4) Промпт для ответа на вопрос
qa_system_prompt = (
    "Ты — помощник по технической документации ФПСУ-IP. "
    "Используй следующие фрагменты извлеченного контекста для ответа "
    "на вопрос. Если не знаешь ответа, просто скажи, что не знаешь. "
    "Отвечай на русском языке.\n\n"
    "ВАЖНО: Если в контексте есть изображения в формате ![описание](имя_файла.png), "
    "ОБЯЗАТЕЛЬНО включай их в свой ответ в том же формате. Изображения помогают "
    "пользователю лучше понять интерфейс и процедуры.\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 5) Question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# 6) RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 7) Управление историей сообщений
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 8) Conversational RAG chain с историей
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ——————————————————————————————
# Функция chat(user_input) для интерактивной CLI
# ——————————————————————————————
def chat(user_input: str, session_id: str = "default_session", debug_context: bool = False):
    """Отправить вопрос в RAG систему с сохранением истории"""
    
    result = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    
    answer = result["answer"]
    sources = result.get("context", [])  # context содержит исходные документы
    
    # Отладка: проверяем, есть ли изображения в контексте
    if debug_context:
        print("\n🔍 DEBUG: Анализ контекста:")
        for i, source in enumerate(sources[:3], 1):
            import re
            images = re.findall(r'\!\[.*?\]\(.*?\)', source.page_content)
            print(f"  Источник {i}: {len(images)} изображений найдено")
            if images:
                for j, img in enumerate(images, 1):
                    print(f"    {j}. {img}")
        print()
    
    # Печатаем результат
    print("\n🪄 Ответ:\n", answer)
    if sources:
        print("\n📄 Источники:")
        for i, source in enumerate(sources[:3], 1):  # показываем первые 3 источника
            title = source.metadata.get("title", "Неизвестный источник")
            source_path = source.metadata.get("source", "")
            
            # Если есть путь к источнику, выводим его как ссылку
            if source_path:
                print(f"  {i}. {title}")
                print(f"     📎 Источник: {source_path}")
                
                # Дополнительная информация из метаданных (если есть)
                if "page" in source.metadata:
                    print(f"     📄 Страница: {source.metadata['page']}")
            else:
                print(f"  {i}. {title}")
    
    return result

# — Пример запуска —
if __name__ == "__main__":
    print("🚀 RAG-чат с историей запущен!")
    print("💡 Задавайте вопросы о ФПСУ-IP (exit — выйти)")
    
    session_id = "user_session_001"  # можно сделать уникальным для каждого пользователя
    
    debug_mode = False
    
    while True:
        try:
            q = input("\n👤 Вы: ")
            if q.lower() in {"exit", "quit", "выход"}:
                print("👋 До свидания!")
                break
            elif q.lower() in {"debug", "дебаг"}:
                debug_mode = not debug_mode
                print(f"🔧 Debug режим: {'включен' if debug_mode else 'выключен'}")
                continue
            
            if q.strip():  # проверяем, что вопрос не пустой
                chat(q, session_id, debug_context=debug_mode)
            else:
                print("⚠️  Пожалуйста, введите вопрос")
                print("💡 Подсказка: введите 'debug' для включения режима отладки")
                
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            print("🔄 Попробуйте снова...")
