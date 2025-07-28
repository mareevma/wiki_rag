from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from index_documents import get_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

"""Agentic RAG chatbot (ReAct-style) ---------------------------------------

This script demonstrates an "agentic" flavour of Retrieval-Augmented Generation.
Instead of running a fixed RAG chain, we expose the Vector Store retriever as an
explicit *tool*.  Using OpenAI function-calling, the language-model can decide,
step-by-step, when to call the retriever (possibly several times with different
queries) before formulating a final answer.

The overall design follows the Hugging Face "agent_rag" cookbook notebook
(https://huggingface.co/learn/cookbook/agent_rag) while re-using the existing
indexing/retrieval code that already ships with the project.
"""

# --------------------------------------------------------------------------------------
# 1. initialise LLM
# --------------------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# --------------------------------------------------------------------------------------
# 2. tool: semantic retriever
# --------------------------------------------------------------------------------------

# we keep using the ParentDocument retriever that proved to work well in chat_rag.py
retriever = get_retriever(use_parent_document=True)


def _search_knowledge_base(query: str, k: int = 7) -> str:  # noqa: D401
    """Retrieve *k* most relevant docs and return them as a printable string."""

    results = retriever.get_relevant_documents(query)[:k]
    if not results:
        return "No documents found."

    blob_parts = [
        f"===== Document {i} =====\n{doc.page_content}\n" for i, doc in enumerate(results)
    ]
    return "\n".join(blob_parts)


retriever_tool = Tool(
    name="retriever",
    func=_search_knowledge_base,
    description=(
        "Используйте **retriever** для поиска по технической документации. "
        "На вход подаётся **утвердительное предложение** (не вопрос), кратко описывающее, что нужно найти. "
        "Инструмент возвращает сырые текстовые фрагменты, которые затем можно использовать в ответе. "
        "Вызывайте инструмент столько раз, сколько требуется, чтобы собрать достаточно доказательств, "
        "прежде чем формулировать финальный ответ."
    ),
)

TOOLS = [retriever_tool]

# --------------------------------------------------------------------------------------
# 3. system prompt (+function agent)
# --------------------------------------------------------------------------------------

SYSTEM_MSG = (
    "Вы — полезный ассистент, специализирующийся на технической документации ФПСУ-IP. "
    "При необходимости спланируйте короткую цепочку размышлений (она не демонстрируется пользователю), "
    "вызовите инструмент `retriever` с точными утвердительными формулировками для поиска контекста, "
    "а затем сформулируйте исчерпывающий ответ **на русском языке**. "
    "Если ответа нет в базе знаний — честно скажите, что не знаете. "
    "ВАЖНО: если в найденных отрывках присутствуют изображения в формате Markdown (![alt](file.png)), "
    "обязательно включайте их в ответ."
)

# Формируем prompt для агента (OpenAI function calling)
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Создаем агента
agent = create_openai_functions_agent(llm, TOOLS, agent_prompt)
executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)

# --------------------------------------------------------------------------------------
# 4. CLI helper --------------------------------------------------------------
# --------------------------------------------------------------------------------------

def chat(user_input: str) -> str:
    """Run the Agentic RAG pipeline on *user_input* and return the model answer."""

    result = executor.invoke({"input": user_input})
    return result["output"]  # AgentExecutor stores agent answer under key "output"


if __name__ == "__main__":
    print("🚀 Agentic RAG (function-calling) запущен! (Введите 'exit' для выхода.)")
    try:
        while True:
            q = input("\n👤 Вы: ")
            if q.lower().strip() in {"exit", "quit", "выход"}:
                print("👋 До свидания!")
                break
            if not q.strip():
                continue

            answer = chat(q)
            print("\n🪄 Ответ:\n", answer)
    except KeyboardInterrupt:
        print("\n👋 До свидания!") 