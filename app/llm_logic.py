import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import Tool
from langchain_ainetwork import AINetworkTool
from langchain_google_genai import ChatGoogleGenerativeAI
from app.rag.rag_retriever import get_retriever
from langchain.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent
from app.rag.rag_vectorstore import load_vectorstore
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# Set up Gemini LLM
llm = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GEMINI_API_KEY"),
    model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-pro")
)

# Example: Set up FAISS vector store retriever (persistent, Gemini-compatible)
try:
    retriever = get_retriever()
except Exception:
    # fallback to dummy retriever if no vectorstore exists yet
    docs = ["Addiction recovery is a process.", "Urge management strategies include mindfulness."]
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=os.getenv("GEMINI_API_KEY"),
        model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
    )
    from langchain.vectorstores import FAISS
    vectorstore = FAISS.from_texts(docs, embeddings)
    retriever = vectorstore.as_retriever()

# AINetwork tool integration
ainet_tool = AINetworkTool(api_key=os.getenv("AINETWORK_API_KEY"))

# System prompt for conversational RAG
CONVERSATIONAL_RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant for addiction recovery, trained on recovery literature. "
    "Use the following pieces of retrieved context to answer the user's question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    "Always include a disclaimer that you are not a human professional and cannot provide medical advice. "
    "If a user mentions crisis, guide them to human support and hotlines."
)

# Prompt template for RAG
rag_prompt = PromptTemplate(
    template=(
        "{system_message}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Helpful Answer:"
    ),
    input_variables=["system_message", "context", "question"]
)

# RAG pipeline
def get_rag_response(query):
    # Retrieve context
    context_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt = rag_prompt.format(
        system_message=CONVERSATIONAL_RAG_SYSTEM_PROMPT,
        context=context,
        question=query
    )
    return llm.invoke(prompt)

# Chat history QA
def get_chat_history_qa(query, history=None):
    # Use chat history in prompt
    chat_history = history or []
    context_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in context_docs)
    # Combine chat history into a string
    history_str = "\n".join(f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in chat_history if isinstance(msg, dict) and 'user' in msg and 'assistant' in msg)
    prompt = rag_prompt.format(
        system_message=CONVERSATIONAL_RAG_SYSTEM_PROMPT,
        context=(history_str + "\n" + context) if history_str else context,
        question=query
    )
    return llm.invoke(prompt)

# Tool calling example
def get_tool_calling_response(query):
    tool = Tool.from_function(ainet_tool)
    return tool.run(query)

# Streaming response example
def get_streaming_response(query):
    # Streaming with Gemini (pseudo-code, see LangChain docs for details)
    return "[Streaming not implemented in this scaffold]"

# Multimodal prompt example
def get_multimodal_response(query, image_url=None):
    # Multimodal prompt (pseudo-code, see LangChain docs for details)
    return "[Multimodal not implemented in this scaffold]"

# Agentic RAG setup (multi-step retrieval)
def get_agentic_rag_response(messages, thread_id=None):
    """
    Run an agentic RAG flow using LangGraph's ReAct agent for multi-step retrieval.
    messages: list of dicts, e.g. [{"role": "user", "content": "..."}, ...]
    thread_id: optional, for persistent memory across turns
    """
    # Load vectorstore and define retrieval tool
    vectorstore = load_vectorstore()
    def retrieve(query: str):
        docs = vectorstore.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in docs
        )
        return serialized, docs

    # Create agent executor
    memory = MemorySaver()
    agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)
    config = {"configurable": {"thread_id": thread_id or "default"}}
    # Run agent with messages
    result = agent_executor.invoke({"messages": messages}, config=config)
    return result["messages"][-1].content

# Tool-calling retrieval tool for LangGraph
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    docs = retriever.get_relevant_documents(query)
    serialized = "\n\n".join(
        (f"Source: {getattr(doc, 'metadata', {})}\nContent: {getattr(doc, 'page_content', '')}") for doc in docs
    )
    return serialized, docs

# LangGraph conversational RAG with memory/checkpointing
memory = MemorySaver()

def run_conversational_rag(messages, thread_id=None):
    """
    Run a conversational RAG flow using LangGraph's state and tool-calling.
    messages: list of dicts, e.g. [{"role": "user", "content": "..."}, ...]
    thread_id: optional, for persistent memory across turns
    """
    # Step 1: Generate an AIMessage that may include a tool-call to be sent.
    def query_or_respond(state: MessagesState):
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # Step 2: Execute the retrieval.
    tools = ToolNode([retrieve])

    # Step 3: Generate a response using the retrieved content.
    def generate(state: MessagesState):
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]
        docs_content = "\n\n".join(getattr(doc, 'content', '') for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n" + docs_content
        )
        conversation_messages = [
            message for message in state["messages"]
            if message.type in ("human", "system") or (message.type == "ai" and not getattr(message, 'tool_calls', None))
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages
        response = llm.invoke(prompt)
        return {"messages": [response]}

    # Build the graph
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("query_or_respond", query_or_respond)
    graph_builder.add_node("tools", tools)
    graph_builder.add_node("generate", generate)
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    graph = graph_builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": thread_id or "default"}}
    result = graph.invoke({"messages": messages}, config=config)
    return result["messages"][-1].content
