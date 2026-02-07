"""
Author: Karan Kadam

LangGraph Chatbot Backend (Tools + Optional PDF RAG + SQLite Checkpointing)

Overview:
- Creates a LangGraph-powered chatbot that can:
  1) Answer normally using an LLM
  2) Call tools when needed (web search, calculator, stock price, PDF RAG)
  3) Persist conversation state using a SQLite checkpointer
  4) Maintain per-thread PDF retrievers so each chat thread can query its own uploaded document

Key Concepts:
- Thread-scoped RAG:
  Each thread_id can have its own FAISS retriever built from an uploaded PDF.
- Tool calling:
  The LLM is bound to tools and can decide when to call them.
- Checkpointing:
  LangGraph checkpoints conversation state to sqlite, enabling thread persistence.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """
    Retrieve the FAISS-backed retriever for a given chat thread.

    Purpose:
    - This chatbot supports per-thread PDF ingestion. Each thread_id can have
      a dedicated retriever created by ingest_pdf().
    - This helper fetches the retriever from the in-memory store.

    Args:
        thread_id: The unique identifier for the conversation thread.

    Returns:
        - The retriever object if it exists for the given thread_id.
        - None if no retriever has been created for that thread.

    Notes:
    - Retrievers are stored in the module-level dictionary _THREAD_RETRIEVERS.
    - Thread IDs are always normalized to string keys for consistency.
    """
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Ingest a PDF (as raw bytes) and build a thread-scoped FAISS retriever.

    What this does:
    1) Writes the uploaded bytes to a temporary .pdf file
    2) Loads the PDF into Documents using PyPDFLoader
    3) Splits documents into semantically useful chunks using
       RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
    4) Embeds chunks using OpenAIEmbeddings
    5) Builds a FAISS vector store and converts it into a retriever
    6) Stores the retriever and basic metadata keyed by the provided thread_id

    Why thread-scoped:
    - In multi-chat setups (e.g., UI with multiple threads), each thread should
      query only the PDF it uploaded. This avoids cross-thread leakage of context.

    Args:
        file_bytes: Raw PDF bytes received from an upload endpoint/UI.
        thread_id: Unique conversation/thread identifier used to store retriever.
        filename: Optional user-facing filename to store in metadata. If omitted,
                  falls back to the temporary file basename.

    Returns:
        A summary dictionary intended for UI surfacing, containing:
        - filename: Resolved filename used for metadata display
        - documents: Number of loaded PDF pages/documents from PyPDFLoader
        - chunks: Number of chunks created by the text splitter

    Raises:
        ValueError: If file_bytes is empty or missing.

    Cleanup behavior:
    - The temporary PDF file is deleted in a finally block.
    - FAISS stores embedded text/chunks, so removing the temp file is safe
      after the store is created.

    Important:
    - This function stores retrievers in-memory. If your process restarts, the
      retriever will be lost unless you persist FAISS separately.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Calculator tool for basic arithmetic on two numbers.

    Purpose:
    - Exposes a small, deterministic arithmetic helper as a tool so the LLM can
      delegate exact numeric operations rather than doing mental math.

    Args:
        first_num: The first operand.
        second_num: The second operand.
        operation: Operation name as a short string:
            - "add" : addition
            - "sub" : subtraction
            - "mul" : multiplication
            - "div" : division (with division-by-zero protection)

    Returns:
        A dictionary with:
        - first_num, second_num, operation: Echoed inputs for traceability
        - result: The numeric result when successful
        OR
        - error: An error message string if something fails

    Error handling:
    - Returns {"error": "..."} for unsupported operations
    - Returns {"error": "Division by zero is not allowed"} if second_num == 0
    - Catches unexpected exceptions and returns them in "error"
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock quote for a given symbol using Alpha Vantage.

    Purpose:
    - Provides a simple market data lookup tool that the LLM can call when the
      user asks about current stock price/quote data.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "TSLA", "MSFT").

    Returns:
        The parsed JSON response (dict) returned by Alpha Vantage.
        Typically includes a "Global Quote" object when successful.

    Notes / Caveats:
    - Uses a hardcoded Alpha Vantage API key in the URL query string.
      In production, prefer using an environment variable (e.g., ALPHAVANTAGE_API_KEY)
      to avoid leaking secrets in code.
    - Alpha Vantage has rate limits. Frequent calls may return throttling messages.
    - This function does not validate symbol format; it simply forwards to the API.

    Security:
    - Consider moving the API key to .env and reading via os.getenv for safer handling.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant context from the uploaded PDF for the current thread.

    Purpose:
    - Enables Retrieval-Augmented Generation (RAG) over a user-uploaded PDF.
    - The retrieval is thread-scoped: it looks up the FAISS retriever that was built
      via ingest_pdf() for this specific thread_id.

    Args:
        query: The user's natural-language question or search query.
        thread_id: The current conversation thread id. This should be provided by
                   the calling application/config so the tool retrieves from the
                   correct per-thread document store.

    Returns:
        On success:
            {
              "query": <original query>,
              "context": [<top-k chunk text>...],
              "metadata": [<per-chunk metadata>...],
              "source_file": <filename associated with this thread's PDF>
            }

        If no document has been ingested for the thread:
            {
              "error": "No document indexed for this chat. Upload a PDF first.",
              "query": <original query>
            }

    Retrieval details:
    - Uses similarity search with k=4 (configured in ingest_pdf retriever creation).
    - The returned "context" can be provided back to the LLM for grounded answers.

    Implementation notes:
    - retriever.invoke(query) returns a list of Documents.
    - We extract page_content and metadata for transparency/debugging.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    """
    LangGraph state container for the chatbot.

    Fields:
        messages:
            A list of chat messages (Human/AI/System/Tool messages) accumulated
            through the conversation.

            The Annotated[...] with add_messages enables LangGraph's message
            accumulation behavior so each node can append new messages rather
            than overwriting history.
    """
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """
    Primary LLM node for the LangGraph workflow.

    Responsibilities:
    1) Extract thread_id from graph config (when provided by the caller)
    2) Construct a SystemMessage that instructs the model how to use tools:
       - If the question is about the uploaded PDF, call rag_tool with thread_id
       - Otherwise optionally use search/stock/calculator tools
       - If no document exists, ask user to upload a PDF
    3) Invoke the LLM (bound with tools) with the system + conversation messages
    4) Return the new AI response message to be appended into state

    Args:
        state: Current graph state, containing accumulated messages.
        config: Optional LangGraph config. Commonly includes:
            {
              "configurable": {
                "thread_id": "<some-id>"
              }
            }

    Returns:
        A dict with:
            {"messages": [response_message]}
        LangGraph will append this to existing messages because of add_messages.

    Notes:
    - This node does not call tools directly; it allows the LLM to decide.
    - Tool calling is handled by conditional edges + ToolNode in the graph.
    """
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    """
    Retrieve all known thread IDs from the SQLite checkpoint store.

    Purpose:
    - LangGraph checkpoints store conversation progress keyed by a configurable
      thread_id.
    - This helper scans all checkpoints and returns unique thread IDs so a UI
      can list available conversation threads.

    Returns:
        A list of unique thread_id values (strings).

    Notes:
    - checkpointer.list(None) iterates all checkpoints in storage.
    - Each checkpoint includes a config object containing configurable.thread_id.
    - This returns threads that have checkpoints, even if they do not have an
      ingested PDF document.
    """
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    """
    Check whether a given thread currently has an ingested PDF retriever.

    Purpose:
    - Lets a UI or API quickly determine whether RAG is available for a thread.

    Args:
        thread_id: The thread identifier to check.

    Returns:
        True if a retriever exists in _THREAD_RETRIEVERS for this thread_id,
        otherwise False.

    Notes:
    - This checks only in-memory state. If the process restarts, this will be False
      unless you re-ingest or persist and reload the vector stores.
    """
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    """
    Return metadata about the ingested document for a given thread.

    Purpose:
    - Provides UI-friendly details (filename, doc count, chunk count) for the
      PDF currently associated with a thread.

    Args:
        thread_id: The thread identifier.

    Returns:
        A dict containing metadata stored during ingest_pdf(), such as:
            {
              "filename": "...",
              "documents": <int>,
              "chunks": <int>
            }
        If no metadata exists, returns an empty dict.

    Notes:
    - This metadata is stored in the in-memory dictionary _THREAD_METADATA.
    - It does not validate that the retriever still exists, but typically both
      are set together in ingest_pdf().
    """
    return _THREAD_METADATA.get(str(thread_id), {})
