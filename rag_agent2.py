# excel_rag_agent.py
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
import pandas as pd
from langchain_core.documents import Document

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001", temperature=0.1
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

excel_path = "HourWiseData (4).xlsx"

if not os.path.exists(excel_path):
    raise FileNotFoundError(f"Excel file not found: {excel_path}")

# Load Excel with Pandas and convert each row into a Document with descriptive context
def load_excel_as_documents(filepath):
    df = pd.read_excel(filepath, sheet_name=0)
    df.fillna("", inplace=True)
    documents = []

    # Define static structure context
    static_columns = list(df.columns)
    context_header = "The Excel sheet contains the following columns: " + \
        ", ".join(static_columns) + ". Each row contains hourly data per site."

    for idx, row in df.iterrows():
        row_text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        content = context_header + "\n\nRow data:\n" + row_text
        metadata = {"row_index": idx}
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

try:
    documents = load_excel_as_documents(excel_path)
    print(f"Excel loaded and parsed into {len(documents)} documents")
except Exception as e:
    print(f"Error processing Excel: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000
)

chunks = text_splitter.split_documents(documents)

persist_directory = "./excel_chroma_db"
collection_name = "excel_hourly_data"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Chroma vectorstore created for Excel!")
except Exception as e:
    print(f"Error creating ChromaDB: {e}")
    raise

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 100}
)

@tool
def excel_retriever_tool(query: str) -> str:
    """
    Retrieves relevant row-wise context from the Excel file based on user query.
    Each row includes descriptive column names for clarity.
    Use only for unique entries.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant data found in the Excel file."

    results = [f"Document {i+1} (Row {doc.metadata.get('row_index')}):\n{doc.page_content}" for i, doc in enumerate(docs)]
    return "\n\n".join(results)

llm = llm.bind_tools([excel_retriever_tool])

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent assistant that answers questions about data in the Excel file loaded into your memory.
The structure of the data includes columns such as date, region, area, site code, site name, and hourly in-counts from 0 to 23, along with group statistics.
Use the retriever tool to look up relevant rows and cite the specific values clearly. Use the retriever tool only for unique entries.
You are allowed to call multiple tools if required to answer the question.
"""

tools_dict = {excel_retriever_tool.name: excel_retriever_tool}

def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
    response = llm.invoke(messages)
    return {'messages': [response]}

def take_action(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        if t['name'] not in tools_dict:
            result = "Invalid tool called."
        else:
            result = tools_dict[t['name']].invoke(t['args'])
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    return {'messages': results}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

def run_excel_agent():
    print("\n=== Excel RAG AGENT ===")
    while True:
        user_input = input("\nAsk a question about the Excel data (or type 'exit'): ")
        if user_input.lower() in ['exit', 'quit']:
            break
        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


if __name__ == "__main__":
    run_excel_agent()
