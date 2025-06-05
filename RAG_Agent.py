from dotenv import load_dotenv
import os
import pandas as pd
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_ollama import OllamaLLM, OllamaEmbeddings

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
import re

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001", temperature=1)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# llm = OllamaLLM(model="mistral", temperature=1)  
# embeddings = OllamaEmbeddings(model="nomic-embed-text")  

excel_path = "HourWiseData (4).xlsx"

if not os.path.exists(excel_path):
    raise FileNotFoundError(f"Excel file not found: {excel_path}")

# Load Excel as DataFrame
df = pd.read_excel(excel_path)
df.columns = [col.lower().strip() for col in df.columns]  # Normalize column names

df["text"] = df.apply(lambda row: f"Date: {row['date']}, Region: {row['region']}, Area: {row['area']}, Site: {row['site name']}, Hour 10–21: {[row[str(h)] for h in range(10, 22)]}, Total Visitors: {row['total unique groups']}", axis=1)
df_loader = DataFrameLoader(df, page_content_column="text")

try:
    docs = df_loader.load()
    print(f"Excel file loaded with {len(docs)} rows")
except Exception as e:
    print(f"Error loading Excel: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs_split = text_splitter.split_documents(docs)

persist_directory = r"C:\\Users\\anujn\\OneDrive\\Desktop\\excel_RAG\\chroma_db"
collection_name = "client_data"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        documents=docs_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 30}
)

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)

#######################
#######################

import pandas as pd
import re

@tool
def data_calculation_tool(query: str) -> str:
    """
    Parses a user query to calculate sum, max, min, or average
    of 'total unique groups' for a specified 'site name'.

    Args:
        query: Natural language string like
               "Find the total visitors for site Inorbit mall for the whole month."

    Returns:
        Result string or error message.
    """
    # # Load and prepare data
    # df = pd.read_excel("HourWiseData (4).xlsx")
    # df.columns = [col.lower().strip() for col in df.columns]

    visitor_col = "total unique groups"
    site_col = "site name"

    # Parse site name
    site_match = re.search(r"site ([\w\s]+)", query.lower())
    site_name = site_match.group(1).strip() if site_match else None

    # Parse operation
    op_map = {
        "total": "sum",
        "sum": "sum",
        "average": "mean",
        "avg": "mean",
        "maximum": "max",
        "minimum": "min",
        "max": "max",
        "min": "min",
    }
    op = next((op_map[key] for key in op_map if key in query.lower()), None)

    if not site_name or not op:
        return "Please include both an operation (e.g., total, max) and a site name in your query."

    # Filter rows for the site
    filtered_df = df[df[site_col].str.lower().str.strip() == site_name.lower()]
    if filtered_df.empty:
        return f"No data found for site '{site_name}'."

    # Compute result
    try:
        result = getattr(filtered_df[visitor_col], op)()
        return f"{op.capitalize()} of visitors for site '{site_name}': {result}"
    except Exception as e:
        return f"Error computing result: {e}"


###############################################################################
###############################################################################

@tool
def exact_match_tool(query: str) -> str:
    """
    Directly searches the Excel data for specific date and site matches using pandas.
    Supports queries like:
    - "Visitors for Inorbit Mall on 30-05-2025"
    - "Total visitors on 01-06-2025"
    - "Hourly data for site XYZ on 03-06-2025"
    """

    query = query.lower()
    matched_rows = df.copy()

    # Try to extract date in DD-MM-YYYY format
    date_match = re.search(r"\b(\d{2}-\d{2}-\d{4})\b", query)
    if date_match:
        date_val = date_match.group(1)
        matched_rows = matched_rows[matched_rows['date'] == date_val]

    # Try to extract site name using known values
    for site in df['site name'].unique():
        if site.lower() in query:
            matched_rows = matched_rows[matched_rows['site name'].str.lower() == site.lower()]
            break

    if matched_rows.empty:
        return "No exact match found for the given date or site."

    # Format the matched data into a readable string
    results = []
    for _, row in matched_rows.iterrows():
        hourly_data = ", ".join([f"{h}: {row[str(h)]}" for h in range(10, 22) if str(h) in row])
        result = (f"Date: {row['date']}\n"
                  f"Area: {row['area']}\n"
                  f"Region: {row['region']}\n"
                  f"Site: {row['site name']}\n"
                  f"Hourly Visitors (10–21): {hourly_data}\n"
                  f"Total Unique Groups: {row['total unique groups']}")
        results.append(result)

    return "\n\n".join(results)

############################################################################
############################################################################

@tool
def top_performing_store_tool(query: str) -> str:
    """
    Finds the top performing store (based on total unique groups) for a specified region or area.
    
    Example queries:
    - "Show me the top performing store from Karnataka"
    - "Top store in Hyderabad"
    """

    region_match = re.search(r"(?:region|state)\s+(?:of\s+)?([\w\s]+)", query.lower())
    area_match = re.search(r"(?:area|city)\s+(?:of\s+)?([\w\s]+)", query.lower())

    region = region_match.group(1).strip() if region_match else None
    area = area_match.group(1).strip() if area_match else None

    filtered_df = df.copy()

    if region:
        filtered_df = filtered_df[filtered_df["region"].str.lower() == region.lower()]
    if area:
        filtered_df = filtered_df[filtered_df["area"].str.lower() == area.lower()]

    if filtered_df.empty:
        return "No data found for the specified region or area."

    group_by_col = "site name"
    result_df = (
        filtered_df.groupby(group_by_col)["total unique groups"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    top_row = result_df.iloc[0]
    store = top_row[group_by_col]
    total = top_row["total unique groups"]

    return f"🏆 Top performing store is **{store}** with a total of **{total}** visitors."

###############################################################################
###############################################################################

tools = [exact_match_tool, top_performing_store_tool, data_calculation_tool]

tools_dict = {our_tool.name: our_tool for our_tool in tools}

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are a helpful and intelligent AI assistant designed to answer questions based on an Excel document containing hour-wise visitor counts for various sites/stores across different dates for the previous month.

Each row in the dataset represents a unique record with the following information:
- `date` (in DD-MM-YYYY format)
- `site name` name
- 'region' (representing the state where the site/store is based) like Karnataka, Telangana, Goa, etc.
- 'area' (representing cities where the sit/store is based) like Bangalore, Hyderabad, Chennai, etc.
- Hourly visitor counts across columns `10` to `21` (representing each hour of the day)
- Total unique groups (daily count)


You can answer questions like:
- "How many visitors were there on 01-05-2025?"
- "Show hour-wise visitors for 03-06-2025"
- "Total unique groups for site ABC on 02-05-2025"
- "Calculate the total visitors for Inorbit mall for the entire month"
- "Give me the best performing store in Karnataka region on the basis of total unique groups"

Use the data-calculation_tool to calculate sum of the visitor counts across multiple rows or dates or for the whole month.

Use the  exact_match_tool to find relevant information from the document. If the question involves a specific date or site, ensure you extract and match those values precisely.

Use the top_performing_store_tool to find the top performing store across one or more regions

Be concise, numeric when needed, and cite exact values from the table. If the data is not available or ambiguous, politely indicate so.

If the user query mentions both a date and a site, filter based on both.

Your job is to provide accurate and direct answers grounded in the data — not to guess or infer beyond the available values.

"""

def call_llm(state: AgentState) -> AgentState:
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}

def take_action(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        if not t['name'] in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})
        
        last_message = result['messages'][-1]
        
        # Safely print result whether it's a message object or string
        if hasattr(last_message, 'content'):
            print("\n=== ANSWER ===")
            print(last_message.content)
        else:
            print("\n=== ANSWER ===")
            print(last_message)

running_agent()
