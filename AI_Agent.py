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
from langchain_core.documents import Document
import datetime


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

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# docs_split = text_splitter.split_documents(docs)

# persist_directory = r"C:\\Users\\anujn\\OneDrive\\Desktop\\excel_RAG\\chroma_db"
# collection_name = "client_data"

# if not os.path.exists(persist_directory):
#     os.makedirs(persist_directory)

# try:
#     vectorstore = Chroma.from_documents(
#         documents=docs_split,
#         embedding_function=embeddings,
#         persist_directory=persist_directory,
#         collection_name=collection_name
#     )
#     print(f"Created ChromaDB vector store!")
# except Exception as e:
#     print(f"Error setting up ChromaDB: {str(e)}")
#     raise

# retriever = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 30}
# )


if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)


###############################################################################
###############################################################################

@tool
def smart_data_lookup_tool(query: str) -> str:
    """
    Answers data lookup queries like:
    - 'Visitors in Inorbit mall at 10:00'
    - 'Total visitors for Inorbit mall on 03-05-2025'
    - 'Total visitors for Inorbit mall for the whole month'
    - 'Total visitors in Karnataka'
    - 'Total visitors in Telangana on 03-05-2025'
    """

    import re
    from datetime import datetime

    site_col = "site name"
    region_col = "region"
    date_col = "date"
    group_col = "total unique groups"
    hour_cols = [str(h) for h in range(10, 22)]

    query = query.lower()

    site = None
    region = None
    date_val = None
    hour = None

    # Match date (dd-mm-yyyy or 3rd May 2025)
    date_match = re.search(r"\b(\d{1,2})[-/\s](\d{1,2})[-/\s](\d{4})\b", query)
    if date_match:
        try:
            day, month, year = map(int, date_match.groups())
            date_val = datetime(year, month, day).strftime("%d-%m-%Y")
        except:
            pass
    else:
        # "3rd May 2025" format
        spoken_date = re.search(r"\b(\d{1,2})(st|nd|rd|th)?\s+([a-zA-Z]+)\s+(\d{4})\b", query)
        if spoken_date:
            day = int(spoken_date.group(1))
            month_str = spoken_date.group(3)
            year = int(spoken_date.group(4))
            try:
                month = datetime.strptime(month_str[:3], '%b').month
                date_val = datetime(year, month, day).strftime("%d-%m-%Y")
            except:
                pass

    # Extract hour (e.g. 10 am)
    hour_match = re.search(r"(?:at\s*)?(\d{1,2})\s*(am)?", query)
    if hour_match:
        hour = int(hour_match.group(1))
        if hour < 10:  # likely 10-21 are valid hours only
            hour += 12
        if str(hour) not in hour_cols:
            return f"Hour {hour} not tracked. Valid range: 10 to 21."

    # Match known site names
    site = next((s for s in df[site_col].unique() if s.lower() in query), None)

    # Match region
    region_match = re.search(r"in\s+([\w\s]+)", query)
    if region_match:
        possible_region = region_match.group(1).strip().title()
        if possible_region in df[region_col].unique():
            region = possible_region

    # Start filtering
    filtered = df.copy()
    if site:
        filtered = filtered[filtered[site_col].str.lower() == site.lower()]
    if region:
        filtered = filtered[filtered[region_col] == region]
    if date_val:
        filtered = filtered[filtered[date_col] == date_val]

    if filtered.empty:
        return "No data found matching your filters."

    # Hourly visitor count
    if hour:
        if str(hour) in filtered.columns:
            total = filtered[str(hour)].sum()
            return f"Visitors at {hour}:00 in {site if site else region}{' on ' + date_val if date_val else ''}: {int(total)}"

    # Total unique group count
    if group_col in filtered.columns:
        total = filtered[group_col].sum()

        context_parts = []
        if site:
            context_parts.append(f"{site}")
        elif region:
            context_parts.append(f"{region}")
        else:
            context_parts.append("all stores")

        if date_val:
            context_parts.append(f"on {date_val}")
        else:
            context_parts.append("for the whole month")

        return f"Total unique groups for {' '.join(context_parts)}: {int(total)}"

    return "Unable to compute the requested result."

############################################################################
############################################################################

@tool
def top_n_stores_by_region(query: str) -> str:
    """
    Returns the top N best performing stores based on total unique groups.
    Supports both:
    - "Top 5 stores" (general)
    - "Top 5 stores in Karnataka" (region-specific)
    """

    import re

    # Extract N from the query, default to 5
    n = 5
    n_match = re.search(r"top\s+(\d+)", query.lower())
    if n_match:
        try:
            n = int(n_match.group(1))
        except ValueError:
            pass

    # Try to extract region
    region_match = re.search(r"in\s+([\w\s]+)", query.lower())
    region = region_match.group(1).strip() if region_match else None

    store_col = "site name"
    group_col = "total unique groups"
    region_col = "region"

    if store_col not in df.columns or group_col not in df.columns:
        return "Necessary columns not found in the dataset."

    filtered_df = df.copy()

    # Filter by region if specified
    if region:
        filtered_df = filtered_df[filtered_df[region_col].str.lower() == region.lower()]
        if filtered_df.empty:
            return f"No data found for region '{region}'."

    try:
        top_df = (
            filtered_df.groupby(store_col)[group_col]
            .sum()
            .sort_values(ascending=False)
            .head(n)
            .reset_index()
        )

        header = f"Top {n} performing stores in {region}" if region else f"Top {n} performing stores across India"
        result = "\n".join(
            [f"{i+1}. {row[store_col]} – {row[group_col]} unique groups"
             for i, row in top_df.iterrows()]
        )

        return f"{header}:\n\n{result}"

    except Exception as e:
        return f"Error computing top stores: {e}"
    
####################################################################
####################################################################

@tool
def bottom_n_stores_by_region(query: str) -> str:
    """
    Returns the bottom N worst performing stores based on total unique groups.
    Supports both:
    - "Bottom 5 stores"
    - "Bottom 5 stores in Karnataka"
    """

    import re

    # Default number of stores to return
    n = 5
    n_match = re.search(r"bottom\s+(\d+)", query.lower())
    if n_match:
        try:
            n = int(n_match.group(1))
        except ValueError:
            pass

    # Extract region (if any)
    region_match = re.search(r"in\s+([\w\s]+)", query.lower())
    region = region_match.group(1).strip() if region_match else None

    store_col = "site name"
    group_col = "total unique groups"
    region_col = "region"

    if store_col not in df.columns or group_col not in df.columns:
        return "Required columns not found in the dataset."

    filtered_df = df.copy()

    # Filter by region if specified
    if region:
        filtered_df = filtered_df[filtered_df[region_col].str.lower() == region.lower()]
        if filtered_df.empty:
            return f"No data found for region '{region}'."

    try:
        bottom_df = (
            filtered_df.groupby(store_col)[group_col]
            .sum()
            .sort_values(ascending=True)  # Ascending for worst performance
            .head(n)
            .reset_index()
        )

        header = f"Bottom {n} performing stores in {region}" if region else f"Bottom {n} performing stores across India"
        result = "\n".join(
            [f"{i+1}. {row[store_col]} – {row[group_col]} unique groups"
             for i, row in bottom_df.iterrows()]
        )

        return f"{header}:\n\n{result}"

    except Exception as e:
        return f"Error computing bottom stores: {e}"


#####################################################################
#####################################################################

@tool
def average_total_in_count_tool(query: str) -> str:
    """
    Computes average or total in-count for various time and location filters.

    Supports:
    - Specific site or area on a single date.
    - Site or region across a month.
    - Region across a date range.
    - Week-based queries (e.g., "second week of May 2025").
    """
    import re
    from datetime import datetime, timedelta

    query = query.lower()
    site_col = "site name"
    area_col = "area"
    region_col = "region"
    date_col = "date"
    count_col = "total in count"

    site = None
    area = None
    region = None
    start_date = None
    end_date = None
    month = None
    year = None

    # Try to match site
    for s in df[site_col].dropna().unique():
        if s.lower() in query:
            site = s
            break

    # Try to match area
    for a in df[area_col].dropna().unique():
        if a.lower() in query:
            area = a
            break

    # Try to match region
    for r in df[region_col].dropna().unique():
        if r.lower() in query:
            region = r
            break

    # Match date range (e.g., 1st May to 7th May)
    date_range_match = re.search(r"(\\d{1,2})(?:st|nd|rd|th)?\\s+([a-z]+)\\s*(?:to|through|–|-)\\s*(\\d{1,2})(?:st|nd|rd|th)?\\s+([a-z]+)?\\s*(\\d{4})", query)
    if date_range_match:
        d1, m1, d2, m2, y = date_range_match.groups()
        try:
            m1 = datetime.strptime(m1[:3], "%b").month
            m2 = datetime.strptime(m2[:3], "%b").month if m2 else m1
            start_date = datetime(int(y), m1, int(d1))
            end_date = datetime(int(y), m2, int(d2))
        except:
            return "Could not parse date range."

    # Match "second week of May 2025"
    week_match = re.search(r"(first|second|third|fourth|fifth) week of (\\w+) (\\d{4})", query)
    if week_match:
        week_word, month_word, year_str = week_match.groups()
        week_index = ["first", "second", "third", "fourth", "fifth"].index(week_word)
        try:
            month = datetime.strptime(month_word[:3], "%b").month
            year = int(year_str)
            first_day = datetime(year, month, 1)
            start_date = first_day + timedelta(days=week_index * 7)
            end_date = start_date + timedelta(days=6)
        except:
            return "Could not parse week-based date."

    # Match month + year (e.g., May 2025)
    if not start_date and not end_date:
        month_year_match = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)\\s+(\\d{4})", query)
        if month_year_match:
            month_str, year = month_year_match.groups()
            try:
                month = datetime.strptime(month_str[:3], "%b").month
                year = int(year)
                start_date = datetime(year, month, 1)
                # set end date to last day of the month
                next_month = datetime(year, month, 28) + timedelta(days=4)
                end_date = datetime(next_month.year, next_month.month, 1) - timedelta(days=1)
            except:
                return "Could not parse month-year format."

    # Default: no filters = all data
    filtered = df.copy()

    if site:
        filtered = filtered[filtered[site_col].str.lower() == site.lower()]
    if area:
        filtered = filtered[filtered[area_col].str.lower() == area.lower()]
    if region:
        filtered = filtered[filtered[region_col].str.lower() == region.lower()]
    if start_date and end_date:
        filtered = filtered[(df[date_col] >= start_date) & (df[date_col] <= end_date)]

    if filtered.empty:
        return "No matching data found."

    avg = filtered[count_col].astype(float).mean()
    total = filtered[count_col].astype(float).sum()

    context = []
    if site:
        context.append(site)
    elif area:
        context.append(area)
    elif region:
        context.append(region)
    else:
        context.append("all sites")

    if start_date and end_date:
        context.append(f"from {start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')}")
    elif month and year:
        context.append(f"in {datetime(year, month, 1).strftime('%B %Y')}")

    context_str = " ".join(context)

    return f"✅ For {context_str}, the average total in-count is {avg:.2f} and the total in-count is {int(total)}."


########################################################################
########################################################################

tools = [smart_data_lookup_tool, top_n_stores_by_region, bottom_n_stores_by_region, average_total_in_count_tool]

tools_dict = {our_tool.name: our_tool for our_tool in tools}

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are a smart and helpful AI assistant designed to analyze and summarize visitor traffic data for US Polo stores across India. The data is structured as an Excel table where each row corresponds to a specific date and store, with columns representing:

- `date` (in DD-MM-YYYY format)
- `site name` (the store name)
- `region` (state such as Karnataka, Telangana, etc.)
- `area` (city or locality such as Bangalore, Hyderabad, etc.)
- Hour-wise visitor counts in columns `10` to `21` (representing 10 AM to 9 PM)
- `total unique groups` (total number of unique visitor groups on that day)

You have access to the following tools:

---

🔧 **smart_data_lookup_tool**

Use this for precise data lookup and total visitor calculations. It can handle queries like:
- “Visitors in Inorbit mall at 10 am”
- “Total visitors for MOA on 03-05-2025”
- “Total visitors for EA Mall for the whole month”
- “Total visitors in Telangana on 1st June 2025”
- “Total visitors in Karnataka”

This tool supports filters on site name, region, date, and hour.

---

📈 **top_n_stores_by_region**

Use this tool to find the top N best-performing stores based on total unique groups. It supports:
- “Top 5 stores in Karnataka”
- “Top 10 stores across India”
- “Which stores performed best in Telangana?”

If no region is specified, it returns national-level top stores.

---

📉 **bottom_n_stores_by_region**

Use this tool to find the worst performing stores based on total unique groups. It supports:
- “Bottom 5 stores across India”
- “Worst 3 stores in Goa”
- “Least visited stores in Telangana”

If no region is specified, it ranks all-India.

---

📊 **average_total_in_count_tool**

Use this tool when the query asks for **average or total in-count** over any of the following:
- For a **specific site, area, or region**
- For a **specific date or date range**
- For a **specific month and year**
- For a **specific week within a month**

It supports queries like:
- “What is the average total in-count for all sites in Bangalore on 01-05-2025?”
- “Give me the average total visitors for Inorbit mall for May 2025”
- “Calculate the average total in count for Inorbit mall from 1st May to 7th May”
- “What is the average in count for Inorbit mall for the second week of May 2025?”
- “Give me the average total visitors for Karnataka stores for May 2025”
- “What is the average in-count across all sites in May 2025?”
- “Give me the total in-count for Telangana for the first week of June 2025”

---

📌 Guidance:
- If the question involves **specific time, store, region, or date**, prefer `smart_data_lookup_tool`.
- If it involves **ranking stores by performance**, use either the **top_n_stores_by_region** or **bottom_n_stores_by_region**.
- If the question involves **average footfall per site in a city on a date**, use `average_total_in_count_tool`.
- Always provide numerical answers clearly and concisely.
- If the input is ambiguous, politely ask the user for more details (e.g., "Which region or store are you referring to?").

Do not guess. Only respond with information found or calculated from the actual Excel data.


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

def log_to_memory_bot(user_input, ai_response):
    import subprocess

    with open("logging.txt", "a", encoding="utf-8") as f:
        f.write(f"Human: {user_input}\n")
        f.write(f"AI: {ai_response}\n\n")

def running_agent():
    print("\n=== RAG AGENT ===")

    conversation_history = []

    while True:
        user_input = input("\nWhat is your question (Type 'exit' to exit): ")
        if user_input.lower() in ['exit', 'quit']:
            break

        # Add user input to history
        conversation_history.append(HumanMessage(content=user_input))

        # Call agent with full history
        result = rag_agent.invoke({"messages": conversation_history})
        last_message = result['messages'][-1]

        # Extract AI response and add to history
        ai_response = last_message.content if hasattr(last_message, 'content') else str(last_message)
        conversation_history.append(last_message)

        # Display and log
        print("\n=== ANSWER ===")
        print(ai_response)

        log_to_memory_bot(user_input, ai_response)


running_agent()
