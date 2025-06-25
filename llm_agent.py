# llm_agent.py
import os
import requests
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
# Import BaseModel and Field directly from pydantic for Pydantic V2 compatibility
from pydantic import BaseModel, Field
from typing import Optional, Dict, List

# LangChain specific imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # Keep OpenAIEmbeddings for RAG
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# RAG specific imports
# from langchain_community.vectorstores import Chroma # Using ChromaDB for the vector store
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
VECTOR_DB_PATH = "./vector_db" # Path to your ChromaDB persistent directory

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables or .env file.")

# --- Define Pydantic models for Tool Inputs (matching your FastAPI request models) ---
class LogQueryInput(BaseModel):
    log_group_name: str = Field(description="The name of the CloudWatch log group (e.g., '/aws/lambda/my-function').")
    start_time_iso: str = Field(description="Start time in ISO 8601 format (e.g., '2024-01-01T00:00:00Z').")
    end_time_iso: str = Field(description="End time in ISO 8601 format (e.g., '2024-01-01T01:00:00Z').")
    query_string: Optional[str] = Field(None, description="Optional CloudWatch Logs Insights query string (e.g., 'filter @message like /Error/').")
    limit: int = Field(20, description="Maximum number of log events to return.")

class MetricRequestInput(BaseModel):
    namespace: str = Field(description="The CloudWatch metric namespace (e.g., 'AWS/EC2').")
    metric_name: str = Field(description="The name of the metric (e.g., 'CPUUtilization').")
    start_time_iso: str = Field(description="Start time in ISO 8601 format (e.g., '2024-01-01T00:00:00Z').")
    end_time_iso: str = Field(description="End time in ISO 8601 format (e.g., '2024-01-01T01:00:00Z').")
    dimensions: Optional[Dict[str, str]] = Field(None, description="Optional metric dimensions (e.g., {'InstanceId': 'i-1234567890abcdef0'}).")
    statistic: str = Field('Average', description="The statistic to retrieve (e.g., 'Average', 'Sum', 'Maximum', 'Minimum', 'SampleCount').")
    period_seconds: int = Field(300, description="The granularity of the returned data points in seconds (e.g., 300 for 5 minutes).")

class TextAnalysisInput(BaseModel):
    text_data: str = Field(description="The large block of text data to analyze.")
    patterns: List[str] = Field(["error", "failure", "exception", "timeout", "denied", "slow"], description="List of keywords or patterns to search for (case-insensitive).")

class TroubleshootingInput(BaseModel):
    incident_summary: str = Field(description="A summary of the incident or problem requiring troubleshooting.")

# New Pydantic model for the RAG tool input
class KnowledgeBaseQueryInput(BaseModel):
    query: str = Field(description="The question or topic to search for in the internal knowledge base.")

# New Pydantic model for the Log Similarity Search tool input
class LogSimilaritySearchInput(BaseModel):
    query: str = Field(description="A query string to find similar log messages (e.g., 'Lambda timeout errors').")
    k: int = Field(default=5, description="The number of most similar log messages to return.")

# --- Define LangChain Tools that call your FastAPI endpoints ---

@tool(args_schema=LogQueryInput)
def query_aws_logs(
    log_group_name: str,
    start_time_iso: str,
    end_time_iso: str,
    query_string: Optional[str] = None,
    limit: int = 20
) -> str:
    """
    Queries AWS CloudWatch Logs for events within a specified log group and time range.
    Can apply an optional CloudWatch Logs Insights query string for filtering.
    Returns the log events found.
    Example `query_string`: 'filter @message like /Error/'
    Example `log_group_name`: '/aws/lambda/my-function'
    Example `start_time_iso` and `end_time_iso`: '2024-01-01T00:00:00Z'
    """
    payload = {
        "log_group_name": log_group_name,
        "start_time_iso": start_time_iso,
        "end_time_iso": end_time_iso,
        "query_string": query_string,
        "limit": limit
    }
    try:
        response = requests.post(f"{MCP_SERVER_URL}/query-logs", json=payload)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return json.dumps(response.json(), indent=2)
    except requests.exceptions.RequestException as e:
        return f"Error calling query_aws_logs API: {e}"

@tool(args_schema=MetricRequestInput)
def get_aws_metric_statistics(
    namespace: str,
    metric_name: str,
    start_time_iso: str,
    end_time_iso: str,
    dimensions: Optional[Dict[str, str]] = None,
    statistic: str = 'Average',
    period_seconds: int = 300
) -> str:
    """
    Retrieves AWS CloudWatch metric statistics for a given metric.
    Specify the namespace (e.g., 'AWS/EC2'), metric_name (e.g., 'CPUUtilization'),
    time range, and optionally dimensions and statistic type.
    Example `dimensions`: {'InstanceId': 'i-1234567890abcdef0'}
    """
    payload = {
        "namespace": namespace,
        "metric_name": metric_name,
        "start_time_iso": start_time_iso,
        "end_time_iso": end_time_iso,
        "dimensions": dimensions,
        "statistic": statistic,
        "period_seconds": period_seconds
    }
    try:
        response = requests.post(f"{MCP_SERVER_URL}/get-metrics", json=payload)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    except requests.exceptions.RequestException as e:
        return f"Error calling get_aws_metric_statistics API: {e}"

@tool(args_schema=TextAnalysisInput)
def analyze_text_for_patterns(text_data: str, patterns: Optional[List[str]] = None) -> str:
    """
    Analyzes a given block of text to find lines that match specified patterns or keywords.
    Useful for extracting relevant information from large log outputs or reports.
    Default patterns include "error", "failure", "exception", "timeout", "denied", "slow".
    """
    payload = {
        "text_data": text_data,
        "patterns": patterns if patterns is not None else ["error", "failure", "exception", "timeout", "denied", "slow"]
    }
    try:
        response = requests.post(f"{MCP_SERVER_URL}/analyze-text", json=payload)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    except requests.exceptions.RequestException as e:
        return f"Error calling analyze_text_for_patterns API: {e}"

@tool(args_schema=TroubleshootingInput)
def propose_aws_troubleshooting_steps(incident_summary: str) -> str:
    """
    Provides general AWS troubleshooting steps based on an incident summary.
    This tool gives high-level advice, not specific log/metric queries.
    """
    payload = {
        "incident_summary": incident_summary
    }
    try:
        response = requests.post(f"{MCP_SERVER_URL}/troubleshooting-steps", json=payload)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    except requests.exceptions.RequestException as e:
        return f"Error calling propose_aws_troubleshooting_steps API: {e}"

# --- New RAG Tool ---
@tool(args_schema=KnowledgeBaseQueryInput)
def knowledge_base_query(query: str) -> str:
    """
    Searches the internal AWS knowledge base (documentation, best practices, troubleshooting guides)
    to answer conceptual questions. Use this when the user asks about how to do something,
    what a service is, common errors, or best practices, rather than asking for live data.
    """
    try:
        # Initialize embeddings - Using OpenAIEmbeddings as requested
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

        # Load the existing vectorstore
        if not os.path.exists(VECTOR_DB_PATH):
            return f"Error: Knowledge base not found at {VECTOR_DB_PATH}. Please run ingest_docs.py first."

        vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

        # Perform retrieval
        docs = retriever.invoke(query)

        if not docs:
            return "No relevant information found in the knowledge base for this query."

        # Prepare context for the LLM
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Use a sub-LLM call for RAG generation
        rag_llm = ChatOpenAI(model="gpt-4.1.nano", temperature=0, api_key=OPENAI_API_KEY)
        
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant. Use the following retrieved context to answer the user's question concisely and accurately. If the answer is not in the context, state that you don't have enough information."),
            ("human", "Question: {query}\n\nContext:\n{context}")
        ])
        rag_chain = rag_prompt | rag_llm

        response = rag_chain.invoke({"query": query, "context": context})
        return response.content

    except Exception as e:
        return f"Error performing knowledge base query: {e}. Ensure the vector database is initialized and accessible and embedding model is correctly configured. Details: {e}"
    

# --- NEW Tool for Log Similarity Search ---
@tool(args_schema=LogSimilaritySearchInput)
def search_similar_log_messages(query: str, k: int = 5) -> str:
    """
    Searches the stored log embeddings for log messages semantically similar to a given query.
    Useful for finding correlated events, anomalies, or specific error patterns across a large volume of ingested logs.
    Provide a clear 'query' describing the type of log message you are looking for, and 'k' for the number of results.
    """
    payload = {
        "query": query,
        "k": k
    }
    try:
        response = requests.post(f"{MCP_SERVER_URL}/search-similar-logs", json=payload)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    except requests.exceptions.RequestException as e:
        return f"Error calling search_similar_log_messages API: {e}"

# --- Initialize Main LLM and Agent ---
# Using the OpenAI API key from environment variables
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0, api_key=OPENAI_API_KEY) 

# List all tools available to the agent, including the new RAG tool
tools = [
    query_aws_logs,
    get_aws_metric_statistics,
    analyze_text_for_patterns,
    propose_aws_troubleshooting_steps,
    knowledge_base_query # Add your new RAG tool here!
]

# Define the prompt for the agent (UPDATED TO MENTION THE NEW TOOL)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant specialized in AWS observability and troubleshooting.
    You can query AWS CloudWatch logs and metrics, analyze text data, propose troubleshooting steps,
    and answer conceptual questions from an internal AWS knowledge base using the `knowledge_base_query` tool.
    
    When asked about AWS logs or metrics, you MUST ask for the specific 'log_group_name' (for logs), 
    'namespace' and 'metric_name' (for metrics), and 'start_time_iso' and 'end_time_iso' in ISO 8601 format (e.g., '2024-01-01T00:00:00Z'). 
    If the user does not provide enough information for a tool, ask clarifying questions to get the required parameters.

    For time ranges, if the user specifies relative times (e.g., "last 5 minutes", "yesterday"),
    calculate the exact ISO 8601 timestamps before calling the tool.

    Use the `knowledge_base_query` tool to answer questions about AWS services, best practices,
    common errors, or "how-to" questions that can be found in documentation.
    
    Always provide clear, concise, and actionable information.
    If a query returns no data or no relevant knowledge, inform the user about it.
    """),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Main Interaction Loop ---
if __name__ == "__main__":
    print("Welcome to the AWS Observability AI Assistant!")
    print("You can ask me to query logs, get metrics, analyze text, propose troubleshooting steps, or ask conceptual AWS questions.")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nYour query: ")
        if user_input.lower() == 'exit':
            break

        try:
            # Simple date parsing for relative times (can be more sophisticated)
            # The LLM is generally good at figuring out dates if context is given,
            # but explicit conversion helps ensure correct ISO format for tools.
            current_utc_time = datetime.utcnow()
            user_input_with_time_context = f"{user_input} (Current UTC time: {current_utc_time.isoformat(timespec='seconds')}Z)"

            print(f"\nProcessing query: {user_input_with_time_context}") # Added for better clarity
            response = agent_executor.invoke({"input": user_input_with_time_context})
            print("\nAI Assistant:")
            print(response["output"])
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure your FastAPI server is running and accessible at " + MCP_SERVER_URL)
            print("Also check your OPENAI_API_KEY and AWS credentials.")
