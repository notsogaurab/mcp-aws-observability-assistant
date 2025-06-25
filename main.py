from fastapi import FastAPI, HTTPException
from contextlib2 import asynccontextmanager
from pydantic import BaseModel, Field
import boto3
from datetime import datetime, timedelta
import os
import time
import json
from typing import Optional, Dict, List

# New imports for Vector DB and Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI(title="AWS Observability API Server")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Initialize Embedding Model and Vector Store for Logs ---
# This will be initialized once when the server starts
embeddings_model = None
log_vectorstore = None
LOG_VECTOR_DB_PATH = "./log_vector_db"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embeddings_model, log_vectorstore
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Required for log embeddings.")

    print("Initializing OpenAI Embeddings for log processing...")
    embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    print(f"Initializing ChromaDB for log embeddings at {LOG_VECTOR_DB_PATH}...")
    log_vectorstore = Chroma(persist_directory=LOG_VECTOR_DB_PATH, embedding_function=embeddings_model)
    print("Log embedding vector store initialized.")

# Pydantic models for request/response
class LogQueryRequest(BaseModel):
    log_group_name: str
    start_time_iso: str
    end_time_iso: str
    query_string: Optional[str] = None
    limit: int = 20

class MetricRequest(BaseModel):
    namespace: str
    metric_name: str
    start_time_iso: str
    end_time_iso: str
    dimensions: Optional[Dict[str, str]] = None
    statistic: str = 'Average'
    period_seconds: int = 300

class TextAnalysisRequest(BaseModel):
    text_data: str
    patterns: List[str] = ["error", "failure", "exception", "timeout", "denied", "slow"]

class TroubleshootingRequest(BaseModel):
    incident_summary: str

# New Pydantic models for log embedding features
class LogIngestionRequest(BaseModel):
    logs: List[str] = Field(description="A list of log messages to embed and store.")

class LogSimilaritySearchRequest(BaseModel):
    query: str = Field(description="The text query to search for similar log messages.")
    k: int = Field(default=5, description="The number of most similar log messages to return.")

@app.get("/")
async def read_root():
    return {"message": "AWS Observability API Server is running"}

@app.post("/query-logs")
async def query_aws_logs(request: LogQueryRequest):
    """
    Queries AWS CloudWatch Logs for a specific log group within a time range.
    """
    client = boto3.client('logs', region_name=AWS_REGION)

    try:
        start_timestamp = int(datetime.fromisoformat(request.start_time_iso.replace('Z', '+00:00')).timestamp() * 1000)
        end_timestamp = int(datetime.fromisoformat(request.end_time_iso.replace('Z', '+00:00')).timestamp() * 1000)

        # Use different APIs based on whether we have a query string
        if request.query_string and request.query_string.strip():
            # Use CloudWatch Logs Insights for complex queries
            params = {
                'logGroupName': request.log_group_name,
                'startTime': start_timestamp,
                'endTime': end_timestamp,
                'queryString': request.query_string,
                'limit': request.limit
            }
            start_query_response = client.start_query(**params)
        else:
            # Use simple query for basic log retrieval
            params = {
                'logGroupName': request.log_group_name,
                'startTime': start_timestamp,
                'endTime': end_timestamp,
                'queryString': 'fields @timestamp, @message | sort @timestamp desc',
                'limit': request.limit
            }
            start_query_response = client.start_query(**params)

        query_id = start_query_response['queryId']

        # Debug info
        print(f"Started query {query_id} for log group: {request.log_group_name}")
        print(f"Time range: {request.start_time_iso} to {request.end_time_iso}")

        # Poll for query results
        response = None
        status = 'Running'
        max_attempts = 30  # Maximum 30 seconds wait
        attempts = 0

        while status in ['Running', 'Scheduled'] and attempts < max_attempts:
            time.sleep(1)
            response = client.get_query_results(queryId=query_id)
            status = response['status']
            attempts += 1
            print(f"Query {query_id} status: {status} (attempt {attempts})")

        if status == 'Complete' and response['results']:
            # Flatten the results for easier processing
            formatted_results = []
            for result_set in response['results']:
                event_data = {}
                for field in result_set:
                    event_data[field['field']] = field['value']
                formatted_results.append(event_data)
            return {"status": "success", "data": formatted_results, "count": len(formatted_results)}
        elif status == 'Complete':
            return {"status": "success", "message": "No log data found for the given query.", "count": 0}
        else:
            return {"status": "error", "message": f"AWS log query failed or stopped with status: {status}"}

    except Exception as e:
        print(f"Error details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying AWS logs: {e}")

@app.post("/get-metrics")
async def get_aws_metric_statistics(request: MetricRequest):
    """
    Retrieves AWS CloudWatch metrics statistics.
    """
    client = boto3.client('cloudwatch', region_name=AWS_REGION)

    try:
        start_timestamp = datetime.fromisoformat(request.start_time_iso.replace('Z', '+00:00'))
        end_timestamp = datetime.fromisoformat(request.end_time_iso.replace('Z', '+00:00'))

        dimension_list = [{'Name': k, 'Value': v} for k, v in request.dimensions.items()] if request.dimensions else []

        response = client.get_metric_statistics(
            Namespace=request.namespace,
            MetricName=request.metric_name,
            StartTime=start_timestamp,
            EndTime=end_timestamp,
            Period=request.period_seconds,
            Statistics=[request.statistic],
            Dimensions=dimension_list
        )

        datapoints = []
        for dp in response.get('Datapoints', []):
            datapoints.append({
                "Timestamp": dp['Timestamp'].isoformat(),
                request.statistic: dp[request.statistic],
                "Unit": dp['Unit']
            })

        if datapoints:
            return {"status": "success", "data": datapoints}
        return {"status": "success", "message": f"No data for AWS Metric {request.metric_name} in namespace {request.namespace}."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting AWS metric: {e}")

@app.post("/analyze-text")
async def analyze_text_for_patterns(request: TextAnalysisRequest):
    """
    Analyzes a block of text for predefined patterns or keywords.
    """
    found_lines = []
    for line in request.text_data.splitlines():
        if any(p.lower() in line.lower() for p in request.patterns):
            found_lines.append(line.strip())

    if found_lines:
        return {"status": "success", "data": found_lines, "message": f"Found {len(found_lines)} lines matching patterns"}
    return {"status": "success", "data": [], "message": "No specified patterns found in the text data."}

@app.post("/troubleshooting-steps")
async def propose_aws_troubleshooting_steps(request: TroubleshootingRequest):
    """
    Based on an AWS incident summary, suggests initial troubleshooting steps.
    """
    steps = [
        "Check AWS Health Dashboard for any service interruptions in your region.",
        "Verify resource status (EC2, RDS, Lambda, etc.) in the AWS Management Console.",
        "Examine CloudWatch Alarms for any related triggers.",
        "Review recent CloudTrail events for API calls that might have caused changes.",
        "Check Security Groups and Network ACLs for connectivity issues.",
        "For web applications, inspect Application Load Balancer (ALB) metrics and logs.",
        "If using containers, check ECS/EKS service events and pod/task logs.",
        "Ensure IAM roles/permissions are correctly configured for affected services."
    ]

    return {
        "status": "success",
        "incident_summary": request.incident_summary,
        "troubleshooting_steps": steps
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AWS Observability API Server"}


# --- NEW Endpoints for Log Embeddings ---

@app.post("/ingest-log-embeddings")
async def ingest_log_embeddings(request: LogIngestionRequest):
    """
    Receives a list of log messages, embeds them, and stores them in the vector database.
    """
    if not log_vectorstore:
        raise HTTPException(status_code=500, detail="Log vector store not initialized.")
    if not request.logs:
        return {"status": "success", "message": "No logs provided for ingestion."}

    try:
        # Add texts to the vectorstore. Embeddings are generated automatically by Chroma
        # using the embeddings_model provided during initialization.
        log_vectorstore.add_texts(request.logs)
        log_vectorstore.persist() # Save changes to disk
        return {"status": "success", "message": f"Successfully ingested {len(request.logs)} log entries into vector DB."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting log embeddings: {e}")

@app.post("/search-similar-logs")
async def search_similar_logs(request: LogSimilaritySearchRequest):
    """
    Performs a similarity search in the log embeddings database for a given query.
    Returns the 'k' most similar log messages.
    """
    if not log_vectorstore:
        raise HTTPException(status_code=500, detail="Log vector store not initialized.")

    try:
        # Perform similarity search
        # docs are Document objects, which have .page_content and .metadata
        found_docs = log_vectorstore.similarity_search(request.query, k=request.k)
        
        results = [doc.page_content for doc in found_docs]
        
        if results:
            return {"status": "success", "data": results, "count": len(results)}
        return {"status": "success", "data": [], "message": "No similar logs found."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching similar logs: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
