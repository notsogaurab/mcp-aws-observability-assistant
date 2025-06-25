import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma # Or other vector stores
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Configuration
DOCS_PATH = "./aws_docs" # Directory where you'll store your AWS documentation
VECTOR_DB_PATH = "./vector_db" # Where your ChromaDB will be stored

# Ensure you have documents in the DOCS_PATH directory.
# For example, create a file `aws_docs/lambda_troubleshooting.md`

def ingest_documents():
    print("Loading documents...")
    # Example: Load all .md files from a directory
    loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*", # Or "**/*" for all file types
        loader_cls=TextLoader, # Use TextLoader for .md, .txt
        show_progress=True
    )
    # For PDFs:
    # pdf_loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    # docs.extend(pdf_loader.load())

    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks.")

    print("Generating embeddings and storing in vector DB...")
    # Use OpenAIEmbeddings if you prefer:
    # from langchain_openai import OpenAIEmbeddings
    # embeddings = OpenAIEmbeddings()

    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    vectorstore.persist()
    print(f"Vector DB created at {VECTOR_DB_PATH}")

if __name__ == "__main__":
    # Create a dummy documentation file for testing
    os.makedirs(DOCS_PATH, exist_ok=True)
    with open(os.path.join(DOCS_PATH, "aws_api_gateway_errors.md"), "w") as f:
        f.write("""
        # AWS API Gateway 5xx Errors Troubleshooting

        A 5xx error in AWS API Gateway indicates an issue with your backend integration. This could be a Lambda function, EC2 instance, or another AWS service.

        **Common Causes:**
        * **Lambda Timeout:** Your integrated Lambda function took longer than its configured timeout. Check Lambda execution logs in CloudWatch.
        * **Backend Unreachable:** The backend service (e.g., EC2 instance) is not reachable by API Gateway. Check VPC security groups, network ACLs, and routing.
        * **Malformed Response from Backend:** The backend returned a response that API Gateway couldn't understand or map.
        * **Internal Server Error in Backend:** The backend application itself threw an uncaught exception.
        * **Concurrency Limits:** Lambda concurrency limits reached.

        **Troubleshooting Steps:**
        1.  **Check CloudWatch Logs for API Gateway:** Look for `Execution Errors` and `Integration Errors`.
        2.  **Check CloudWatch Logs for Backend:** If using Lambda, check its logs for exceptions or timeouts.
        3.  **Verify Integration Request/Response Mappings:** Ensure your API Gateway integration is correctly configured to send/receive data from the backend.
        4.  **Test Backend Directly:** Bypass API Gateway and test the backend (e.g., invoke Lambda directly, access EC2 endpoint) to isolate the problem.
        5.  **Review IAM Permissions:** Ensure API Gateway has permission to invoke Lambda or access other backend services.
        """)

    with open(os.path.join(DOCS_PATH, "aws_ec2_cpu_metrics.md"), "w") as f:
        f.write("""
        # AWS EC2 CPU Utilization Metrics

        CPU utilization for EC2 instances is a key metric to monitor. High CPU usage can indicate a bottleneck or an application issue.

        **Key Metrics:**
        * `CPUUtilization`: The percentage of allocated EC2 compute units that are currently in use on the instance.
        * `CPUCreditUsage`: (For T2/T3 instances) The number of CPU credits consumed by the instance.
        * `CPUCreditBalance`: (For T2/T3 instances) The number of CPU credits remaining for the instance.

        **Troubleshooting High CPU Utilization:**
        1.  **Identify the Process:** Log into the EC2 instance and use `top`, `htop`, or Task Manager (Windows) to identify processes consuming high CPU.
        2.  **Application Logs:** Check application logs on the instance for errors or intense operations.
        3.  **Scaling:** Consider scaling up the instance type or scaling out with an Auto Scaling Group.
        4.  **Misconfiguration:** Check for runaway scripts or misconfigured cron jobs.
        """)

    ingest_documents()