import os # helps to read environment variables and file paths
import chromadb # local vector database used to store and query mathematical representations 
from openai import OpenAI # imports OpenAI client class to generate text embeddings and execue LLM prompts
from dotenv import load_dotenv # loads environment variables from a .env file and reads key-value pairs

# Executes the function to load the variables inside .env into memory
load_dotenv() 

# Initalize the OpenAI client using our secure secret key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB to save vector data locally on our hard drive
chroma_client = chromadb.PersistentClient(path="./vector.db")

# Connect to a specific database collection (like a table in SQL) for legal rules
collection = chroma_client.get_or_create_collection(name="far_dfars_rules") 


def query_far_regulations(contract_snippet: str):
    """Embeds contract snippet and queries ChromaDB for matching clauses"""

    # Step 1: Send text to OpenAI to get its numberical vector representation
    response = client.embeddings.create(
        input=contract_snippet,
        mode="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding

    # Step 2: Ask ChromaDB to find the top 2 closest matching document chunks 
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    # Step 3: Return the raw text documents if found, otherwise return an empty list
    return results['documents'][0] if results['documents'] else []