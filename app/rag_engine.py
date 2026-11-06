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

def analyze_compliance(contract_test: str):
    """Retrieves relevant FAR context and calls GPT-4o for risk review"""
    
    # Step 1: Retrieve the legal context using our vector search function above
    relevant_clauses = query_far_regulations(contract_test)

    # Format the clauses neatly into a single string block
    context = "\n\n---\n\n".join(relevant_clauses) if relevant_clauses else "No specific regulatory context found."

    # Step 2: Construct the strict prompt framing GPT-4o as an Aerospace Contract manager
    prompt = f"""
    You are an expert Aerospace Contract Management Representative. 
    Analyze the following contract section against the provided FAR/DFARS regulatory context.
    Identify any compliance gaps, risks, or missing mandatory clauses.

    REGULATORY CONTEXT (FAR/DFARS):
    {context}

    CONTRACT SECTION TO REVIEW:
    {contract_test}

    Provide your analysis in a structured format:
    1. Compliance Status (Compliant / High Risk / Missing Clause)
    2. Explanation of Findings
    3. Recommended Action / Redline Suggestion
    """

    #Step 3: Send the prompt to GPT-4o. We use temperature=0.1 to keep the output precise and deterministic
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    # Step 4: Return both the AI's written analysis and the raw clauses it references
    return {
        "analysis": response.choices[0].message.content,
        "referenced_clauses": relevant_clauses
    }