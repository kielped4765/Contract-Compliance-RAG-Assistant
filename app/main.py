from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag_engine import analyze_compliance, collection, client

app = FastAPI(
    title="FAR/DFARS Contract Compliance Assistant",
    description="API for checking aerospace subcontracts against federal regulations",
    version="1.0.0"
)

class ContractRequest(BaseModel):
    contract_text: str


class SeedRequest(BaseModel):
    id: str
    text: str

@app.post("/api/v1/seed")
def seed_clause(request: SeedRequest):
    """Seed sample FAR/DFARS clauses into the vector database."""

    try:
        response = client.embeddings.create(
            input=request.text,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding

        collection.add(
            embeddings=[embedding],
            documents=[request.text],
            ids=[request.id]
        )
        return {"status": "success", "message": f"Added clause {request.id} to vector database."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze")
def analyze_contract_endpoint(request: ContractRequest):
    """Run compliance review against stored regulations."""
    
    # Step 1: Validate that the user actually sent text (not empty whitespace)
    if not request.contract_text.strip():
        raise HTTPException(status_code=400, detail="Contract text cannot be empty.")
    
    try:
        # Step 2: Pass the contract text into our RAG engine to retrieve laws and run GPT-4o analysis
        result = analyze_compliance(request.contract_text)
        
        # Step 3: Return a successful JSON response containing the analysis and referenced clauses
        return {"status": "success", "data": result}
        
    except Exception as e:
        # Step 4: Catch any unexpected server or API errors and return a clean 500 error code
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze")
def analyze_contract_endpoint(request: ContractRequest):
    """Run compliance review against stored regulations."""
    
    # Step 1: Validate that the user actually sent text (not empty whitespace)
    if not request.contract_text.strip():
        raise HTTPException(status_code=400, detail="Contract text cannot be empty.")
    
    try:
        # Step 2: Pass the contract text into our RAG engine to retrieve laws and run GPT-4o analysis
        result = analyze_compliance(request.contract_text)
        
        # Step 3: Return a successful JSON response containing the analysis and referenced clauses
        return {"status": "success", "data": result}
        
    except Exception as e:
        # Step 4: Catch any unexpected server or API errors and return a clean 500 error code
        raise HTTPException(status_code=500, detail=str(e))