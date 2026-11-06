from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag_engine import analyze_compliance, collection, client

app = FastAPI()

FastAPI(
    title="FAR/DFARS Contract Compliance Assistant",
    description="API for checking aerospace subcontracts against federal regulations"
    version="1.0.0"
)

class ContractRequest(BaseModel):
    contract_text: str


class SeedRequest(BaseModel):
    id: str
    text: str

