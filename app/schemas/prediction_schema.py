from pydantic import BaseModel, Field
from typing import List

class PredictionOutput(BaseModel):
    prob_evasao: float = Field(..., description="Probabilidade predita de evasão (0..1)")
    classe_prevista: int = Field(..., description="Classe prevista: 1 = evadiu, 0 = permaneceu")
    threshold: float = Field(..., description="Threshold usado para conversão em classe")

class BatchPredictionRequest(BaseModel):
    alunos: List[dict] = Field(..., description="Lista de alunos (cada um com os mesmos campos do StudentInput)")

class BatchPredictionOutput(BaseModel):
    results: List[PredictionOutput]
