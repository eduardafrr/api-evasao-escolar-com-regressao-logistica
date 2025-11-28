from app.services import ml_service
from app.core.config import THRESHOLD
from app.schemas.prediction_schema import PredictionOutput
from typing import Dict, Any, List

def predict_single(student_dict: Dict[str, Any]) -> PredictionOutput:
    proba = ml_service.predict_proba_one(student_dict)
    classe = 1 if proba >= THRESHOLD else 0
    return PredictionOutput(prob_evasao=proba, classe_prevista=classe, threshold=THRESHOLD)

def predict_batch(alunos: List[Dict[str, Any]]):
    probas = ml_service.predict_proba_batch(alunos)
    results = []
    for p in probas:
        results.append(PredictionOutput(prob_evasao=p, classe_prevista=(1 if p>=THRESHOLD else 0), threshold=THRESHOLD))
    return results
