from fastapi import APIRouter, HTTPException
from app.schemas.students_schemas import StudentInput
from app.schemas.prediction_schema import PredictionOutput, BatchPredictionRequest, BatchPredictionOutput
from app.controllers import prediction_controller
from app.services import ml_service

router = APIRouter()

@router.get("/health", summary="Verifica se a API está viva")
def health():
    model_loaded = True
    try:
        ml_service.load_model()
    except Exception as e:
        model_loaded = False
        # não levantar erro 500 aqui; retornar model_loaded=false
    return {"status": "ok", "model_loaded": model_loaded}

@router.post("/predict", response_model=PredictionOutput, summary="Prediz evasão para um aluno")
def predict(student: StudentInput):
    try:
        result = prediction_controller.predict_single(student.dict())
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erro interno ao processar a predição")

@router.post("/predict_batch", response_model=BatchPredictionOutput, summary="Prediz evasão para vários alunos")
def predict_batch(req: BatchPredictionRequest):
    try:
        results = prediction_controller.predict_batch(req.alunos)
        return BatchPredictionOutput(results=results)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erro interno ao processar predições em batch")
