import joblib
from pathlib import Path
from app.core.config import MODEL_PATH
from typing import Dict, Any, List

_model = None

def load_model():
    global _model
    if _model is None:
        path = Path(MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em: {path}")
        _model = joblib.load(path)
    return _model

def predict_proba_one(features: Dict[str, Any]) -> float:
    model = load_model()
    import pandas as pd
    df = pd.DataFrame([features])
    proba = model.predict_proba(df)[:,1][0]
    return float(proba)

def predict_proba_batch(list_features: List[Dict[str, Any]]) -> List[float]:
    model = load_model()
    import pandas as pd
    df = pd.DataFrame(list_features)
    probas = model.predict_proba(df)[:,1].tolist()
    return [float(p) for p in probas]
