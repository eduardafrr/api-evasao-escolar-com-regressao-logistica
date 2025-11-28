from fastapi import FastAPI
from app.views import prediction_view
from app.core.config import API_VERSION

app = FastAPI(title="API Predição Evasão", version=API_VERSION)

app.include_router(prediction_view.router, prefix="")

@app.get("/")
def root():
    return {"message": "API Predição Evasão. Veja /docs para documentação."}
