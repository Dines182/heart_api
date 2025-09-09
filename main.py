from fastapi import FastAPI
from pydantic import BaseModel, conint, confloat
import joblib, pandas as pd

app = FastAPI(title="Realtime Heart Disease API")

class PredictRequest(BaseModel):
    age: conint(ge=0, le=120)
    sex: conint(ge=0, le=1)
    cp: conint(ge=0, le=3)
    trestbps: confloat(gt=0)
    chol: confloat(gt=0)
    fbs: conint(ge=0, le=1)
    restecg: conint(ge=0, le=2)
    thalach: confloat(gt=0)
    exang: conint(ge=0, le=1)
    oldpeak: float
    slope: conint(ge=0, le=2)
    ca: conint(ge=0, le=3)
    thal: conint(ge=0, le=3)

class PredictResponse(BaseModel):
    prediction: int
    proba: float

@app.on_event("startup")
def on_startup():
    app.state.model = joblib.load("model.pkl")
    app.state.features = joblib.load("feature_names.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    X = pd.DataFrame([req.dict()], columns=app.state.features)
    m = app.state.model
    pred = int(m.predict(X)[0])
    proba = float(m.predict_proba(X)[0, 1]) if hasattr(m, "predict_proba") else 0.0
    return {"prediction": pred, "proba": proba}
