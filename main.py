from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
sentiment_model = pipeline("sentiment-analysis")

class PredictionRequest(BaseModel):
  query_string: str

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"} 

@app.post("/sentiment")
def sentiment(request: PredictionRequest):
    sentiment = sentiment_model(request.query_string)
    return {"sentiment": sentiment}