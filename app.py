
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = FastAPI(title="BERT Sentiment API")

model     = BertForSequenceClassification.from_pretrained("./bert-sentiment-final")
tokenizer = BertTokenizer.from_pretrained("./bert-sentiment-final")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class SingleRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: list[str]

def run_inference(texts):
    tokens = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model(**tokens)

    probs  = torch.softmax(output.logits, dim=1)
    preds  = torch.argmax(probs, dim=1).tolist()
    confs  = probs.max(dim=1).values.tolist()

    label_map = {0: "Negative", 1: "Positive"}
    return [
        {"text": t, "label": label_map[p], "confidence": round(c, 4)}
        for t, p, c in zip(texts, preds, confs)
    ]

@app.get("/")
def root():
    return {"message": "BERT Sentiment API is running"}

@app.post("/predict")
def predict(req: SingleRequest):
    return run_inference([req.text])[0]

@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    return {"results": run_inference(req.texts)}
