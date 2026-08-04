from model import AudioClassifier
import librosa
import numpy as np 
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch
import os 
import pandas as pd
from fastapi import FastAPI ,File


app = FastAPI()
#python-multipar should be installed

with open("mapping.txt", "r") as f:
    lines = f.readlines()
    mapping = [line.split(": ")[1].strip() for line in lines]



transform = T.Compose([
        T.ToTensor(),
        T.CenterCrop((128,128)),
        
    ])



@app.on_event("startup")
def load_model():
    global model
    model = AudioClassifier()
    chkpt = torch.load("best_model.pth",map_location=torch.device('cpu'))
    model.load_state_dict(chkpt["model_state_dict"])
    model.eval()
def predict(model, mel_spec_db):
    with torch.no_grad():
        model.eval() 
        outputs = model(mel_spec_db)
        preds = torch.argmax(outputs, dim=-1)
        return mapping[preds.item()]

@app.get("")
def health_check():
    return {"status": "API is running"}

#recieve a file upload and return the prediction
@app.post("/predict")
async def predict_audio(file: bytes = File(...)):
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(file)
    

    wav, sr = librosa.load(temp_file_path, sr=16000)
    mel_spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128, fmax=8000, hop_length=625)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = transform(mel_spec_db).unsqueeze(0) 
    prediction = predict(model, mel_spec_db)
    os.remove(temp_file_path)
    return {"prediction": prediction}






