import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import subprocess
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO

ip = "localhost"
port = 8000

model = tf.keras.models.load_model(r"model/model.keras")

app = FastAPI()

@app.get("/", response_class=FileResponse)
def read_root():
    return FileResponse("templates/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Preprocess image
    img = Image.open(BytesIO(contents)).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predictions
    pred = model.predict(img_array)
    class_idx = np.argmax(pred[0])
    confidence = float(np.max(pred[0]))
    
    if class_idx == 0:
        label = "Tumor Detected"
    else:
        label = "Normal (No Tumor)"
        
    return {"prediction": label, "confidence": confidence}

if __name__ == "__main__":
    try:
        length = len(sys.argv)
        print(sys.argv)
        if length > 2 and sys.argv[2] == "new":
            print("Starting data preprocessing...")
            subprocess.run(["python", "src/data_preprocessor.py"], check=True)
        if length > 1 and sys.argv[1] == "retrain":               
            print("Starting training...")
            subprocess.run(["python", "src/train.py"], check=True)
            print("Starting evaluation...")
            subprocess.run(["python", "src/evaluation.py"], check=True)
            print("Retraining pipeline complete.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during retraining: {e}")
        sys.exit(1)

    uvicorn.run("app:app", host=ip, port=port, reload=False)