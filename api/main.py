from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np 
from io import BytesIO
from PIL import Image
import tensorflow as tf
from keras.layers import TFSMLayer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins=[
    "http://localhost",
    "http://localhost:5500",  
    "http://127.0.0.1",
    "http://127.0.0.1:5500",  
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=["*"]
)
MODEL = TFSMLayer("../models/3", call_endpoint="serving_default")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return 'hello, I am alive'

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    return np.array(image)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image = np.expand_dims(image, axis=0).astype(np.float32)  
    tensor = tf.convert_to_tensor(image)
    predictions = MODEL(tensor)
    output = list(predictions.values())[0].numpy()[0]  

    predicted_class = CLASS_NAMES[np.argmax(output)]
    confidence = float(np.max(output))
    print("prediction is called from the front")
    return {
        "class": predicted_class,
        "confidence": confidence
    }
    

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)