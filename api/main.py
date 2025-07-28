from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np 
from io import BytesIO
from PIL import Image
import tensorflow as tf
from keras.layers import TFSMLayer

app = FastAPI()

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

    return {
        "class": predicted_class,
        "confidence": confidence
    }
    

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)