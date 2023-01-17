from fastapi import FastAPI, File, UploadFile
import uvicorn
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import cv2


# Loading the trained model
loaded_model = tf.keras.models.load_model("model final.h5")

app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

CLASS_NAMES = ["NORMAL","PNEUMONIA"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image = cv2.merge((image,image,image))
    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    image = np.expand_dims(image , 0)

    prediction = loaded_model.predict(image)
    if prediction[0] <= 0.5:
         predicted_class = CLASS_NAMES[0]
         confidence = (1 - prediction[0])*100
    else:
        predicted_class = CLASS_NAMES[1]
        confidence = prediction[0] * 100

    return {
        'class': predicted_class,
        'confidence' : float(confidence)}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)