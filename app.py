from flask import Flask,request, jsonify, render_template
import tensorflow as tf
import cv2


CLASS_NAMES = ["NORMAL","PNEUMONIA"]

# Loading the trained model
loaded_model = tf.keras.models.load_model("model.h5")

app = Flask(__name__)

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files.get('file')
    img_bytes = file.read()

    image = tf.image.resize(img_bytes , [224 , 224] , method="nearest")
    image = tf.expand_dims(image , 0)

    prediction = loaded_model.predict(image)
    if prediction[0] <= 0.5:
        class_name = CLASS_NAMES[0]
    else:
        class_name = CLASS_NAMES[1]

    return render_template('result.html',class_name=class_name)



if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)