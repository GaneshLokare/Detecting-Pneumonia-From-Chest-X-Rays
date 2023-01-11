from flask import Flask,request, jsonify, render_template
#import tensorflow as tf
import cv2


CLASS_NAMES = ["NORMAL","PNEUMONIA"]

# Loading the trained model
#loaded_model = tf.keras.models.load_model("model.h5")

app = Flask(__name__)

@app.route('/')
def upload_form():
    return render_template('upload_form.html')

@app.route('/predict', methods=['POST'])
def predict():

    image = request.files['image']

    image = tf.image.resize(image , [224 , 224] , method="nearest")
    image = tf.expand_dims(image , 0)

    prediction = loaded_model.predict(image)
    if prediction[0] <= 0.5:
         predicted_class = CLASS_NAMES[0]
    else:
        predicted_class = CLASS_NAMES[1]

    return {
        'class': predicted_class}



if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)