import tensorflow as tf
import sys
from exception import XrayException

# Loading the trained model
model = tf.keras.models.load_model("model final.h5")
class Prediction:
    def __init__():
        pass

      
      # read and transform image into tensor
    def RealtimePrediction(image_path):
        ''' after uploading x-ray image it will predict the result'''
        try:
            image = tf.io.read_file(image_path)
            image = tf.io.decode_jpeg(image , channels = 3)
            image = tf.image.resize(image , [224 , 224] , method="nearest")
            image = tf.expand_dims(image , 0)
            prediction = model.predict(image)
            if prediction[0] <= 0.5:
              print("NORMAL")
            else:
              print("PNEUMONIA")

        except  Exception as e:
            raise  XrayException(e,sys)


Prediction.RealtimePrediction("src/chest_xray/train/NORMAL/IM-0115-0001.jpeg")