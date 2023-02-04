import tensorflow as tf
import sys
from exception import XrayException


class Prediction:
    def __init__(self):
        # Loading the trained model
        self.model = tf.keras.models.load_model("model final.h5")

      
      # read and transform image into tensor
    def Real_time_Prediction(self,image_path):
        ''' after uploading x-ray image it will predict the result'''
        try:
            image = tf.io.read_file(image_path)
            image = tf.io.decode_jpeg(image , channels = 3)
            image = tf.image.resize(image , [224 , 224] , method="nearest")
            image = tf.expand_dims(image , 0)
            prediction = self.model.predict(image)
            if prediction[0] <= 0.5:
              print("NORMAL")
            else:
              print("PNEUMONIA")

        except  Exception as e:
            raise  XrayException(e,sys)


