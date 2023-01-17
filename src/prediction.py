import tensorflow as tf
import sys
from exception import XrayException

# Loading the trained model
model = tf.keras.models.load_model("model final.h5")
image_path = "src/chest_xray/test/NORMAL/IM-0065-0001.jpeg"
class Prediction:
    def __init__():
        pass

    def get_result():
      ''' after uploading x-ray image it will predict the result'''

    try:

      # read and transform image into tensor
      def LoadImage(image_path):
          image = tf.io.read_file(image_path)
          image = tf.io.decode_jpeg(image , channels = 3)
          image = tf.image.resize(image , [224 , 224] , method="nearest")
          image = tf.expand_dims(image , 0)
          return image

      # get prediction
      def RealtimePrediction(image_path , model):
          image = LoadImage(image_path)
          prediction = model.predict(image)
          if prediction[0] <= 0.5:
            print("NORMAL")
          else:
            print("PNEUMONIA")

    except  Exception as e:
        raise  XrayException(e,sys)


Prediction.get_result()