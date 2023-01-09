import tensorflow as tf


# Loading the trained model
loaded_model = tf.keras.models.load_model("model.h5")

def LoadImage(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image , channels = 3)
    image = tf.image.resize(image , [224 , 224] , method="nearest")
    image = tf.expand_dims(image , 0)
    return image

def RealtimePrediction(image_path , model):
    image = LoadImage(image_path)
    prediction = model.predict(image)
    if prediction[0] <= 0.5:
      print("NORMAL")
    else:
      print("PNEUMONIA")


RealtimePrediction("src/chest_xray/test/NORMAL/IM-0065-0001.jpeg",loaded_model)  