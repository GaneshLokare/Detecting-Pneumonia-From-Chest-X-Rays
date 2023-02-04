import tensorflow as tf
from pathlib import Path
import sys
from exception import XrayException
from logger import logging

tf.random.set_seed(4)



class Evaluate_model_performance:
    def __init__(self):
        # Loading the trained model
        self.loaded_model = tf.keras.models.load_model("model final.h5")

    def Evaluate(self):
        ''' Evaluate model's performance '''
        try:
            
            # Creating the Pathlib PATH objects
            test_path = Path("src/chest_xray/val")

            # Every Image has Label in its path , so lets slice it 
            LABELS = {'NORMAL' : 0 , 'PNEUMONIA' : 1}
            INV_LABELS = {0 : 'NORMAL', 1 : 'PNEUMONIA'}

            # extract labels from image path
            def get_label(path : str) -> int:
                return LABELS[path.split("\\")[-2]]

            # Create a Dataset Object for 'Testing' Set just the way we did for Training and Validation
            test_image_paths = list(test_path.glob("*/*"))
            test_image_paths = list(map(lambda x : str(x) , test_image_paths))
            test_labels = list(map(lambda x : get_label(x) , test_image_paths))

            test_image_paths = tf.convert_to_tensor(test_image_paths)
            test_labels = tf.convert_to_tensor(test_labels)

            BATCH_SIZE = 32

            def decode_image(image , label):
                image = tf.io.read_file(image)
                image = tf.io.decode_jpeg(image , channels = 3)
                image = tf.image.resize(image , [224 , 224] , method="nearest")
                return image , label

            test_dataset = (
                tf.data.Dataset
                .from_tensor_slices((test_image_paths, test_labels))
                .map(decode_image)
                .batch(BATCH_SIZE)
            )

            # Evaluating the loaded model
            loss, acc, prec, rec = self.loaded_model.evaluate(test_dataset)
            # print evaluation report
            print(" Testing Acc : " , acc)
            print(" Testing Precision : " , prec)
            print(" Testing Recall : " , rec)
            print(" Testing Loss : ", loss)

            # log evaluation report
            logging.info(" Testing Acc : {}".format(acc))
            logging.info(" Testing Precision : {}".format(prec))
            logging.info(" Testing Recall : {}".format(rec))
            logging.info(" Testing Loss : {}".format(loss))

        except  Exception as e:
                raise  XrayException(e,sys)

