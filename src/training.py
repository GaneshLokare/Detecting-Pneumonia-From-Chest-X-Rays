# Imports required for this project
import tensorflow as tf
from pathlib import Path
import sys
from exception import XrayException

tf.random.set_seed(4)


class TrainModel:
    def __init__():
        pass

    def train():
        '''It will train the model'''
        try:
            
            # Creating the Pathlib PATH objects
            train_path = Path("src/chest_xray/train/")
            validation_path = Path("src/chest_xray/test")


            # Collecting all the Paths Inside "Normal" and "Pneumonia" folders of the above paths
            train_image_paths = train_path.glob("*/*")
            val_image_paths = validation_path.glob("*/*")

            # Convert Generator Object to List of elements 
            train_image_paths = list(train_image_paths)
            val_image_paths = list(val_image_paths)

            # Convert Posix paths to normal strings
            train_image_paths = list(map(lambda x : str(x) , train_image_paths))
            val_image_paths = list(map(lambda x : str(x) , val_image_paths)) 

            # Collect Length for Training and Validation Datasets
            train_dataset_length = len(train_image_paths)
            val_dataset_length = len(val_image_paths)


            # Every Image has Label in its path , so lets slice it 
            LABELS = {'NORMAL' : 0 , 'PNEUMONIA' : 1}
            INV_LABELS = {0 : 'NORMAL', 1 : 'PNEUMONIA'}

            def get_label(path : str) -> int:
                return LABELS[path.split("\\")[-2]]

            train_labels = list(map(lambda x : get_label(x) , train_image_paths))
            val_labels = list(map(lambda x : get_label(x) , val_image_paths))

            # Now we have all training, validation image paths and their respective labels 

            BATCH_SIZE = 32

            # Function used for Transformation
            def load_and_transform(image , label , train = True):
                image = tf.io.read_file(image)
                image = tf.io.decode_jpeg(image , channels = 3)
                image = tf.image.resize(image , [224 , 224] , method="nearest")
                if train:
                    image = tf.image.random_flip_left_right(image)
                return image , label

            # Function used to Create a Tensorflow Data Object
            def get_dataset(paths , labels , train = True):
                image_paths = tf.convert_to_tensor(paths)
                labels = tf.convert_to_tensor(labels)

                image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
                label_dataset = tf.data.Dataset.from_tensor_slices(labels)

                dataset = tf.data.Dataset.zip((image_dataset , label_dataset)).shuffle(1000)

                dataset = dataset.map(lambda image , label : load_and_transform(image , label , train))
                dataset = dataset.repeat()
                dataset = dataset.shuffle(2048)
                dataset = dataset.batch(BATCH_SIZE)

                return dataset

            # Creating Train Dataset object
            train_dataset = get_dataset(train_image_paths , train_labels)

            # Creating Validation Dataset object
            val_dataset = get_dataset(val_image_paths , val_labels , train = False)

            # Building ResNet50 model
            from tensorflow.keras.applications import EfficientNetB3

            backbone = EfficientNetB3(
                input_shape=(224, 224, 3),
                include_top=False
            )

            model = tf.keras.Sequential([
                backbone,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            # Compiling your model by providing the Optimizer , Loss and Metrics
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
                loss = 'binary_crossentropy',
                metrics=['accuracy' , tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')]
            )

            # Defining our callbacks 
            checkpoint = tf.keras.callbacks.ModelCheckpoint("best_weights.h5",verbose=1,save_best_only=True,save_weights_only = True)
            early_stop = tf.keras.callbacks.EarlyStopping(patience=4)

            # Train the model
            history = model.fit(
                train_dataset,
                steps_per_epoch=train_dataset_length//BATCH_SIZE,
                epochs=2,
                callbacks=[checkpoint , early_stop],
                validation_data=val_dataset,
                validation_steps = val_dataset_length//BATCH_SIZE,
            )

            # After training best weights will be automatically saved
            # Load the best weights
            model.load_weights("best_weights.h5")
            # Save the whole model (weigths + architecture)
            model.save("model final.h5")

        except  Exception as e:
                raise  XrayException(e,sys)
