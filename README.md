# Detecting-Pneumonia-From-Chest-X-Rays

## Introduction:
This project aims to develop a deep learning model to detect pneumonia from x-ray images. Pneumonia is an infection in one or both lungs, which causes inflammation in the air sacs. Early detection of pneumonia is crucial for prompt treatment and recovery. However, manual analysis of x-ray images by radiologists can be time-consuming and prone to errors. This project aims to use deep learning to automate the process and improve the accuracy of pneumonia detection.

## Data:
The dataset used in this project is the Chest X-Ray Images (Pneumonia) dataset from Kaggle. It contains 5,863 x-ray images, with 2,828 images of pneumonia cases and 3,035 images of healthy cases. The images are of varying sizes and have been pre-processed to have consistent size and quality. The dataset is divided into a training set (5,216 images) , validation set (16 images) and a test set (624 imagges).

Data Source: https://www.dropbox.com/s/jkgg6azweowwiaf/archive.zip?dl=0

## Model:
The model used in this project is a pretrained convolutional neural network (CNN) implemented using the Keras library with Tensorflow backend (EfficientNetB3). The model is trained using the Adam optimizer and the binary cross-entropy loss function.

## Evaluation:
The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.
- Testing Acc :  0.844
- Testing Precision  0.808
- Testing Recall  0.984

## Conclusion:
The model developed in this project is able to achieve high accuracy in detecting pneumonia from x-ray images. This can potentially aid radiologists in their analysis and ultimately improve patient outcomes. However, it is important to note that deep learning models may not be 100% accurate and should be used as a tool to aid in the diagnostic process, not replace it.

### Instructions to run the code:

- Download the project files
- Download the data from provided link and store in the src folder with same folder structure.
- Install the required libraries mentioned in the requirements.txt file
- Run the main.py file
- The model will be trained and evaluated on the dataset
- The performance metrics will be displayed on the console as well as it will be stored in the logs.
- Model is deployed using streamlit. Run "streamlit run Home.py" command and go to "http://192.168.43.166:8501" link.

Note: The trained model and the weights are not included in this repository, please train the model on your own machine.

## Demo Video:




https://user-images.githubusercontent.com/90838133/218657261-ecb71db2-b039-4db3-a17b-a6d9e2d3132d.mp4

