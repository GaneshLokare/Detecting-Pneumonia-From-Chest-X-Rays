import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

CLASS_NAMES = ["NORMAL","PNEUMONIA"]

st.set_page_config(page_title="Chest X-Ray checking",
                   layout='wide',
                   page_icon='./icon/object.png')

st.header('Get Pneumonia Detection for Chest X-Ray Image')
st.write('Please Upload Only Chest X-Ray Image to get detection')

with st.spinner('Please wait while your model is loading'):
    model = tf.keras.models.load_model("model final.h5")

    #st.balloons()

def upload_image():
    # Upload Image
    image_file = st.file_uploader(label='Upload Image')
    if image_file is not None:
        size_mb = image_file.size/(1024**2)
        file_details = {"filename":image_file.name,
                        "filetype":image_file.type,
                        "filesize": "{:,.2f} MB".format(size_mb)}
        #st.json(file_details)
        # validate file
        if file_details['filetype'] in ('image/png','image/jpeg'):
            st.success('VALID IMAGE file type (png or jpeg')
            return {"file":image_file,
                    "details":file_details}
        
        else:
            st.error('INVALID Image file type')
            st.error('Upload only png,jpg, jpeg')
            return None
        
def main():
    object = upload_image()
    
    if object:
    
        image_obj = Image.open(object['file'])     
        
        col1 , col2 = st.columns(2)
        
        with col1:
            st.info('Preview of Image')
            st.image(image_obj)
            
        with col2:
            st.subheader('Check below for file details')
            st.json(object['details'])
            button = st.button('Check X-Ray')
            if button:
                with st.spinner("""
                Checking X-Ray. please wait
                                """):
                    # below command will convert
                    # obj to array
                    
                    image = np.array(image_obj)
                    # some times we get X-Ray as a gray scale image and some time RGB image to handle both we need below
                    # if else condition.
                    if len(image.shape) == 3: 
                        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                        image = np.expand_dims(image , 0)
                    else:
                        image = cv2.merge((image,image,image))
                        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                        image = np.expand_dims(image , 0)
                    
                    # get prediction
                    prediction = model.predict(image)
                    if prediction[0] <= 0.5:
                        predicted_class = CLASS_NAMES[0]
                        confidence = (1 - prediction[0])*100
                    else:
                        predicted_class = CLASS_NAMES[1]
                        confidence = prediction[0] * 100
                    pred = {
                        'class': predicted_class,
                        'confidence' : float(confidence)}
                    
                    st.success("Result")
                    st.json(pred)
                   
        
                
    
    
    
if __name__ == "__main__":
    main()