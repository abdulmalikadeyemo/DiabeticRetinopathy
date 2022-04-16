from fastai.vision import *
from fastai.imports import *
from fastai.learner import *
from fastai.vision.all import *

import streamlit as st
import numpy as np
import matplotlib.image as mpimg
import os
import time
from PIL import Image
import requests
from io import BytesIO

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

path = Path('.')


def get_x(r): return image_path/r['train_image_name']
def get_y(r): return r['class']


st.title("Diabetic Retinopathy Detection")

def predict(img, display_img):

    # Display the test image
    st.image(display_img, use_column_width=True)

    # Temporarily displays a message while executing 
    with st.spinner('Wait for it...'):
        time.sleep(3)

    # Load model and make prediction
    model = load_learner(path/'export.pkl')
    pred, pred_idx, prob = model.predict(img)
#     pred_prob = round(torch.max(model.predict(img)[2]).item()*100)
    
    # Display the prediction
    if pred == '1':
        st.success(f"Presence of Diabetic Retinopathy with a confidence of {prob[pred_idx]*100:.0f}%.")
    else:
        st.success(f"Absence of Diabetic Retinopathy with a confidence of {prob[pred_idx]*100:.0f}%.")
        

 # Image source selection
option = st.radio('', ['Choose a test image'])

if option == 'Choose a test image':

    # Test image selection
    test_images = os.listdir(path/'sample')
    test_image = st.selectbox(
        'Please select a test image:', test_images)

    # Read the image
    file_path = path/'sample'/test_image
    img = PILImage.create(file_path)
    # Get the image to display
    display_img = mpimg.imread(file_path)

    # Predict and display the image
    predict(img, display_img)

# else:
#     url = st.text_input("Please input a url:")

#     if url != "":
#         try:
#             # Read image from the url
#             response = requests.get(url)
#             pil_img = PIL.Image.open(BytesIO(response.content))
#             display_img = np.asarray(pil_img) # Image to display

#             # Transform the image to feed into the model
#             img = pil_img.convert('RGB')
#             img = image.pil2tensor(img, np.float32).div_(255)
#             img = image.Image(img)

#             # Predict and display the image
#             predict(img, display_img)

#         except:
#             st.text("Invalid url!")
