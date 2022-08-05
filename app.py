#-----Make all the Necessary imports-----

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

from streamlit_option_menu import option_menu

# st.set_page_config(layout="wide")


#for windows deployment
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


#For linux deployment
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath


path = Path('.')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



#------Create/Define all functions

def get_x(r): 
    return image_path/r['train_image_name']

def get_y(r): 
    return r['class']


def load_model():
    model = load_learner(path/'export.pkl')
    return model

model = load_model()

def display_image(display_img):
    st.image(display_img, width=400)
    # use_column_width=True
    

def make_pred(model, img):
  
    # Temporarily displays a message while executing 
    with st.spinner('Performing Diagnosis...'):
        time.sleep(1)
        
    pred, pred_idx, prob = model.predict(img)
    pred_prob = f'{prob[pred_idx]*100:.0f}%'
    
    # Display the prediction
    if pred == '1':
        pred_state = 'Presence of Diabetic Retinopathy'
        # st.success("Presence of Diabetic Retinopathy.")

        if prob[pred_idx]*100 <= 85.0:
            rec_action = 'Please visit an Opthalmologist to confirm your Daignosis'
        else:
            rec_action = 'Please visit an Opthalmologist for prompt treatment.'


    else:
        pred_state = 'Absence of Diabetic Retinopathy'
        # st.success("Absence of Diabetic Retinopathy.")

        if prob[pred_idx]*100 <= 85.0:
            rec_action = 'Please visit an Opthalmologist to confirm your Daignosis'
        else:
            rec_action = 'Please be sure to go for Diagnosis once a year.'
        
    return pred_state, rec_action, pred_prob
    

########--------Setup Diagnosis Page--------########

# if selected_nav == 'Diagnosis':

    ########-------Create Side Bar---------########

#For image upload
img_upload = st.sidebar.file_uploader(label = 'Upload a Fundus Image for Diagnosis', 
                        type=['png', 'jpg', 'jpeg'])

# For image selection
test_images = os.listdir(path/'sample')
img_selected = st.sidebar.selectbox(
        'Please Select a Sample Fundus Image:', test_images)


if img_selected:
    # Read the image
    file_path = path/'sample'/img_selected
    # Get the image to display
    display_img = Image.open(file_path)
    # display_img = display_img.resize((244,224))
    img = PILImage.create(file_path)


if img_upload:
    display_img = Image.open(img_upload)
    img = PILImage.create(img_upload)




st.markdown("""

<h3 style="text-align:center;color:#006ef5;">Diagnosing Diabetic Retinopathy with AI (DEMO)</h3>


""", unsafe_allow_html=True)

st.markdown("##")

st.markdown("""

<p> <b>Instruction:</b> Please upload a fundus image (using the sidebar) for diagnosis or select a sample image</p>

""", unsafe_allow_html=True)

with st.container():

    left_col, right_col = st.columns((2,1))

    with left_col:

        display_image(display_img)

    with right_col:

        pred_state, rec_action, pred_prob = make_pred(model, img)

        # st.markdown("##")

        if pred_state=="Presence of Diabetic Retinopathy":
            st.markdown(f""" 
            <p style="text-align:left;"> <b>Diagnosis Result:</b></p> <br>
            <p style="color:red;text-align:left;font-size:16px;">{pred_state}</p>
            """
            , unsafe_allow_html=True)
            
        else:
                st.markdown(f""" 
            <p style="text-align:left;"> <b>Diagnosis Result:</b></p>
            <p style="color:green;text-align:left;font-size:16px;">{pred_state}</p>
            """
            , unsafe_allow_html=True)
        
        st.markdown(f"""
        <p style="text-align:left;"> <b>Diagnosis Confidence:</b></b></p>
        <p style="color:#006ef5;text-align:left;font-size:16px;">{pred_prob}</p>""", unsafe_allow_html=True)
            
        # st.markdown(f" **Diagnosis Confidence:** {pred_prob}")
        st.markdown(f""" 
        <p style="text-align:left;"> <b>Recommendaton:</b></p>
        <p style="color:#006ef5;text-align:left;font-size:16px;">{rec_action}</p>
        
        """, unsafe_allow_html=True)



 