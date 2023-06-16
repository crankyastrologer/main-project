
from operator import index
import plotly.express as px
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import streamlit as st
import pandas as pd
import pytesseract
import ydata_profiling
import os
from streamlit_pandas_profiling import st_profile_report
import os
def apply_threshold(img, argument):
    switcher = {
        1: cv2.threshold(cv2.GaussianBlur(img, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        2: cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        3: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],

        18: cv2.adaptiveThreshold(cv2.medianBlur(img, 7), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
        19: cv2.adaptiveThreshold(cv2.medianBlur(img, 5), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
        20: cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    }
    return switcher.get(argument, "Invalid method")
def get_string(img, method):
    # Read image using opencv

    # Rescale the image, if needed.
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # Apply threshold to get image with only black and white
    img = apply_threshold(img, method)
    # Save the filtered image in the output directory

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(img, lang="eng")
    return result
st.header("Some projects by Ansh Verma")
st.text("Hello everyone i am Ansh Verma and this are some projects are made over the years \n that i am showcasing here ")
st.text('select from the projects below')
project = st.selectbox("Projects: ",['Select','Number plate detection and OCR','Disaster detection from twitter','Crytography with iris','data processing'])
if project =="Number plate detection and OCR":
    st.write("you have chosen numberplate detection")
    st.write('upload your picture')
    file = st.file_uploader("Upload Your picture")
    if file:
        try:
            img = Image.open(file)
            model = YOLO("best6.pt")  # build a new model from scratch

            # Use the model
            result = model(img)
            print(result[0].boxes.xyxy)
            imag = np.array(img)
            image = cv2.rectangle(imag, (497, 481), (758, 556), (255, 0, 0), 2)
            crop = imag[475:550,505:766 ]
            text = get_string(crop,18)
            st.write(text)

            # Now do something with the image! For example, let's display it:
            st.image(image, channels="RGB")

        except UnidentifiedImageError:
            st.write('image format not supported')





if project == 'data processing':
    st.title('data processing')
    st.write('upload your dataset')
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
        choice = st.radio("Navigation", ["Upload", "Profiling",'cleaning', "Modelling", "Download"])
        if choice == "Profiling":
            st.title("Exploratory Data Analysis")

            profile_df = df.profile_report()

            st_profile_report(profile_df)
        if choice == "Modelling":
            chosen_target = st.selectbox('Choose the Target Column', df.columns)
            if st.button('Run Modelling'):
                setup(df, target=chosen_target)
                setup_df = pull()
                st.dataframe(setup_df)
                best_model = compare_models()
                compare_df = pull()
                st.dataframe(compare_df)
                save_model(best_model, 'best_model')

        if choice == "Download":
            with open('best_model.pkl', 'rb') as f:
                st.download_button('Download Model', f, file_name="best_model.pkl")

