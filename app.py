## Importing the necessary libraries

import pathlib
import textwrap

import google.generativeai as genai

import PIL.Image

# Machine Learning libraries
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

import os
import matplotlib.pyplot as plt
import streamlit as st

# 1.3 Retieve Google API key
os.environ["G_Key"] = st.secrets["G_Key"]
os.environ["H_Key"] = st.secrets["H_Key"]
api_key = os.getenv("G_Key")
genai.configure(api_key=api_key)

## Obtaining HuggingFace API Key**

# You will need a HuggingFace API Key and place in Streamlit secrets manager as `HF_TOKEN`.
HF_TOKEN_KEY=os.getenv("H_Key")

# Streamlit Interface
# -------------------
st.title("Artisan Aware")
st.write("With Generative AI, we may not know which image are drawn or photographed by an artist.  This project hope to help artist to understand the capabilities of Generative AI. Hence, this will help him/her to capture photos or produce artwork that excels the limits of generative AI.  Art is the expression of human being who is alive and living.")

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if "tokens_count_key" not in st.session_state:
    st.session_state["tokens_count_key"] = 0

uploaded_file = st.file_uploader("Upload Image", type = ['jpg', 'jpeg', 'png', 'bmp'], key=st.session_state["file_uploader_key"],)
if uploaded_file is not None:
  st.session_state["file_uploader_key"] += 1 # https://discuss.streamlit.io/t/are-there-any-ways-to-clear-file-uploader-values-without-using-streamlit-form/40903/2

st.image(uploaded_file, uploaded_file.name)

st.title("Generated Images:")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
   st.image('gen_image1.png', 'Generated Image 1')

with col2:
   st.image('gen_image2.png', 'Generated Image 2')

with col3:
   st.image('gen_image3.png', 'Generated Image 3')

with col4:
   st.image('gen_image4.png', 'Generated Image 4')

with col5:
   st.image('gen_image5.png', 'Generated Image 5')