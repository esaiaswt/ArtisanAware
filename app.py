## Importing the necessary libraries

import pathlib
import textwrap

import google.generativeai as genai

import PIL.Image

# Machine Learning libraries
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import logging

import os
import matplotlib.pyplot as plt
import streamlit as st

# Generate image from text
def generate_image(prompt):
    """ This function generate image from a text with stable diffusion"""
    with autocast(device):
      image = stable_diffusion_model(prompt,guidance_scale=8.5)["images"][0]

    return image

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

   img = PIL.Image.open(uploaded_file)

   model_genai = genai.GenerativeModel('gemini-pro-vision')
   response = model_genai.generate_content(["Write a detail description based on this picture.", img], stream=True)
   response.resolve()
   st.write("Generated Description: ")
   st.write(response.text)

   # Download stable diffusion model from hugging face
   logging.disable_progress_bar()   # Prevent Streamlit TQDM output error 
   modelid = "CompVis/stable-diffusion-v1-4"
   device = "cuda" if torch.cuda.is_available() else "cpu"
   stable_diffusion_model = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16)
   stable_diffusion_model.to(device)
 
   save_name = ""
   for i in range(1, 4):
      image = generate_image(response.text)

      # Save the generated image
      save_name = "gen_image" + str(i) + ".png"
      image.save(save_name)
      st.write(save_name)

   st.title("Generated Images:")
   #col1, col2, col3, col4, col5 = st.columns(5)
   #col1, col2, col3 = st.columns(3)

   #with col1:
   #   st.image('gen_image1.png', 'Generated Image 1')

   #with col2:
   #   st.image('gen_image2.png', 'Generated Image 2')

   #with col3:
   #   st.image('gen_image3.png', 'Generated Image 3')

   #with col4:
   #   st.image('gen_image4.png', 'Generated Image 4')

   #with col5:
   #   st.image('gen_image5.png', 'Generated Image 5')