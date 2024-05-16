## Importing the necessary libraries

import google.generativeai as genai

import PIL.Image
import requests
import io
from PIL import Image

from sentence_transformers import util
import time

import os
import streamlit as st

def query_stabilitydiff(payload, headers):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

#def imageEncoder(img):
#    img1 = Image.fromarray(img).convert('RGB')
#    img1 = preprocess(img1).unsqueeze(0).to(device)
#    img1 = model.encode_image(img1)
#    return img1
def imageEncoder(data, headers):
   API_URL = "https://api-inference.huggingface.co/models/helenai/CLIP-ViT-B-16-plus-240"

   with open(data["image_path"], "rb") as f:
      img = f.read()
        
   payload={
		"parameters": data["parameters"],
		"inputs": base64.b64encode(img).decode("utf-8")
	}
   
   response = requests.post(API_URL, headers=headers, json=payload)
   return response.json()

def generateScore(image1, image2, headers):
    img1 = imageEncoder(image1, headers)
    img2 = imageEncoder(image2, headers)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0])*100, 2)
    return score

# Invoke Message Sending
# https://www.googlecloudcommunity.com/gc/AI-ML/Gemini-Pro-Quota-Exceeded/m-p/693185
# ResourceExhausted: 429 Resource has been exhausted (e.g. check quota).
def Invoke_SendMessage(convo, prompt, image):
  completed = False
  sleep = 0
  sleep_time = 2
  while not completed:
      try:
          response = convo.generate_content([prompt, image], stream=True)
      except Exception as re:
          #print(f"ResourceExhausted exception occurred while processing property: {re}")
          st.write(f"Exception: {re} But we are retrying...")
          sleep += 1
          if sleep > 5:
              #print(f"ResourceExhausted exception occurred 5 times in a row. Exiting.")
              st.write(f"Exception occurred 5 times in a row. Exiting.")
              break
          time.sleep(sleep_time)
          sleep_time *= 2
      else:
          completed = True

  return completed, response

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

st.write('Tokens used: ', st.session_state["tokens_count_key"]) 

uploaded_file = st.file_uploader("Upload Image", type = ['jpg', 'jpeg', 'png', 'bmp'], key=st.session_state["file_uploader_key"],)
if uploaded_file is not None:
   st.session_state["file_uploader_key"] += 1 # https://discuss.streamlit.io/t/are-there-any-ways-to-clear-file-uploader-values-without-using-streamlit-form/40903/2

   st.image(uploaded_file, uploaded_file.name)

   img = PIL.Image.open(uploaded_file)

   my_bar = st.progress(0, text="Starting the model...")

   model_genai = genai.GenerativeModel('gemini-pro-vision')

   response_text = []
   my_progress = 30
   for i in range(1, 6):
      mytext = f"Generating the description of the image {i}..."
      my_bar.progress(my_progress, text=mytext)
      if i == 1:
         prompt = "Write a detail description based on this picture."
      else: 
         prompt = "Write another detail description based on this picture."

      #response = model_genai.generate_content([prompt, img], stream=True)
      bRet, response = Invoke_SendMessage(model_genai, prompt, img)
      if bRet == False:
         break

      response.resolve()
      response_text.append(response.text)
      my_progress += 6
   if bRet:
      completed_to = 6
   else:
      completed_to = i

   #model_genai.count_tokens()
   #st.session_state["tokens_count_key"] += tokens.total_tokens

   my_progress = 60
   my_bar.progress(60, text="Generating Images...")
   save_name = ""
   save_list = []
   for i in range(1, completed_to):
      # Query Stable Diffusion - https://github.com/dmitrimahayana/Py-LangChain-ChatGPT-VirtualAssistance/blob/main/03_Streamlit_Stable_Diff.py
      headers = {"Authorization": f"Bearer {HF_TOKEN_KEY}"}
      image_bytes = query_stabilitydiff({
         "inputs": response_text[i-1],
      }, headers)

      # Return Image
      image = Image.open(io.BytesIO(image_bytes))
      
      # Save the generated image
      save_name = "gen_image" + str(i) + ".png"
      image.save(save_name)
      save_list.append(save_name)
      #st.write(save_name)
      my_progress += 8
      mytext = "Generated " +  save_name + "... Generating next..."
      my_bar.progress(my_progress, text=mytext)

   my_bar.progress(100, text="Completed.")
   st.title("Generated Images:")
   col1, col2, col3, col4, col5 = st.columns(5)

   # image processing model
   similarity_results = []
   ave = 0
   i = 1
   for each_image in save_list:
      result = generateScore(uploaded_file.name, each_image, headers)
      #print (compare_image_path, result, '%')
      similarity_results.append(result)
      ave += result

   ave = ave / len(similarity_results)
   #print('similarity average = ', round(ave, 2),'%')

   with col1:
      if completed_to > 1:
         st.write(response_text[0])
         st.image('gen_image1.png', 'Generated Image 1')
         st.write('Similarity: ', similarity_results[0])

   with col2:
      if completed_to > 2:
         st.write(response_text[1])
         st.image('gen_image2.png', 'Generated Image 2')
         st.write('Similarity: ', similarity_results[1])

   with col3:
      if completed_to > 3:
         st.write(response_text[2])
         st.image('gen_image3.png', 'Generated Image 3')
         st.write('Similarity: ', similarity_results[2])

   with col4:
      if completed_to > 4:
         st.write(response_text[3])
         st.image('gen_image4.png', 'Generated Image 4')
         st.write('Similarity: ', similarity_results[3])

   with col5:
      if completed_to > 5:
         st.write(response_text[4])
         st.image('gen_image5.png', 'Generated Image 5')
         st.write('Similarity: ', similarity_results[4])

   st.write('Average Similarity: ', ave)