import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64

url = 'https://final-model-2-nfkry3aggq-ez.a.run.app'

st.set_page_config(layout="wide", page_title="Digestive Disease Image Classifier")

st.write("## Digestive Disease Image Classifier")


img_file_buffer = st.file_uploader("Upload an image.")


if img_file_buffer is not None:

  col1, col2 = st.columns(2)

  with col1:
    ### Display the image user uploaded
    st.image(Image.open(img_file_buffer), caption="Here's the image you uploaded ☝️")

  with col2:
    with st.spinner("Wait for it..."):
      ### Get bytes from the file buffer
      img_bytes = img_file_buffer.getvalue()

      ### Make request to  API (stream=True to stream response as bytes)
      res = requests.post(url + "/upload_image", files={'img': img_bytes})

      if res.status_code == 200:
        st.write(" Image received by server!")
        # Extract the JSON response
        json_data = res.json()
        # st.write(json_data)
        # Access the prediction value
        prediction_normal = json_data["prediction"]["0"]
        prediction_uc = json_data["prediction"]["1"]
        prediction_polyps = json_data["prediction"]["2"]

        # Print the prediction value
        a = f"### Probability of Normal: {round(prediction_normal*100,2)}%"
        b = f"### Probability of UC: {round(prediction_uc*100,2)}%"
        c = f"### Probability of Polyps: {round(prediction_polyps*100,2)}%"

        st.write(a)
        st.write(b)
        st.write(c)


      else:
        st.write("Error: Image not received by server.")
