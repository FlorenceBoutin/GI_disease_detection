import streamlit as st
import requests
# from rembg import remove
from PIL import Image
from io import BytesIO
import base64

url = 'https://baseline-model1-nfkry3aggq-ez.a.run.app/'

st.set_page_config(layout="wide", page_title="Image Background Remover")

st.write("## GI Tract Image Classifier ")
st.write("### Try uploading an image to evaluate potential diseased tissue.")
st.sidebar.write("## Upload Here")

img_file_buffer = st.file_uploader('Upload an image')


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
        st.write("Image received by server!")
        # Extract the JSON response
        json_data = res.json()
        # st.write(json_data)
        # Access the prediction value
        prediction_normal = json_data["prediction"]["0"]
        prediction_uc = json_data["prediction"]["1"]
        prediction_polyps = json_data["prediction"]["2"]
        prediction_eso = json_data["prediction"]["3"]
        # Print the prediction value
        a = f"Probability of Normal:{round(prediction_normal*100)}%"
        b = f"Probability of UC:{round(prediction_uc*100)}%"
        c = f"Probability of Polyps:{round(prediction_polyps*100)}%"
        d = f"Probability of Eso:{round(prediction_eso*100)}%"
        st.write(a)
        st.write(b)
        st.write(c)
        st.write(d)

      else:
        st.write("Error: Image not received by server.")
    #   response = requests.get(url)
    #   prediction = response.json()
    #   st.header(prediction)

    #   if res.status_code == 200:
    #     ### Display the image returned by the API
    #     st.image(res.content, caption="Image returned from API ☝️")
    #   else:
    #     st.markdown("**Oops**, something went wrong 😓 Please try again.")
    #     print(res.status_code, res.content)

# if st.button('Classify'):
#     # print is visible in the server output, not in the page
#     print('Processing')
#     st.write('Processing :hourglass_flowing_sand:')
#     # url = "https://api30-nfkry3aggq-ez.a.run.app/predict"
#     url ="http://127.0.0.1:8000/"
#     response = requests.get(url)
#     prediction = response.json()
#     st.header(prediction)
#     st.write('Further clicks are not visible but are executed')
# else:
#     st.write('Click to Classify')
