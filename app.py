import streamlit as st
import requests
# from rembg import remove
# from PIL import Image
# from io import BytesIO
# import base64

st.set_page_config(layout="wide", page_title="Image Background Remover")

st.write("## Colonoscopy Image Classifier ")
st.write("### Try uploading an image to evaluate potential diseased tissue.")
st.sidebar.write("## Upload Here")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if st.button('Classify'):
    # print is visible in the server output, not in the page
    print('Processing')
    st.write('Processing :hourglass_flowing_sand:')
    url = "https://api30-nfkry3aggq-ez.a.run.app/predict"
    response = requests.get(url)
    prediction = response.json()
    st.header(prediction)
    st.write('Further clicks are not visible but are executed')
else:
    st.write('Click to Classify')


# url = "https://api30-nfkry3aggq-ez.a.run.app/predict"

# response = requests.get(url)
# prediction = response.json()
# st.header(prediction)
# pred = prediction['Normal?']

# st.header(f'This tissue is: ${pred}')

# st.file_uploader(label, type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible")



# st.set_option('deprecation.showfileUploaderEncoding', False)

# image = Image.open('sunrise.jpg')
# st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
#import requests
#path -> model

#params
#response
#prediction
