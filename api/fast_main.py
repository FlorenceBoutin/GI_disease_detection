from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
import io
import numpy as np
from PIL import Image
from tensorflow import keras
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
#Create a variable app for FastAPI
app = FastAPI()

# Optional, good practice for dev purposes. Allow all middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    )

#Load the trained model
model = models.load_model("../models/EfficientNetB5")

# Define the image size
img_size = (128, 128)
# Function to preprocess the uploaded image
def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA

    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w)/h
    saspect = float(sw)/sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape)==3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    # # normalize
    # normalized_img = scaled_img / 255.

    return scaled_img


@app.post("/upload_image")
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
    #What type of object is the cv2_img
    # print(type(cv2_img))
    # print(cv2_img.shape)

    # Define the image size
    img_size = (128, 128)

    preprocessed_image = resizeAndPad(cv2_img,img_size)
    # print(preprocessed_image.shape)
    #Expand dimensions of np array
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    pred = model.predict(preprocessed_image)
    # print(type(pred))
    print(pred)
    #Slice and choose the first item of the arrany, which is the probability of Normal
    dict_pred ={0 : float(pred[0][0]),1 : float(pred[0][1]),2: float(pred[0][2])}

    return {"prediction": dict_pred}



# End point to confirm communication with the server
@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="It works!")
    # $CHA_END

if __name__ == "__main__":
    # Get the port number from the PORT environment variable, or use a default value
    port = int(os.environ.get("PORT", 8080))
