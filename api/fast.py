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
model = models.load_model("models/baseline_model.h5")
# Define the image size
img_size = (224, 224,3)


@app.post("/upload_image")
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
    #What type of object is the cv2_img
    # print(type(cv2_img))
    # print(cv2_img.shape)

    #Load the trained model
    model = models.load_model("models/baseline_model.h5")
    # Define the image size
    img_size = (224, 224)

    # # Define the ImageDataGenerator with the same preprocessing parameters as in your machine learning code
    # datagen = ImageDataGenerator(rescale=1./255)

    # Function to preprocess the uploaded image
    def preprocess_image(image):
         # Check if the image was uploaded successfully
        if image is None:
            return None
        img = image.copy()
        # Resize the image to the desired size
        img = cv2.resize(img,img_size)
        #Expand the dimension of your array to fit the model requirements
        img = np.expand_dims(img,axis=0)
        # Apply rescaling
        img = np.array(img) / 255.0
        # Return the preprocessed image
        return img

    preprocessed_image = preprocess_image(cv2_img)
    print(preprocessed_image.shape)
    pred = model.predict(preprocessed_image)
    # print(type(pred))
    print(pred)
    #Slice and choose the first item of the arrany, which is the probability of Normal
    dict_pred ={0 : float(pred[0][0]),1 : float(pred[0][1]),2: float(pred[0][2]), 3: float(pred[0][3])}
    #Convert that to a float, in order to overcome a type error
    # pred= float(pred)
    # a = f"Probability of Normal:{(pred[0][0]*100).round(2)}%")
    # b = f"Probability of UC:{(pred[0][1]*100).round(2)}%")
    # c = f"Probability of Polyps:{(pred[0][2]*100).round(2)}%")
    # d = f"Probability of Eso:{(pred[0][3]*100).round(2)}%")
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
