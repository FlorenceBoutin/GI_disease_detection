import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ml_logic.model import dummy_model


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

#Endpoint applied when prediction button is clicked
@app.get("/predict")
def predict():
    predict = dummy_model()
    return predict

# predict = dummy_model()


# Endpoint to confirm communication with the server (root)
@app.get("/")
def root():
    # $CHA_BEGIN
    return "It's working! WOOP! WOOP!"
    # $CHA_END
