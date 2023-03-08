import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#Create a variable app for FastAPI
app = FastAPI()

@app.get("/predict")

# Optional, good practice for dev purposes. Allow all middlewares
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
#     )

# End point to confirm communication with the server
@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END
