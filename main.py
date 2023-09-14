# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 00:00:52 2023

@author: hp
"""
from fastapi import FastAPI, HTTPException
from geopy.distance import great_circle
import pickle

app = FastAPI()

# Load the machine learning model during application startup
loaded_model = None

def load_model():
    global loaded_model
    try:
        if loaded_model is None:
            loaded_model = pickle.load(open('una_tourist.sav', 'rb'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Ensure the model is loaded when the application starts
load_model()

@app.get("/")
def read_root():
    return {"message": "Welcome to the recommendation API!"}

@app.post("/recommend")
async def recommend_destination(
    theme: str,
    rating: float,
    days: int,
    latitude: float,
    longitude: float,
):
    try:
        # Create a user profile
        user_profile = {
            'theme': theme,
            'rating': rating,
            'days': days
        }

        # Calculate distances between user location and destinations
        user_location = (latitude, longitude)

        # Initialize data from the loaded model
        data = loaded_model  # Replace this with your actual data structure

        # ... (rest of your code)

        # Filter destinations based on content attributes
        filtered_data = data[(data['Theme'] == user_profile['theme']) &
                             (data['Rating'] >= user_profile['rating'])]

        # Sort destinations by content-based score in descending order
        recommended_destinations = filtered_data.sort_values(by='content_score', ascending=False)

        # Extract and return the top recommendations
        recommendations = recommended_destinations[['Places', 'Theme', 'Rating']].to_dict(orient='records')

        return {"recommendations": recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
