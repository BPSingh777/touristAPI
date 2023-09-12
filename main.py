# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 00:00:52 2023

@author: hp
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import haversine_distances
from geopy.distance import great_circle
from fastapi import FastAPI, HTTPException
import pickle

app = FastAPI()

# Load the machine learning model
loaded_model = pickle.load(open('una_tourist.sav', 'rb'))

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
        loaded_model['distance'] = loaded_model.apply(
            lambda row: great_circle(user_location, (row['Latitude'], row['Longitude'])).miles,
            axis=1
        )
        scaler = MinMaxScaler()
        # Normalize the distance and rating for scoring
        loaded_model['normalized_distance'] = scaler.fit_transform(loaded_model[['distance']])
        loaded_model['normalized_rating'] = scaler.fit_transform(loaded_model[['Rating']])

        # Calculate a composite score for each destination based on content attributes
        loaded_model['content_score'] = 0.6 * loaded_model['normalized_rating'] + 0.4 * (1 - loaded_model['normalized_distance'])

        # Filter destinations based on content attributes
        filtered_data = loaded_model[(loaded_model['Theme'] == user_profile['theme']) &
                                     (loaded_model['Rating'] >= user_profile['rating'])]

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
