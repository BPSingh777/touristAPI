import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import great_circle
from fastapi import FastAPI, Query
import pickle

app = FastAPI()

# Load the dataset (you should adjust the path to your dataset)
data = pd.read_csv('hpdataset.csv')

# Load the SVD model from a pickle file
with open('svd_model.pkl', 'rb') as svd_file:
    svd = pickle.load(svd_file)

# Define a route for recommending destinations
@app.post("/recommend")
async def recommend_destination(
    theme: str = Query(..., description="Preferred theme"),
    rating: float = Query(..., description="Preferred rating (0-5)"),
    days: int = Query(..., description="Number of days for travel"),
    latitude: float = Query(..., description="Current latitude"),
    longitude: float = Query(..., description="Current longitude")
):
    # Calculate user profile
    user_location = (latitude, longitude)
    data['distance'] = data.apply(
        lambda row: great_circle(user_location, (row['Latitude'], row['Longitude'])).miles,
        axis=1
    )

    # Normalize the distance and rating for scoring
    scaler = MinMaxScaler()
    data['normalized_distance'] = scaler.fit_transform(data[['distance']])
    data['normalized_rating'] = scaler.fit_transform(data[['Rating']])

    # Generate user-based collaborative filtering recommendations
    recommendations = []
    for index, row in data.iterrows():
        prediction = svd.predict(0, index)  # User ID is assumed to be 0
        recommendations.append({
            'Place': row['Place Name'],
             # Convert to a float
            'Place_ID': int(index),  # Convert to an integer
            'Predicted_Rating': prediction.est,
        
        })

    # Sort destinations by rating in descending order
    collaborative_recommendations_df = pd.DataFrame(recommendations)
    recommended_destinations = data.merge(collaborative_recommendations_df, left_index=True, right_on='Place_ID', how='left')

    # Calculate theme matching score (higher weight for the same theme)
    theme_weight = 0.3

    # Calculate the final score by combining the rating and theme matching score
    recommended_destinations['Combined_Score'] = (
        0.2 * recommended_destinations['normalized_rating'] +
        0.6 * (1 - recommended_destinations['normalized_distance']) +
        0.2 * recommended_destinations['Predicted_Rating'] +
        theme_weight * (recommended_destinations['Theme'] == theme).astype(float)  # Convert to float
    )

    # Sort recommendations by the combined score in descending order
    recommended_destinations = recommended_destinations.sort_values(by='Combined_Score', ascending=False)

    # Return only the top 10 recommended destinations with numbering
    top_10_destinations = recommended_destinations.head(10)
    top_10_destinations = top_10_destinations[['District','Place Name','Theme','Rating']]
    # Wrap the list with a title
    output = {"Recommended Destinations": top_10_destinations.to_dict(orient='records')}  # Convert DataFrame to a list of dictionaries

    return output

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
