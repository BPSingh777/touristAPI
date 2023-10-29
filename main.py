import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import great_circle
from surprise import Dataset, Reader, SVD
import pickle
from fastapi import FastAPI, Query

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
    # Your user input validation and recommendation logic here

    # Sample code for recommendation (replace this with your logic)
    user_location = (latitude, longitude)
    data['distance'] = data.apply(
        lambda row: great_circle(user_location, (row['Latitude'], row['Longitude'])).miles,
        axis=1
    )

    # Normalize the distance and rating for scoring
    scaler = MinMaxScaler()
    data['normalized_distance'] = scaler.fit_transform(data[['distance']])
    data['normalized_rating'] = scaler.fit_transform(data[['Rating']])

    # Add a dummy 'User' column
    data['User'] = 0

    # Generate user-based collaborative filtering recommendations
    recommendations = []
    for index, row in data.iterrows():
        prediction = svd.predict(0, index)  # User ID is assumed to be 0
        recommendations.append({
            'Place': row['Place Name'],
            'Theme': row['Theme'],
            'Rating': row['Rating']
        })

    # Sort destinations by the combined score in descending order
    recommendations = sorted(recommendations, key=lambda x: x['Rating'], reverse=True)

    # Return only the top 10 recommended destinations with numbering
    top_10_destinations = [{"Order": i + 1, **dest} for i, dest in enumerate(recommendations[:10])]

    # Wrap the list with a title
    output = {"Recommended Destinations": top_10_destinations}

    return output

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
