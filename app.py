from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# ---------------- Load Models and Data ----------------
try:
    # Features used in prediction
    features = ['Name_x', 'State', 'Type', 'BestTimeToVisit', 'Preferences', 
                'Gender', 'NumberOfAdults', 'NumberOfChildren']

    # Load trained model and label encoders
    model = pickle.load(open('code and dataset/model.pkl', 'rb'))
    label_encoders = pickle.load(open('code and dataset/label_encoders.pkl', 'rb'))

    # Load datasets
    destinations_df = pd.read_csv("code and dataset/Expanded_Destinations.csv")
    userhistory_df = pd.read_csv("code and dataset/Final_Updated_Expanded_UserHistory.csv")
    df = pd.read_csv("code and dataset/final_df.csv")

    # Collaborative filtering setup
    user_item_matrix = userhistory_df.pivot(index='UserID', columns='DestinationID', values='ExperienceRating')
    user_item_matrix.fillna(0, inplace=True)
    user_similarity = cosine_similarity(user_item_matrix)

except Exception as e:
    print("Error loading data or model:", e)
    user_item_matrix = pd.DataFrame()
    user_similarity = np.array([])

# ---------------- Collaborative Filtering ----------------
def collaborative_recommend(user_id, user_similarity, user_item_matrix, destinations_df):
    try:
        similar_users = user_similarity[user_id - 1]
        similar_users_idx = np.argsort(similar_users)[::-1][1:6]
        similar_user_ratings = user_item_matrix.iloc[similar_users_idx].mean(axis=0)
        recommended_destinations_ids = similar_user_ratings.sort_values(ascending=False).head(5).index
        recommendations = destinations_df[destinations_df['DestinationID'].isin(recommended_destinations_ids)][[
            'DestinationID', 'Name', 'State', 'Type', 'Popularity', 'BestTimeToVisit'
        ]]
        return recommendations
    except Exception as e:
        print("Error in collaborative_recommend:", e)
        return pd.DataFrame()

# ---------------- Popularity Prediction ----------------
def recommend_destinations(user_input, model, label_encoders, features, data):
    try:
        encoded_input = {}
        for feature in features:
            if feature in label_encoders:
                encoded_input[feature] = label_encoders[feature].transform([user_input[feature]])[0]
            else:
                encoded_input[feature] = user_input[feature]
        input_df = pd.DataFrame([encoded_input])
        predicted_popularity = model.predict(input_df)[0]
        return predicted_popularity
    except Exception as e:
        print("Error in recommend_destinations:", e)
        return None

# ---------------- Routes ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

@app.route("/recommend", methods=['GET', 'POST'])
def recommend():
    if request.method == "POST":
        try:
            user_id = int(request.form['user_id'])
            user_input = {
                'Name_x': request.form['name'],
                'Type': request.form['type'],
                'State': request.form['state'],
                'BestTimeToVisit': request.form['best_time'],
                'Preferences': request.form['preferences'],
                'Gender': request.form['gender'],
                'NumberOfAdults': request.form['adults'],
                'NumberOfChildren': request.form['children'],
            }

            recommended_destinations = collaborative_recommend(user_id, user_similarity, user_item_matrix, destinations_df)
            predicted_popularity = recommend_destinations(user_input, model, label_encoders, features, df)

            return render_template('recommendation.html',
                                   recommended_destinations=recommended_destinations,
                                   predicted_popularity=predicted_popularity)
        except Exception as e:
            print("Error in /recommend route:", e)
            return render_template('recommendation.html',
                                   recommended_destinations=pd.DataFrame(),
                                   predicted_popularity=None)

    return render_template('recommendation.html')

# ---------------- Run the app ----------------
if __name__ == "__main__":
    # Render automatically sets PORT
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
