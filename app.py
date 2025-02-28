import pandas as pd
import numpy as np
import pickle  # To save/load models
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv('Mall_Customers.csv')

# Train Regression Model
X = data[['Age', 'Annual Income (k$)']]
y = data['Spending Score (1-100)']
model = LinearRegression()
model.fit(X, y)

# Save model
with open('regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Train Clustering Model
X_cluster = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Save clustering model
with open('clustering_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# Flask API
app = Flask(__name__)

@app.route('/')
def home():
    return "Customer Segmentation & Spending Score Prediction API"

@app.route('/predict', methods=['POST'])
def predict_spending():
    data = request.get_json()
    X_new = np.array([[data['Age'], data['Annual Income']]])
    
    with open('regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    spending_score = model.predict(X_new)[0]
    return jsonify({'Predicted Spending Score': spending_score})

@app.route('/cluster', methods=['POST'])
def cluster_customer():
    data = request.get_json()
    X_new = np.array([[data['Age'], data['Annual Income'], data['Spending Score']]])
    
    with open('clustering_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
        
    cluster = kmeans.predict(X_new)[0]
    return jsonify({'Cluster': int(cluster)})

if __name__ == '__main__':
    app.run(debug=True)
