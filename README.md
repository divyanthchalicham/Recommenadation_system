# Recommenadation_system
  
  A machine learning-based recommendation system for e-commerce platforms that suggests products to users based on their browsing and purchase history. The system combines content-based filtering and collaborative filtering approaches to provide personalized product recommendations.

**Features:**
Content-Based Filtering: Recommends products with similar attributes to those the user has interacted with.
Collaborative Filtering: Recommends products based on similar user behavior using matrix factorization.
Hybrid Approach: Combines both methods using a weighted average for better recommendations.
Real-time Data Ingestion: Simulates streaming of user activity data.
REST API: Provides endpoints for retrieving recommendations and adding user activities.
Database Integration: Stores data and recommendations in MongoDB.
Evaluation Metrics: Measures performance using precision@k at both product and category levels.

**Setting up the system:** 

1)Installing python and mongodb locally

2)**MongoDB setup:**
	brew tap mongodb/brew
  brew install mongodb-community

2) **Navigating to the directory:**
	cd project_folder location

3)**Create a virtual environment:**
	python3 -m venv venv 
	source venv/bin/activate (for windows:venv\Scripts\activate)

4)**Installing dependencies:**
	pip install -r requirements.txt

5)**In terminal 1 :  (DATA INGESTION)**
	Generate simulated product and user activity data
  python -m src.data_simulation.data_generator

6)**In terminal 1: Start API server**
python -m src.api.app

7) **In new terminal terminal 2: (Run Data Ingestion Simulation)**
	Using the API endpoint (in a new terminal):
	curl -X POST "http://localhost:8000/start-streaming?speed_factor=100"

8)**Getting recommendation:**
	# Replace USER_ID with an actual user ID from the data
curl -X GET "http://localhost:8000/recommendations/USER_ID?limit=10"

9)**Evaluate Model Performance :**
# Replace USER_ID with an actual user ID from the data
curl -X GET "http://localhost:8000/recommendations/USER_ID?limit=10"
python -m src.evaluation.metrics

**10) Running Tests : **


# Run specific test files
python -m unittest tests.test_content_based
python -m unittest tests.test_collaborative
python -m unittest tests.test_hybrid
python -m unittest tests.test_api

**Output :**




**Ingestion of data:** 
<img width="602" alt="image" src="https://github.com/user-attachments/assets/bded0552-11e0-49c3-a9e6-376da643be3c" />


 **Get Recommendations:**
<img width="1194" alt="image" src="https://github.com/user-attachments/assets/a743cac0-cef4-4fa4-9a03-9ab6cb1586a6" />



**Evaluation Metrics:**
<img width="1449" alt="image" src="https://github.com/user-attachments/assets/8a5de148-c6f1-43de-86de-f7b8f9a6ce22" />

**Exact Precision:** 
Calculates System isn't recommending the exact same products users eventually interacted with

**Category Precision:**
 Calaculates recommended products are in the same categories that users actually showed interest in.
