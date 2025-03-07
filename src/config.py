import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "ecommerce_recommendations")

COLLECTION_PRODUCTS = "products"
COLLECTION_USER_ACTIVITY = "user_activity"
COLLECTION_RECOMMENDATIONS = "recommendations"

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

TOP_K_RECOMMENDATIONS = 10

CONTENT_BASED_WEIGHT = 0.6
COLLABORATIVE_WEIGHT = 0.4

CONTENT_MIN_INTERACTIONS = 2

COLLAB_NUM_FACTORS = 20

PRECISION_K = 5
EVALUATION_TEST_SIZE = 0.2

NUM_USERS = 100
NUM_PRODUCTS = 500
NUM_CATEGORIES = 10
NUM_BRANDS = 20
SIMULATION_DAYS = 30
AVG_ACTIONS_PER_USER_PER_DAY = 5