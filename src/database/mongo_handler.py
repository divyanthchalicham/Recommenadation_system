
from pymongo import MongoClient, ASCENDING
from datetime import datetime

from src.config import MONGO_URI, MONGO_DB, COLLECTION_PRODUCTS, COLLECTION_USER_ACTIVITY, COLLECTION_RECOMMENDATIONS


client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

def init_db():

    try:

        db[COLLECTION_PRODUCTS].drop_indexes()
        db[COLLECTION_USER_ACTIVITY].drop_indexes()
        db[COLLECTION_RECOMMENDATIONS].drop_indexes()
    
        db[COLLECTION_PRODUCTS].create_index([("product_id", ASCENDING)], unique=True)
        

        db[COLLECTION_USER_ACTIVITY].create_index([("user_id", ASCENDING)])
        db[COLLECTION_USER_ACTIVITY].create_index([("product_id", ASCENDING)])
        db[COLLECTION_USER_ACTIVITY].create_index([("timestamp", ASCENDING)])
        
       
        db[COLLECTION_RECOMMENDATIONS].create_index([("user_id", ASCENDING)], unique=True)
        
        print("Database indexes initialized successfully")
    except Exception as e:
        print(f"Error initializing database indexes: {e}")
    
        pass

def insert_products(products):
 
    if not products:
        return False
    

    for product in products:
        db[COLLECTION_PRODUCTS].update_one(
            {"product_id": product["product_id"]},
            {"$set": product},
            upsert=True
        )
    
    return True

def insert_activity(activity):

    if not activity:
        return False
    

    if "timestamp" not in activity:
        activity["timestamp"] = datetime.now().isoformat()
    
    db[COLLECTION_USER_ACTIVITY].insert_one(activity)
    return True

def insert_activities(activities):

    if not activities:
        return False
    
    db[COLLECTION_USER_ACTIVITY].insert_many(activities)
    return True

def save_recommendations(user_id, recommendations):

    if not recommendations:
        return False
    
    recommendation_doc = {
        "user_id": user_id,
        "recommended_products": recommendations,
        "timestamp": datetime.now().isoformat()
    }
    
    # Upsert to ensure one document per user
    db[COLLECTION_RECOMMENDATIONS].update_one(
        {"user_id": user_id},
        {"$set": recommendation_doc},
        upsert=True
    )
    
    return True

def get_user_activity(user_id=None, limit=None):

    query = {}
    if user_id:
        query["user_id"] = user_id
    
    cursor = db[COLLECTION_USER_ACTIVITY].find(query).sort("timestamp", -1)
    
    if limit:
        cursor = cursor.limit(limit)
    
    return list(cursor)

def get_all_users():
   
    return db[COLLECTION_USER_ACTIVITY].distinct("user_id")

def get_all_products():
 
    return list(db[COLLECTION_PRODUCTS].find())

def get_product_details(product_ids):
  
    if not product_ids:
        return {}
    
    products = db[COLLECTION_PRODUCTS].find({"product_id": {"$in": product_ids}})
    
 
    product_map = {}
    for product in products:
        product_map[product["product_id"]] = product
    
    return product_map

def get_recommendations(user_id):
   
    return db[COLLECTION_RECOMMENDATIONS].find_one({"user_id": user_id})