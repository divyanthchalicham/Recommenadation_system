
import numpy as np
from datetime import datetime

from src.database.mongo_handler import get_user_activity, get_all_users, get_product_details
from src.models.hybrid import HybridRecommender
from src.config import PRECISION_K

def precision_at_k(actual, predicted, k):
    
    predicted_k = predicted[:k]
    
   
    num_relevant = sum(1 for item in predicted_k if item in actual)
    
    
    print(f"Actual items (first 5): {actual[:5]}")
    print(f"Predicted top-{k}: {predicted_k}")
    print(f"Relevant items in predictions: {num_relevant}/{k}")
    
    return num_relevant / k if k > 0 else 0

def category_precision_at_k(actual_ids, predicted_ids, k):
    
    all_ids = list(set(actual_ids + predicted_ids[:k]))
    product_details = get_product_details(all_ids)
    
    
    actual_categories = set()
    for pid in actual_ids:
        if pid in product_details and 'category' in product_details[pid]:
            actual_categories.add(product_details[pid]['category'])
    

    category_matches = 0
    for pid in predicted_ids[:k]:
        if pid in product_details and 'category' in product_details[pid]:
            if product_details[pid]['category'] in actual_categories:
                category_matches += 1
    
    print(f"Category matches: {category_matches}/{k}")
    return category_matches / k if k > 0 else 0

def split_user_activities(test_size=0.2, action_type=None, min_activities=10):
    """Split user activities into training and testing sets."""
    
    user_ids = get_all_users()
    
    holdout_data = {}
    
    for user_id in user_ids:
        
        activities = get_user_activity(user_id=user_id)
        
        
        if action_type:
            activities = [a for a in activities if a["action_type"] == action_type]
        
        if len(activities) < min_activities:  
            continue
        
       
        products = {}
        for activity in activities:
            pid = activity["product_id"]
            if pid not in products or activity["timestamp"] > products[pid]["timestamp"]:
                products[pid] = activity
        
        
        if len(products) < 5:
            continue
            
       
        sorted_products = sorted(products.values(), key=lambda x: x["timestamp"])
        
       
        n_test = max(1, int(len(sorted_products) * test_size))
        holdout_activities = sorted_products[-n_test:]
        
       
        holdout_products = [a["product_id"] for a in holdout_activities]
        holdout_data[user_id] = holdout_products
    
    print(f"Created holdout set with {len(holdout_data)} users")
    if holdout_data:
        sample_user = next(iter(holdout_data))
        print(f"Sample user {sample_user} has {len(holdout_data[sample_user])} held-out products")
    
    return holdout_data

def evaluate_recommender(recommender, holdout_data, k=PRECISION_K):
    """Evaluate the recommendation system using precision@k."""
    exact_precisions = []
    category_precisions = []
    
    for user_id, actual_products in holdout_data.items():
        print(f"\nEvaluating user: {user_id}")
       
        recommendations = recommender.recommend(user_id, top_k=max(k, 20))
        
        
        predicted_products = [rec["product_id"] for rec in recommendations]
        
        
        exact_p_at_k = precision_at_k(actual_products, predicted_products, k)
        
        
        category_p_at_k = category_precision_at_k(actual_products, predicted_products, k)
        
        print(f"Exact Precision@{k}: {exact_p_at_k:.4f}, Category Precision@{k}: {category_p_at_k:.4f}")
        
        exact_precisions.append(exact_p_at_k)
        category_precisions.append(category_p_at_k)
    
   
    avg_exact = np.mean(exact_precisions) if exact_precisions else 0
    avg_category = np.mean(category_precisions) if category_precisions else 0
    
    return avg_exact, avg_category

def run_evaluation():
    """Run evaluation with additional metrics."""
    print("Training recommendation models...")
    recommender = HybridRecommender()
    recommender.train()
    

    print("\nEvaluating with stricter user filtering (more activities)...")
    all_holdout = split_user_activities(test_size=0.2, action_type=None, min_activities=10)
    
    if all_holdout:

        for k_value in [5, 10, 20]:
            exact_precision, category_precision = evaluate_recommender(recommender, all_holdout, k=k_value)
            print(f"\nResults for k={k_value}:")
            print(f"  Exact Precision@{k_value}: {exact_precision:.4f}")
            print(f"  Category Precision@{k_value}: {category_precision:.4f}")
    else:
        print("Insufficient data for evaluation.")
    
    return {
        "exact_precision": exact_precision if 'exact_precision' in locals() else None,
        "category_precision": category_precision if 'category_precision' in locals() else None,
        "k": k_value if 'k_value' in locals() else PRECISION_K,
        "num_users": len(all_holdout) if 'all_holdout' in locals() else 0
    }

if __name__ == "__main__":
    run_evaluation()