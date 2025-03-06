from collections import defaultdict
from datetime import datetime

from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeFilteringRecommender
from src.database.mongo_handler import save_recommendations, get_product_details
from src.config import TOP_K_RECOMMENDATIONS, CONTENT_BASED_WEIGHT, COLLABORATIVE_WEIGHT

class HybridRecommender:
    
    def __init__(self):
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeFilteringRecommender()
        self.content_weight = CONTENT_BASED_WEIGHT
        self.collab_weight = COLLABORATIVE_WEIGHT
    
    def train(self):
        print("Training content-based model...")
        content_trained = self.content_recommender.train()
        print(f"Content-based model trained: {content_trained}")
        
        print("Training collaborative filtering model...")
        collab_trained = self.collaborative_recommender.train()
        print(f"Collaborative model trained: {collab_trained}")
        
        return content_trained and collab_trained
    
    def _normalize_scores(self, recommendations):
        if not recommendations:
            return []
            
        max_score = max(rec["score"] for rec in recommendations)
        min_score = min(rec["score"] for rec in recommendations)
        
        normalized_recs = []
        for rec in recommendations:
            rec_copy = rec.copy()
            if max_score > min_score:
                rec_copy["score"] = (rec["score"] - min_score) / (max_score - min_score)
            else:
                rec_copy["score"] = 1.0 if rec["score"] > 0 else 0.0
            normalized_recs.append(rec_copy)
            
        return normalized_recs
    
    def recommend(self, user_id, top_k=TOP_K_RECOMMENDATIONS):
        content_recs = self.content_recommender.recommend(user_id, top_k=top_k*2)
        collab_recs = self.collaborative_recommender.recommend(user_id, top_k=top_k*2)
        
        print(f"Content-based recommendations for {user_id}: {len(content_recs)}")
        if content_recs:
            for i, rec in enumerate(content_recs[:3]):
                print(f"  {i+1}. {rec['product_id']} - Score: {rec['score']}")
                
        print(f"Collaborative recommendations for {user_id}: {len(collab_recs)}")
        if collab_recs:
            for i, rec in enumerate(collab_recs[:3]):
                print(f"  {i+1}. {rec['product_id']} - Score: {rec['score']}")
        
        content_recs = self._normalize_scores(content_recs)
        collab_recs = self._normalize_scores(collab_recs)
        
        content_dict = {rec["product_id"]: rec["score"] for rec in content_recs}
        collab_dict = {rec["product_id"]: rec["score"] for rec in collab_recs}
        
        all_product_ids = set(content_dict.keys()) | set(collab_dict.keys())
        
        final_scores = {}
        final_reasons = {}
        content_count = 0
        collab_count = 0
        hybrid_count = 0
        
        for product_id in all_product_ids:
            content_score = content_dict.get(product_id, 0)
            collab_score = collab_dict.get(product_id, 0)
            
            weighted_avg = (content_score * self.content_weight) + (collab_score * self.collab_weight)
            final_scores[product_id] = weighted_avg
            
            if content_score > 0 and collab_score > 0:
                final_reasons[product_id] = "Hybrid: Content + Collaborative"
                hybrid_count += 1
            elif content_score > 0:
                final_reasons[product_id] = "Content-based similarity"
                content_count += 1
            else:
                final_reasons[product_id] = "Collaborative filtering similarity"
                collab_count += 1
        
        sorted_products = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for product_id, score in sorted_products[:top_k]:
            recommendations.append({
                "product_id": product_id,
                "score": float(score),
                "reason": final_reasons[product_id]
            })
        
        print(f"Final recommendations: Content: {content_count}, Collaborative: {collab_count}, Hybrid: {hybrid_count}")
        return recommendations
    
    def generate_recommendations(self, user_id, top_k=TOP_K_RECOMMENDATIONS):
        recommendations = self.recommend(user_id, top_k=top_k)
        
        if not recommendations:
            return []
        
        save_recommendations(user_id, recommendations)
        
        return recommendations
    
    def get_formatted_recommendations(self, user_id, top_k=TOP_K_RECOMMENDATIONS):
        recommendations = self.generate_recommendations(user_id, top_k=top_k)
        
        if not recommendations:
            return {
                "user_id": user_id,
                "recommended_products": []
            }
        
        product_ids = [rec["product_id"] for rec in recommendations]
        product_details = get_product_details(product_ids)
        
        formatted_recs = []
        for rec in recommendations:
            product_id = rec["product_id"]
            
            if product_id in product_details:
                product = product_details[product_id]
                
                formatted_rec = {
                    "product_id": product_id,
                    "product_name": product["product_name"],
                    "score": rec["score"],
                    "category": product["category"],
                    "price": product["price"],
                    "reason": rec["reason"]
                }
                
                formatted_recs.append(formatted_rec)
        
        return {
            "user_id": user_id,
            "recommended_products": formatted_recs
        }