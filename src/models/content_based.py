import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.database.mongo_handler import get_all_products, get_user_activity
from src.config import TOP_K_RECOMMENDATIONS

class ContentBasedRecommender:
    
    def __init__(self):
        self.products = []
        self.product_features = None
        self.product_id_to_index = {}
        self.index_to_product_id = {}
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.is_trained = False
    
    def _preprocess_products(self):
        if not self.products:
            return False
        
        self.product_id_to_index = {p["product_id"]: i for i, p in enumerate(self.products)}
        self.index_to_product_id = {i: p["product_id"] for i, p in enumerate(self.products)}
        
        product_texts = []
        for product in self.products:
            text = f"{product['category']} {product['brand']} {product.get('description', '')}"
            
            price = product['price']
            if price < 50:
                price_range = "affordable cheap budget"
            elif price < 200:
                price_range = "mid-range moderate standard"
            else:
                price_range = "premium expensive luxury"
            
            text = f"{text} {price_range} {price_range}"
            
            product_texts.append(text)
        
        self.product_features = self.tfidf_vectorizer.fit_transform(product_texts)
        
        return True
    
    def train(self):
        self.products = get_all_products()
        
        if not self.products:
            print("No products found in database.")
            return False
        
        success = self._preprocess_products()
        
        if success:
            self.is_trained = True
            print(f"Content-based recommender trained on {len(self.products)} products.")
        
        return success
    
    def _get_user_product_interactions(self, user_id):
        user_activity = get_user_activity(user_id=user_id)
        
        if not user_activity:
            return {}
        
        product_weights = {}
        for activity in user_activity:
            product_id = activity["product_id"]
            weight = 3.0 if activity["action_type"] == "BUY" else 1.0
            
            if product_id in product_weights:
                product_weights[product_id] += weight
            else:
                product_weights[product_id] = weight
        
        return product_weights
    
    def recommend(self, user_id, top_k=TOP_K_RECOMMENDATIONS):
        if not self.is_trained:
            if not self.train():
                return []
        
        product_weights = self._get_user_product_interactions(user_id)
        
        if not product_weights:
            print(f"No interaction data for user {user_id}")
            return []
        
        user_profile = np.zeros(self.product_features.shape[1])
        total_weight = 0
        
        for product_id, weight in product_weights.items():
            if product_id in self.product_id_to_index:
                idx = self.product_id_to_index[product_id]
                user_profile += weight * self.product_features[idx].toarray().flatten()
                total_weight += weight
        
        if total_weight > 0:
            user_profile /= total_weight
        
        similarities = cosine_similarity(
            user_profile.reshape(1, -1),
            self.product_features
        ).flatten()
        
        product_scores = []
        for i, similarity in enumerate(similarities):
            product_id = self.index_to_product_id[i]
            
            if product_id in product_weights:
                continue
            
            product_scores.append((product_id, similarity))
        
        product_scores.sort(key=lambda x: x[1], reverse=True)
        
        max_score = max([score for _, score in product_scores[:top_k]]) if product_scores else 1.0
        recommendations = []
        for product_id, score in product_scores[:top_k]:
            normalized_score = 0.3 + (score / max_score) * 0.7 if max_score > 0 else 0.3
            recommendations.append({
                "product_id": product_id,
                "score": float(normalized_score),
                "reason": "Content-based similarity"
            })
        return recommendations