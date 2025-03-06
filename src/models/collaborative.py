
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from src.database.mongo_handler import get_user_activity, get_all_users
from src.config import TOP_K_RECOMMENDATIONS

class CollaborativeFilteringRecommender:
   
    
    def __init__(self, num_factors=20):
        self.num_factors = num_factors
        self.user_id_to_index = {}
        self.product_id_to_index = {}
        self.index_to_user_id = {}
        self.index_to_product_id = {}
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0
        self.is_trained = False
    
    def _create_user_item_matrix(self, user_activities):
        
        user_ids = sorted(set(a["user_id"] for a in user_activities))
        product_ids = sorted(set(a["product_id"] for a in user_activities))
        
        
        self.user_id_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
        self.product_id_to_index = {product_id: i for i, product_id in enumerate(product_ids)}
        self.index_to_user_id = {i: user_id for i, user_id in enumerate(user_ids)}
        self.index_to_product_id = {i: product_id for i, product_id in enumerate(product_ids)}
        
     
        rows, cols, data = [], [], []
        
        for activity in user_activities:
            user_idx = self.user_id_to_index[activity["user_id"]]
            product_idx = self.product_id_to_index[activity["product_id"]]
            
            
            weight = 5.0 if activity["action_type"] == "BUY" else 1.0
            
            rows.append(user_idx)
            cols.append(product_idx)
            data.append(weight)
        

        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_ids), len(product_ids))
        )
        
        return matrix
    
    def train(self):
        """Train collaborative filtering model."""
       
        user_ids = get_all_users()
        
        if not user_ids:
            print("No users found in database.")
            return False
        
       
        all_activities = []
        for user_id in user_ids:
            activities = get_user_activity(user_id=user_id)
            all_activities.extend(activities)
        
        if not all_activities:
            print("No user activities found in database.")
            return False
        
       
        self.user_item_matrix = self._create_user_item_matrix(all_activities)
        
       
        self.global_mean = np.mean(self.user_item_matrix.data) if self.user_item_matrix.nnz > 0 else 0
        
       
        matrix_dense = self.user_item_matrix.toarray()
        mask = matrix_dense != 0
        matrix_dense[mask] -= self.global_mean
        
      
        k = min(self.num_factors, min(matrix_dense.shape) - 1)
        U, sigma, Vt = svds(matrix_dense, k=k)
        
       
        self.user_factors = U
        self.item_factors = Vt.T
        
        self.is_trained = True
        print(f"Collaborative filtering trained on {self.user_item_matrix.shape[0]} users and {self.user_item_matrix.shape[1]} products.")
        
        return True
    
    def recommend(self, user_id, top_k=TOP_K_RECOMMENDATIONS):
      
        if not self.is_trained:
            if not self.train():
                return []
        
       
        if user_id not in self.user_id_to_index:
            print(f"User {user_id} not found in training data.")
            return []
        
        
        user_idx = self.user_id_to_index[user_id]
        
   
        user_interactions = self.user_item_matrix[user_idx].toarray().flatten()
        interacted_items = set(np.where(user_interactions > 0)[0])
        

        user_vector = self.user_factors[user_idx]
        predicted_ratings = self.global_mean + np.dot(user_vector, self.item_factors.T)
        
       
        product_scores = []
        for product_idx, score in enumerate(predicted_ratings):
           
            if product_idx in interacted_items:
                continue
            
         
            if product_idx < len(self.index_to_product_id):
                product_id = self.index_to_product_id[product_idx]
                product_scores.append((product_id, float(score)))
        
        
        product_scores.sort(key=lambda x: x[1], reverse=True)
        
       
        recommendations = []
        for product_id, score in product_scores[:top_k]:
            # Normalize score to be between 0 and 1
            norm_score = min(max(score / 5.0, 0), 1)
            
            recommendations.append({
                "product_id": product_id,
                "score": norm_score,
                "reason": "Collaborative filtering similarity"
            })
        
        return recommendations