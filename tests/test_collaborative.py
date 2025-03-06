import unittest
import os
import sys
import random
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.collaborative import CollaborativeFilteringRecommender


class TestCollaborativeFilteringRecommender(unittest.TestCase):

    def setUp(self):
        self.recommender = CollaborativeFilteringRecommender(num_factors=10)
        
        self.user_ids = [f"U{i:04d}" for i in range(1, 21)]
        
        self.product_ids = [f"P{i:04d}" for i in range(1, 51)]
        
        self.mock_activities = []
        for user_id in self.user_ids:
            num_interactions = random.randint(5, 10)
            products = random.sample(self.product_ids, num_interactions)
            
            for product_id in products:
                action_type = "BUY" if random.random() < 0.2 else "VIEW"
                
                self.mock_activities.append({
                    "user_id": user_id,
                    "action_type": action_type,
                    "product_id": product_id,
                    "timestamp": "2024-01-01T10:00:00Z",
                    "category": random.choice(["Electronics", "Clothing", "Home"]),
                    "price": random.uniform(10, 1000)
                })

    @patch('src.models.collaborative.get_all_user_ids')
    @patch('src.models.collaborative.get_user_activity')
    def test_train(self, mock_get_user_activity, mock_get_all_user_ids):
        mock_get_all_user_ids.return_value = self.user_ids
        mock_get_user_activity.side_effect = lambda user_id: [
            activity for activity in self.mock_activities if activity["user_id"] == user_id
        ]
        
        result = self.recommender.train()
        
        self.assertTrue(result)
        self.assertTrue(self.recommender.is_trained)
        self.assertIsNotNone(self.recommender.user_item_matrix)
        self.assertIsNotNone(self.recommender.user_factors)
        self.assertIsNotNone(self.recommender.item_factors)
        self.assertEqual(len(self.recommender.user_id_to_index), len(self.user_ids))

    @patch('src.models.collaborative.get_all_user_ids')
    @patch('src.models.collaborative.get_user_activity')
    def test_recommend(self, mock_get_user_activity, mock_get_all_user_ids):
        mock_get_all_user_ids.return_value = self.user_ids
        mock_get_user_activity.side_effect = lambda user_id: [
            activity for activity in self.mock_activities if activity["user_id"] == user_id
        ]
        
        self.recommender.train()
        
        user_id = self.user_ids[0]
        recommendations = self.recommender.recommend(user_id, top_k=5)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)
        
        if recommendations:
            first_rec = recommendations[0]
            self.assertIn("product_id", first_rec)
            self.assertIn("score", first_rec)
            self.assertIn("reason", first_rec)
            self.assertEqual(first_rec["reason"], "Collaborative filtering similarity")
            self.assertIsInstance(first_rec["score"], float)
            self.assertGreaterEqual(first_rec["score"], 0)
            self.assertLessEqual(first_rec["score"], 1)

    @patch('src.models.collaborative.get_all_user_ids')
    @patch('src.models.collaborative.get_user_activity')
    def test_unknown_user(self, mock_get_user_activity, mock_get_all_user_ids):
        mock_get_all_user_ids.return_value = self.user_ids
        mock_get_user_activity.side_effect = lambda user_id: [
            activity for activity in self.mock_activities if activity["user_id"] == user_id
        ]
        
        self.recommender.train()
        
        unknown_user = "U9999"
        recommendations = self.recommender.recommend(unknown_user)
        
        self.assertEqual(recommendations, [])

    @patch('src.models.collaborative.get_all_user_ids')
    def test_no_users(self, mock_get_all_user_ids):
        mock_get_all_user_ids.return_value = []
        
        result = self.recommender.train()
        
        self.assertFalse(result)
        self.assertFalse(self.recommender.is_trained)


if __name__ == '__main__':
    unittest.main()