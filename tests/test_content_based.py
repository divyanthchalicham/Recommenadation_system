import unittest
import os
import sys
import random
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.content_based import ContentBasedRecommender


class TestContentBasedRecommender(unittest.TestCase):

    def setUp(self):
        self.recommender = ContentBasedRecommender()
        
        self.mock_products = [
            {
                "product_id": f"P{i:04d}",
                "product_name": f"Test Product {i}",
                "category": random.choice(["Electronics", "Clothing", "Home"]),
                "brand": random.choice(["BrandA", "BrandB", "BrandC"]),
                "price": random.uniform(10, 1000),
                "description": f"This is a test product {i} description."
            }
            for i in range(1, 101)
        ]
        
        self.user_id = "U0001"
        self.mock_activities = [
            {
                "user_id": self.user_id,
                "action_type": "VIEW",
                "product_id": "P0001",
                "timestamp": "2024-01-01T10:00:00Z",
                "category": "Electronics",
                "price": 500
            },
            {
                "user_id": self.user_id,
                "action_type": "BUY",
                "product_id": "P0002",
                "timestamp": "2024-01-01T11:00:00Z",
                "category": "Electronics",
                "price": 600
            }
        ]

    @patch('src.models.content_based.get_all_products')
    def test_train(self, mock_get_all_products):
        mock_get_all_products.return_value = self.mock_products
        
        result = self.recommender.train()
        
        self.assertTrue(result)
        self.assertTrue(self.recommender.is_trained)
        self.assertEqual(len(self.recommender.products), 100)
        self.assertIsNotNone(self.recommender.product_features)
        self.assertEqual(len(self.recommender.product_id_to_index), 100)

    @patch('src.models.content_based.get_user_activity')
    @patch('src.models.content_based.get_all_products')
    def test_recommend(self, mock_get_all_products, mock_get_user_activity):
        mock_get_all_products.return_value = self.mock_products
        mock_get_user_activity.return_value = self.mock_activities
        
        self.recommender.train()
        
        recommendations = self.recommender.recommend(self.user_id, top_k=5)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)
        
        if recommendations:
            first_rec = recommendations[0]
            self.assertIn("product_id", first_rec)
            self.assertIn("score", first_rec)
            self.assertIn("reason", first_rec)
            self.assertEqual(first_rec["reason"], "Content-based similarity")
            self.assertIsInstance(first_rec["score"], float)
            self.assertGreaterEqual(first_rec["score"], 0)
            self.assertLessEqual(first_rec["score"], 1)

    @patch('src.models.content_based.get_user_activity')
    def test_no_user_activity(self, mock_get_user_activity):
        mock_get_user_activity.return_value = []
        
        self.recommender.is_trained = True
        
        recommendations = self.recommender.recommend("non_existent_user")
        
        self.assertEqual(recommendations, [])

    @patch('src.models.content_based.get_all_products')
    def test_no_products(self, mock_get_all_products):
        mock_get_all_products.return_value = []
        
        result = self.recommender.train()
        
        self.assertFalse(result)
        self.assertFalse(self.recommender.is_trained)


if __name__ == '__main__':
    unittest.main()