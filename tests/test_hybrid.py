import unittest
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.hybrid import HybridRecommender


class TestHybridRecommender(unittest.TestCase):

    def setUp(self):
        self.recommender = HybridRecommender()
        
        self.user_id = "U0001"
        
        self.mock_content_recs = [
            {"product_id": "P0001", "score": 0.9, "reason": "Content-based similarity"},
            {"product_id": "P0002", "score": 0.8, "reason": "Content-based similarity"},
            {"product_id": "P0003", "score": 0.7, "reason": "Content-based similarity"},
            {"product_id": "P0004", "score": 0.6, "reason": "Content-based similarity"},
            {"product_id": "P0005", "score": 0.5, "reason": "Content-based similarity"},
        ]
        
        self.mock_collab_recs = [
            {"product_id": "P0003", "score": 0.95, "reason": "Collaborative filtering similarity"},
            {"product_id": "P0006", "score": 0.85, "reason": "Collaborative filtering similarity"},
            {"product_id": "P0007", "score": 0.75, "reason": "Collaborative filtering similarity"},
            {"product_id": "P0008", "score": 0.65, "reason": "Collaborative filtering similarity"},
            {"product_id": "P0009", "score": 0.55, "reason": "Collaborative filtering similarity"},
        ]
        
        self.mock_product_details = {
            f"P000{i}": {
                "product_id": f"P000{i}",
                "product_name": f"Test Product {i}",
                "category": "Electronics",
                "price": 100 * i,
            } for i in range(1, 10)
        }

    def test_init(self):
        self.assertIsNotNone(self.recommender.content_recommender)
        self.assertIsNotNone(self.recommender.collaborative_recommender)
        self.assertGreater(self.recommender.content_weight, 0)
        self.assertGreater(self.recommender.collab_weight, 0)
        self.assertAlmostEqual(self.recommender.content_weight + self.recommender.collab_weight, 1.0, places=1)

    @patch('src.models.hybrid.ContentBasedRecommender')
    @patch('src.models.hybrid.CollaborativeFilteringRecommender')
    def test_train(self, MockCollabRecommender, MockContentRecommender):
        mock_content = MagicMock()
        mock_content.train.return_value = True
        MockContentRecommender.return_value = mock_content
        
        mock_collab = MagicMock()
        mock_collab.train.return_value = True
        MockCollabRecommender.return_value = mock_collab
        
        recommender = HybridRecommender()
        
        result = recommender.train()
        
        mock_content.train.assert_called_once()
        mock_collab.train.assert_called_once()
        self.assertTrue(result)

    def test_normalize_scores(self):
        test_recs = [
            {"product_id": "P1", "score": 10, "reason": "Test"},
            {"product_id": "P2", "score": 20, "reason": "Test"},
            {"product_id": "P3", "score": 30, "reason": "Test"},
        ]
        
        normalized = self.recommender._normalize_scores(test_recs)
        
        self.assertEqual(normalized[0]["score"], 0.0)
        self.assertEqual(normalized[2]["score"], 1.0)
        self.assertEqual(normalized[1]["score"], 0.5)
        
        self.assertEqual(self.recommender._normalize_scores([]), [])
        
        single_rec = [{"product_id": "P1", "score": 5, "reason": "Test"}]
        normalized = self.recommender._normalize_scores(single_rec)
        self.assertEqual(normalized[0]["score"], 1.0)

    @patch.object(HybridRecommender, '_normalize_scores')
    def test_recommend(self, mock_normalize):
        self.recommender.content_recommender = MagicMock()
        self.recommender.content_recommender.recommend.return_value = self.mock_content_recs
        
        self.recommender.collaborative_recommender = MagicMock()
        self.recommender.collaborative_recommender.recommend.return_value = self.mock_collab_recs
        
        mock_normalize.side_effect = lambda x: x
        
        recommendations = self.recommender.recommend(self.user_id, top_k=5)
        
        self.assertIsInstance(recommendations, list)
        self.assertEqual(len(recommendations), 5)
        
        has_hybrid = False
        for rec in recommendations:
            if rec["product_id"] == "P0003" and "Hybrid" in rec["reason"]:
                has_hybrid = True
                break
        
        self.assertTrue(has_hybrid, "Should contain a hybrid recommendation for product P0003")

    @patch('src.models.hybrid.save_recommendations')
    def test_generate_recommendations(self, mock_save):
        self.recommender.recommend = MagicMock()
        self.recommender.recommend.return_value = self.mock_content_recs[:3]
        
        mock_save.return_value = True
        
        result = self.recommender.generate_recommendations(self.user_id, top_k=3)
        
        self.recommender.recommend.assert_called_once_with(self.user_id, top_k=3)
        mock_save.assert_called_once()
        self.assertEqual(result, self.mock_content_recs[:3])

    @patch('src.models.hybrid.get_product_details')
    def test_get_formatted_recommendations(self, mock_get_details):
        self.recommender.generate_recommendations = MagicMock()
        self.recommender.generate_recommendations.return_value = self.mock_content_recs[:3]
        
        mock_get_details.return_value = {k: v for k, v in self.mock_product_details.items() if k in ["P0001", "P0002", "P0003"]}
        
        result = self.recommender.get_formatted_recommendations(self.user_id, top_k=3)
        
        self.assertEqual(result["user_id"], self.user_id)
        self.assertEqual(len(result["recommended_products"]), 3)
        
        first_rec = result["recommended_products"][0]
        self.assertEqual(first_rec["product_id"], "P0001")
        self.assertEqual(first_rec["product_name"], "Test Product 1")
        self.assertEqual(first_rec["category"], "Electronics")
        self.assertEqual(first_rec["price"], 100)
        self.assertEqual(first_rec["reason"], "Content-based similarity")


if __name__ == '__main__':
    unittest.main()