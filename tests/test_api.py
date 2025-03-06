import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.app import app


class TestRecommendationAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        
        self.user_id = "U0001"
        
        self.mock_recommendations = {
            "user_id": self.user_id,
            "recommended_products": [
                {
                    "product_id": "P0001",
                    "product_name": "Test Product 1",
                    "score": 0.9,
                    "category": "Electronics",
                    "price": 499.99,
                    "reason": "Hybrid: Content + Collaborative"
                },
                {
                    "product_id": "P0002",
                    "product_name": "Test Product 2",
                    "score": 0.8,
                    "category": "Clothing",
                    "price": 99.99,
                    "reason": "Content-based similarity"
                }
            ]
        }
        
        self.activity_data = {
            "user_id": self.user_id,
            "action_type": "VIEW",
            "product_id": "P0003",
            "category": "Electronics",
            "price": 299.99
        }

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
        self.assertIn("running", response.json()["message"])

    @patch("src.api.routes.Request")
    def test_get_recommendations(self, MockRequest):
        mock_recommender = MagicMock()
        mock_recommender.get_formatted_recommendations.return_value = self.mock_recommendations
        
        mock_request = MagicMock()
        mock_request.app.state.recommender = mock_recommender
        MockRequest.return_value = mock_request
        
        response = self.client.get(f"/recommendations/{self.user_id}")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user_id"], self.user_id)
        self.assertEqual(len(response.json()["recommended_products"]), 2)
        
        first_rec = response.json()["recommended_products"][0]
        self.assertEqual(first_rec["product_id"], "P0001")
        self.assertEqual(first_rec["product_name"], "Test Product 1")
        self.assertEqual(first_rec["category"], "Electronics")
        
        mock_recommender.get_formatted_recommendations.assert_called_once_with(self.user_id, top_k=10)

    @patch("src.api.routes.insert_activity")
    def test_add_activity(self, mock_insert):
        mock_insert.return_value = True
        
        response = self.client.post("/activity", json=self.activity_data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
        self.assertIn("added successfully", response.json()["message"])
        
        mock_insert.assert_called_once()
        call_args = mock_insert.call_args[0][0]
        self.assertEqual(call_args["user_id"], self.activity_data["user_id"])
        self.assertEqual(call_args["action_type"], self.activity_data["action_type"])
        self.assertEqual(call_args["product_id"], self.activity_data["product_id"])

    @patch("src.api.routes.Request")
    def test_streaming_endpoints(self, MockRequest):
        mock_service = MagicMock()
        mock_service.start_streaming.return_value = True
        mock_service.stop_streaming.return_value = True
        
        mock_request = MagicMock()
        mock_request.app.state.stream_service = mock_service
        MockRequest.return_value = mock_request
        
        start_response = self.client.post("/start-streaming?speed_factor=50")
        self.assertEqual(start_response.status_code, 200)
        self.assertIn("message", start_response.json())
        self.assertIn("started", start_response.json()["message"])
        mock_service.start_streaming.assert_called_once_with(speed_factor=50)
        
        stop_response = self.client.post("/stop-streaming")
        self.assertEqual(stop_response.status_code, 200)
        self.assertIn("message", stop_response.json())
        self.assertIn("stopped", stop_response.json()["message"])
        mock_service.stop_streaming.assert_called_once()

    @patch("src.api.routes.Request")
    def test_generate_recommendations_endpoint(self, MockRequest):
        mock_recommender = MagicMock()
        mock_recommender.generate_recommendations.return_value = [
            {"product_id": "P0001", "score": 0.9, "reason": "Hybrid: Content + Collaborative"},
            {"product_id": "P0002", "score": 0.8, "reason": "Content-based similarity"}
        ]
        
        mock_request = MagicMock()
        mock_request.app.state.recommender = mock_recommender
        MockRequest.return_value = mock_request
        
        with patch("src.api.routes.get_user_activity") as mock_get_activity:
            mock_get_activity.return_value = [{"id": 1}, {"id": 2}]
            
            response = self.client.post(f"/generate-recommendations/{self.user_id}")
            
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["user_id"], self.user_id)
            self.assertEqual(response.json()["recommendations_generated"], 2)
            self.assertIn("message", response.json())
            
            mock_recommender.generate_recommendations.assert_called_once_with(self.user_id)


if __name__ == '__main__':
    unittest.main()