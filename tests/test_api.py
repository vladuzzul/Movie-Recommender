import unittest
import requests
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.api import app

client = TestClient(app)

class Test_api(unittest.TestCase):
    def test_health(self):
        response = client.get(
            "http://127.0.0.1:8000/api/health"
        )
        self.assertEqual(response.status_code, 200)

    def test_movie(self):
        response = client.get(
            "http://127.0.0.1:8000/api/movies?limit=1"
        )
        json_response = response.json()
        print(json_response)
        self.assertIsInstance(json_response[0]["movieId"], int)
        self.assertIsInstance(json_response[0]["title"], str)
        self.assertIsInstance(json_response[0]["genres"], str)
        self.assertIsInstance(json_response[0]["year"], int)
        self.assertIsInstance(json_response[0]["rating"], float)

    def test_recommendations(self):
        data = {
            "ratings": [
                {
                    "title": "Won't You Be My Neighbor?",
                    "rating": 5,
                    "movieId": 187717
                }
            ]
        }
        response = client.post(
            "http://127.0.0.1:8000/api/recommendations",
            json=data
        )
        json_response = response.json()
        self.assertIsInstance(json_response[0]["movieId"], int)
        self.assertIsInstance(json_response[0]["title"], str)
        self.assertIsInstance(json_response[0]["genres"], str)
        self.assertIsInstance(json_response[0]["year"], int)
        self.assertIsInstance(json_response[0]["rating"], float)

if __name__ == "__main__":
    unittest.main()
            
