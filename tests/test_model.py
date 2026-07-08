import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.nlp_processor import load_dataframes, build_recommender, recommendations

class TestModel(unittest.TestCase):
    def test_model(self):
        movies_df, ratings_df = load_dataframes()
        titles, indices, cosine_sim = build_recommender(movies_df)
        recommendation = recommendations(
            movies_df,
            ratings_df,
            titles,
            indices,
            cosine_sim, 
            "Fast & Furious Presents: Hobbs & Shaw",
            None
        )
        self.assertEqual(str(type(recommendation)), "<class 'pandas.core.frame.DataFrame'>")

if __name__ == "__main__":
    unittest.main()
            