import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rescue_pipeline import RESCuePipeline

class TestRESCuePipeline(unittest.TestCase):
    @patch('src.rescue_pipeline.Planner')
    @patch('src.rescue_pipeline.Executor')
    @patch('src.rescue_pipeline.Verifier')
    def test_pipeline_flow(self, MockVerifier, MockExecutor, MockPlanner):
        # Setup Mocks
        planner = MockPlanner.return_value
        executor = MockExecutor.return_value
        verifier = MockVerifier.return_value
        
        # Planner returns 2 hypotheses
        planner.generate_hypotheses.return_value = [
            {'box': [0,0,10,10], 'reasoning': 'r1', 'noun_phrase': 'n1'},
            {'box': [20,20,30,30], 'reasoning': 'r2', 'noun_phrase': 'n2'}
        ]
        
        # Executor returns 3 masks for each call
        mask = np.zeros((100, 100), dtype=bool)
        executor.execute.return_value = [mask, mask, mask]
        
        # Verifier returns scores
        # We expect 2 hypotheses * 3 masks = 6 calls
        # Let's make the last one the best
        verifier.verify.side_effect = [10.0, 20.0, 30.0, 40.0, 50.0, 99.0]
        
        # Init pipeline
        pipeline = RESCuePipeline(device="cpu")
        
        # Create dummy image
        img_path = "dummy.jpg"
        Image.new('RGB', (100, 100)).save(img_path)
        
        try:
            # Run
            result = pipeline.run(img_path, "query", N=2)
            
            # Verify Planner called once
            planner.generate_hypotheses.assert_called_once()
            
            # Verify Executor called twice (once per hypoth)
            self.assertEqual(executor.execute.call_count, 2)
            
            # Verify Verifier called 6 times
            self.assertEqual(verifier.verify.call_count, 6)
            
            # Verify Best Score selected
            self.assertEqual(result['best_score'], 99.0)
            
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)

if __name__ == '__main__':
    unittest.main()
