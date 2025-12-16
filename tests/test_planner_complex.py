import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Adjust path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock vLLM before importing planner
sys.modules["vllm"] = MagicMock()
from src.planner import Planner, PlannerConfig, Hypothesis

class TestComplexPlanner(unittest.TestCase):
    def setUp(self):
        self.config = PlannerConfig(
            model_path="mock/path",
            use_stratified_sampling=True,
            diversity_factor=1.0, # Test 1:1 first
            iou_threshold=0.5
        )
        self.planner = Planner(self.config)
        
        # Mock the LLM
        self.planner.llm = MagicMock()
        
    def test_stratified_sampling_counts(self):
        """Verify that N=5 gets split into correct strategies."""
        # We need to mock _sample_batch to just return dummy hypotheses
        self.planner._sample_batch = MagicMock(return_value=[])
        
        self.planner.generate_hypotheses("dummy.jpg", "query", N=5)
        
        # Strategies: Conservative, Exploratory, Spatial
        # N=5. Cons=ceil(2) -> 2. Expl=ceil(2) -> 2. Spatial=1.
        
        calls = self.planner._sample_batch.call_args_list
        self.assertEqual(len(calls), 3)
        
        # Check strategy names in calls
        strategies = [c.kwargs['strategy'] for c in calls]
        self.assertIn("conservative", strategies)
        self.assertIn("exploratory", strategies)
        self.assertIn("spatial", strategies)
        
    def test_deduplication(self):
        """Test the select_diverse_subset logic."""
        # Create 3 hypotheses. A and B overlap heavily. C is distinct.
        h1 = Hypothesis([0, 0, 100, 100], "reason1", "obj", 0.9, "strat1", "raw")
        h2 = Hypothesis([5, 5, 105, 105], "reason2", "obj", 0.8, "strat2", "raw") # High IOU with h1
        h3 = Hypothesis([200, 200, 300, 300], "reason3", "obj", 0.7, "strat3", "raw") # No overlap
        
        candidates = [h1, h2, h3]
        
        # Request N=2. Should pick h1 (higher score) and h3 (diverse). h2 should be skipped due to IoU.
        selected = self.planner._select_diverse_subset(candidates, N=2, query="test")
        
        self.assertEqual(len(selected), 2)
        self.assertIn(h1, selected)
        self.assertIn(h3, selected)
        self.assertNotIn(h2, selected)

    def test_parsing(self):
        """Test robust parsing of LLM output."""
        text = """
        Reasoning: It is a cat.
        Target Concept: cat
        Box: [100, 100, 200, 200]
        """
        # Mock image size 1000x1000 for simplicity (so 100->100)
        h = self.planner._parse_completion(text, 1000, 1000, 0.9, "test", 1000000)
        
        self.assertIsNotNone(h)
        self.assertEqual(h.box, [100, 100, 200, 200])
        self.assertEqual(h.target_concept, "cat")

if __name__ == '__main__':
    unittest.main()
