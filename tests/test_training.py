import unittest

from co_retrieval.chunking import CodeChunk
from co_retrieval.training import CoTrainingConfig, CoTrainingTrainer, TrainingSample


class TrainingTests(unittest.TestCase):
    def test_co_training_updates_model_components(self):
        chunks = [
            CodeChunk(
                file_path="services.py",
                start_line=1,
                end_line=3,
                chunk_type="method",
                text="class UserService:\n    def fetch_user(self, user_id):\n        return user_id",
                defined_symbols=["fetch_user"],
                used_symbols=["UserService", "user_id"],
                call_names=["fetch_user"],
                parent_class="UserService",
            )
        ]
        samples = [
            TrainingSample(
                left_context="result = UserService().fetch_",
                target="fetch_user(user_id)",
            )
        ]
        trainer = CoTrainingTrainer(
            chunks,
            CoTrainingConfig(epochs=2, top_k=2, sampled_contexts=2),
        )

        history = trainer.train(samples)

        self.assertEqual(len(history), 2)
        self.assertGreaterEqual(trainer.model.soft_prompt.update_count, 2)
        self.assertIn("fetch_user", trainer.retriever.weights)


if __name__ == "__main__":
    unittest.main()
