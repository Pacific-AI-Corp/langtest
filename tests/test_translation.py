import unittest
from langtest import Harness


class TranslationTestCase(unittest.TestCase):
    """
    Test case for translation functionality using the langtest harness.
    """

    def setUp(self) -> None:
        """
        Set up the test case by initializing the langtest Harness and configuring it.
        """
        self.harness = Harness(
            task="translation",
            model={"model": "t5-base", "hub": "huggingface"},
            data={"data_source": "Translation", "split": "test"},
        )

        # configure the harness
        self.harness.configure(
            {
                "model_parameters": {"target_language": "German"},
                "tests": {
                    "defaults": {"min_pass_rate": 1.0},
                    "robustness": {
                        "add_typo": {"min_pass_rate": 0.7},
                        "lowercase": {"min_pass_rate": 0.7},
                    },
                },
            }
        )
        self.harness.data = self.harness.data[:5]

    def test_translation_workflow(self):
        """
        Test the translation workflow by generating translations, running tests, and generating a report.
        """
        self.harness.generate().run().report()
