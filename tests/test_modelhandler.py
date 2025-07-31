import unittest
import langtest
from langtest.modelhandler import ModelAPI
from langtest.modelhandler.spacy_modelhandler import PretrainedModelForNER as SpacyNER
from langtest.modelhandler.llm_modelhandler import ConfigError
from langtest.tasks import TaskManager


class ModelAPITestCase(unittest.TestCase):
    """
    Test case for the ModelHandler class.

    Note: we mainly want to check hubs access and loading mechanism
    """

    def setUp(self):
        pass

    def test_unsupported_task(self) -> None:
        """
        Test loading an unsupported task.
        """
        with self.assertRaises(AssertionError) as _:
            TaskManager("unsupported_task")

    def test_unsupported_hub(self) -> None:
        """
        Test loading an unsupported hub.
        """
        with self.assertRaises(AssertionError) as _:
            task = TaskManager("ner")
            task.model(model_path="en_core_web_sm", model_hub="unsupported_hub")

    def test_ner_transformers_model(self) -> None:
        """
        Test loading an NER model from Hugging Face Transformers.
        """
        task = TaskManager("ner")
        model = task.model(model_path="dslim/bert-base-NER", model_hub="huggingface")
        self.assertIsInstance(model, ModelAPI)
        self.assertIsInstance(
            model,
            langtest.modelhandler.transformers_modelhandler.PretrainedModelForNER,
        )

    def test_ner_spacy_model(self) -> None:
        """
        Test loading an NER model from spaCy.
        """
        task = TaskManager("ner")

        model = task.model(model_path="en_core_web_sm", model_hub="spacy")
        self.assertIsInstance(model, ModelAPI)
        self.assertIsInstance(
            model,
            SpacyNER,
        )

    def test_ai21_model(self) -> None:
        """
        Test loading a model from the Ai21 hub
        """
        with self.assertRaises(Exception) as _:
            task = TaskManager("question-answering")
            task.model(model_path="j2-jumbo-instruct", model_hub="ai21")


    def test_cohere_model(self) -> None:
        """
        Test loading a model from Cohere
        """
        with self.assertRaises(Exception) as _:
            task = TaskManager("question-answering")
            task.model(model_path="command-xlarge-nightly", model_hub="cohere")

    def test_generic_api_model(self) -> None:
        """
        Test loading a model from a generic API
        """

        # check the web hub is available
        from langtest.modelhandler import ModelAPI

        AssertionError("web" in ModelAPI.model_registry.keys())

        # check the harness is loading correctly
        from langtest import Harness

        # with self.assertRaises(AssertionError) as _:

        # endpoint to the model
        url = "https://generic-api.com/completion"

        # lambda functions to process the input and output
        input_data = lambda content: {
            "contents": [{"role": "user", "parts": [{"text": content}]}]
        }

        output_praser = lambda response: response["candidates"][0]["content"]["parts"][0][
            "text"
        ]

        # create the harness
        harness = Harness(
            task="question-answering",
            model={
                "model": {
                    "url": url,
                    "headers": {
                        "Content-Type": "application/json",
                    },
                    "input_processor": input_data,
                    "output_parser": output_praser,
                },
                "hub": "web",
            },
            data={
                "data_source": "OpenBookQA",
                "split": "test-tiny",
            },
        )

        # slice the dataset
        harness.data = harness.data[:10]

        # generate a testcase
        harness.generate()
