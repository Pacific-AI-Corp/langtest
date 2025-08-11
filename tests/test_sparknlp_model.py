import unittest

from langtest import Harness
from langtest.modelhandler import ModelAPI


class SparkNLPTestCase(unittest.TestCase):
    """
    Test case for the SparkNLP integration in the langtest module.
    """

    def setUp(self) -> None:
        self.params = {
            "task": "ner",
            "model": {"model": "ner_dl_bert", "hub": "johnsnowlabs"},
            "data": {"data_source": "tests/fixtures/test.conll"},
            "config": "tests/fixtures/config_ner.yaml",
        }

    def test_predict(self):
        """
        Testing Instance after in Harness Class
        """
        harness = Harness(**self.params)
        self.assertIsInstance(harness, Harness)

    def test_outputCol(self):
        """
        Testing Attributes of Harness Class
        """
        harness = Harness(**self.params)
        self.assertIsInstance(harness.model, (str, ModelAPI))
        self.assertIsInstance(harness.model.output_col, str)
