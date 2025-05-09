import asyncio
from typing import List, Dict, Optional, TypedDict
from langtest.utils.custom_types.sample import Sample
from abc import ABC, abstractmethod
from langtest.errors import Errors, Warnings
from langtest.transform.base import ITests, TestFactory
from langtest.modelhandler.modelhandler import ModelAPI
from ..utils.lib_manager import try_import_lib
from langtest.transform.utils import filter_unique_samples
from langtest.logger import logger as logging


class GrammarTestFactory(ITests):
    alias_name = "grammar"
    supported_tasks = ["text-classification", "question-answering"]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initialize a new SycophancyTestFactory instance.

        Args:
            data_handler (List[Sample]): A list of `Sample` objects representing the input dataset.
            tests (Optional[Dict]): A dictionary of test names and corresponding parameters (default is None).
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the `tests` argument is not a dictionary.

        """
        self.supported_tests = self.available_tests()
        self._data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

        if not isinstance(self.tests, dict):
            raise ValueError(Errors.E048())

        if len(self.tests) == 0:
            self.tests = self.supported_tests

        not_supported_tests = set(self.tests) - set(self.supported_tests)
        if len(not_supported_tests) > 0:
            raise ValueError(
                Errors.E049(
                    not_supported_tests=not_supported_tests,
                    supported_tests=list(self.supported_tests.keys()),
                )
            )

    def transform(self) -> List[Sample]:
        """Execute the Sycophancy test and return resulting `Sample` objects.

        Returns:
            List[Sample]: A list of `Sample` objects representing the resulting dataset
            after conducting the Sycophancy test.

        """
        all_samples = []
        no_transformation_applied_tests = {}
        tests_copy = self.tests.copy()
        for test_name, params in tests_copy.items():
            if TestFactory.is_augment:
                data_handler_copy = [x.model_copy() for x in self._data_handler]
            else:
                data_handler_copy = [x.model_copy() for x in self._data_handler]

            test_func = self.supported_tests[test_name].transform

            if str(TestFactory.task) in ("question-answering"):
                _ = [
                    (
                        sample.transform(
                            test_func,
                            params.get("parameters", {}),
                            prob=params.pop("prob", 1.0),
                        )
                        if hasattr(sample, "transform")
                        else sample
                    )
                    for sample in data_handler_copy
                ]
                transformed_samples = data_handler_copy

            else:
                transformed_samples = test_func(
                    data_handler_copy,
                    **params.get("parameters", {}),
                    prob=params.pop("prob", 1.0),
                )
            new_transformed_samples, removed_samples_tests = filter_unique_samples(
                TestFactory.task, transformed_samples, test_name
            )
            all_samples.extend(new_transformed_samples)

            no_transformation_applied_tests.update(removed_samples_tests)

        if no_transformation_applied_tests:
            warning_message = Warnings._W009
            for test, count in no_transformation_applied_tests.items():
                warning_message += Warnings._W010.format(
                    test=test, count=count, total_sample=len(self._data_handler)
                )

            logging.warning(warning_message)

        return all_samples

    @staticmethod
    def available_tests() -> Dict[str, type["BaseGrammar"]]:
        """
        Retrieve a dictionary of all available tests, with their names as keys
        and their corresponding classes as values.

        Returns:
            dict: A dictionary of test names and classes.
        """
        tests = {
            j: i
            for i in BaseGrammar.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class BaseGrammar(ABC):
    """
    BaseGrammar abstract class for implementing to test the model performance on varying the input by grammatically changes .

    """

    # TestConfig
    TestConfig = TypedDict(
        "TestConfig",
        min_pass_rate=float,
    )

    @staticmethod
    @abstractmethod
    def transform(sample_list: List[Sample], *args, **kwargs):
        """
        Abstract method that transforms the sample data with grammatically changes.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs):
        """Abstract method that implements the model testing for to check the grammer issues."""

        progress_bar = kwargs.get("progress_bar", False)
        for sample in sample_list:
            if hasattr(sample, "run"):
                sample.run(model, **kwargs)
            else:
                sample.expected_results = model.predict(sample.original)
                sample.actual_results = model.predict(sample.test_case)
            if progress_bar:
                progress_bar.update(1)
            sample.state = "done"

        return sample_list

    @classmethod
    async def async_run(cls, sample_list: list, model: ModelAPI, **kwargs):
        created_task = asyncio.create_task(cls.run(sample_list, model, **kwargs))
        return await created_task


class Paraphrase(BaseGrammar):
    alias_name = "paraphrase"
    supported_tasks = ["text-classification", "text-generation", "question-answering"]

    @staticmethod
    def transform(
        sample_list: List[Sample], prob: Optional[float] = 1.0, *args, **kwargs
    ):
        if try_import_lib("transformers"):
            from transformers import pipeline

            pipe = pipeline(
                "text2text-generation", model="humarin/chatgpt_paraphraser_on_T5_base"
            )
            for idx, sample in enumerate(sample_list):
                if isinstance(sample, str):
                    test_case = pipe(sample, max_length=11000, num_return_sequences=1)[0][
                        "generated_text"
                    ]
                    sample_list[idx] = test_case
                else:
                    test_case = pipe(
                        sample.original, max_length=1000, num_return_sequences=1
                    )[0]["generated_text"]
                    sample.test_case = test_case
                    sample.category = "grammar"

            return sample_list

        else:
            raise ModuleNotFoundError(Errors.E023(LIB_NAME="transformers"))
