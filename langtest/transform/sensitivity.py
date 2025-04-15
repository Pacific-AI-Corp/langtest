import asyncio
import random
from abc import ABC, abstractmethod
from collections import defaultdict

from langtest.transform.utils import filter_unique_samples
from langtest.errors import Errors, Warnings
from typing import List, Optional, Dict, TypedDict
from langtest.modelhandler import ModelAPI
from langtest.transform.base import ITests, TestFactory

from langtest.utils.custom_types import Sample
from langtest.logger import logger as logging


class SensitivityTestFactory(ITests):
    """A class for performing Sensitivity tests on a given dataset.

    This class provides functionality to perform sensitivity tests on a given dataset
    using various test configurations.

    Attributes:
        alias_name (str): A string representing the alias name for this test factory.

    """

    alias_name = "sensitivity"

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initialize a new SensitivityTestFactory instance.

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
        """Run the sensitivity test and return the resulting `Sample` objects.

        Returns:
            List[Sample]: A list of `Sample` objects representing the resulting dataset after running the sensitivity test.

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

            _ = [
                (
                    sample.transform(
                        test_func,
                        params.get("parameters", {}),
                    )
                    if hasattr(sample, "transform")
                    else sample
                )
                for sample in data_handler_copy
            ]
            transformed_samples = data_handler_copy

            new_transformed_samples, removed_samples_tests = filter_unique_samples(
                TestFactory.task.category, transformed_samples, test_name
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
    def available_tests() -> dict:
        """
        Get a dictionary of all available tests, with their names as keys and their corresponding classes as values.

        Returns:
            dict: A dictionary of test names and classes.
        """

        return BaseSensitivity.test_types


class BaseSensitivity(ABC):
    """Abstract base class for implementing sensitivity measures.

    Attributes:
        alias_name (str): A name or list of names that identify the sensitivity measure.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented sensitivity measure.
    """

    test_types = defaultdict(lambda: BaseSensitivity)

    alias_name = None
    supported_tasks = [
        "sensitivity",
        "question-answering",
    ]

    # TestConfig
    TestConfig = TypedDict(
        "TestConfig",
        min_pass_rate=float,
    )

    @staticmethod
    @abstractmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """Abstract method that implements the sensitivity measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.

        Returns:
            Any: The transformed data based on the implemented sensitivity measure.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs) -> List[Sample]:
        """Abstract method that implements the sensitivity measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the sensitivity measure.

        Returns:
            List[Sample]: The transformed data based on the implemented sensitivity measure.

        """
        progress = kwargs.get("progress_bar", False)
        for sample in sample_list:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
                else:
                    sample.expected_results = model(sample.original)
                    sample.actual_results = model(sample.test_case)
                    sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list

    @classmethod
    async def async_run(cls, sample_list: List[Sample], model: ModelAPI, **kwargs):
        """Creates a task to run the sensitivity measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the sensitivity measure.

        Returns:
            asyncio.Task: The task that runs the sensitivity measure.

        """
        created_task = asyncio.create_task(cls.run(sample_list, model, **kwargs))
        return created_task

    def __init_subclass__(cls) -> None:
        """Register the sub-classes of the BaseSensitivity class"""
        alias = cls.alias_name if isinstance(cls.alias_name, list) else [cls.alias_name]
        for name in alias:
            cls.test_types[name] = cls


class AddNegation(BaseSensitivity):
    """A class for negating sensitivity-related phrases in the input text.

    This class identifies common sensitivity-related phrases such as 'is', 'was', 'are', and 'were' in the input text
    and replaces them with their negations to make the text less sensitive.

    Attributes:
        alias_name (str): The alias name for this sensitivity transformation.

    Methods:
        transform(sample_list: List[Sample]) -> List[Sample]: Applies the sensitivity negation transformation to a list
            of samples.
    """

    alias_name = "add_negation"

    @staticmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        def apply_transformation(perturbed_text):
            """
            Filters the dataset for 'is', 'was', 'are', or 'were' and negates them.

            Args:
                perturbed_text (str): The input text to be transformed.

            Returns:
                str: The transformed text with sensitivity-related phrases negated.
            """
            transformations = {
                " is ": " is not ",
                " was ": " was not ",
                " are ": " are not ",
                " were ": " were not ",
            }

            for keyword, replacement in transformations.items():
                if keyword in perturbed_text and replacement not in perturbed_text:
                    perturbed_text = perturbed_text.replace(keyword, replacement, 1)

            return perturbed_text

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx] = apply_transformation(sample)

        return sample_list


class AddToxicWords(BaseSensitivity):
    """A class for handling sensitivity-related phrases in the input text, specifically related to toxicity.

    Attributes:
        alias_name (str): The alias name for this sensitivity transformation.

    Methods:
        transform(
            sample_list: List[Sample],
            starting_context: Optional[List[str]] = None,
            ending_context: Optional[List[str]] = None,
            strategy: str = None,
        ) -> List[Sample]: Transform the input list of samples to add toxicity-related text.

    Raises:
        ValueError: If an invalid context strategy is provided.
    """

    alias_name = "add_toxic_words"

    @staticmethod
    def transform(
        sample_list: List[Sample],
        starting_context: Optional[List[str]] = None,
        ending_context: Optional[List[str]] = None,
        strategy: str = None,
    ) -> List[Sample]:
        """
        Transform the input list of samples to add toxicity-related text.

        Args:
            sample_list (List[Sample]): A list of samples to transform.
            starting_context (Optional[List[str]]): A list of starting context tokens.
            ending_context (Optional[List[str]]): A list of ending context tokens.
            strategy (str): The strategy for adding context. Can be 'start', 'end', or 'combined'.

        Returns:
            List[Sample]: The transformed list of samples.

        Raises:
            ValueError: If an invalid context strategy is provided.
        """

        def context(text, strategy):
            possible_methods = ["start", "end", "combined"]
            if strategy is None:
                strategy = random.choice(possible_methods)
            elif strategy not in possible_methods:
                raise ValueError(Errors.E066(strategy=strategy))

            if strategy == "start" or strategy == "combined":
                add_tokens = random.choice(starting_context)
                add_string = (
                    " ".join(add_tokens) if isinstance(add_tokens, list) else add_tokens
                )
                if text != "-":
                    text = add_string + " " + text

            if strategy == "end" or strategy == "combined":
                add_tokens = random.choice(ending_context)
                add_string = (
                    " ".join(add_tokens) if isinstance(add_tokens, list) else add_tokens
                )

                if text != "-":
                    text = text + " " + add_string

            return text

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx] = context(sample, strategy)

        return sample_list
