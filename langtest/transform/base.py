from typing import List, Dict, Type, Union
from abc import ABC, abstractmethod
from tqdm.auto import tqdm
import asyncio
import nest_asyncio
from langtest.modelhandler.modelhandler import ModelAPI
from langtest.errors import Errors
from langtest.tasks.task import TaskManager
from .custom_data import add_custom_data
from ..utils.custom_types.sample import (
    Sample,
    Result,
)

nest_asyncio.apply()


class TestFactory:
    """
    A factory class for creating and running different types of tests on data.
    """

    is_augment = False
    task: TaskManager = None

    # Additional operations can be performed here using the validated data

    @staticmethod
    def call_add_custom_bias(data: Union[list, dict], name: str, append: bool) -> None:
        """
        Add custom bias to the given data.

        Args:
            data (Union[list, dict]): The data to which the custom bias will be added.
            name (str): The name of the custom bias.
            append (bool): Indicates whether to append the custom bias or replace the existing bias.

        Returns:
            None
        """
        add_custom_data(data, name, append)

    @staticmethod
    def transform(
        task: str, data: List[Sample], test_types: dict, *args, **kwargs
    ) -> List[Result]:
        """
        Runs the specified tests on the given data and returns a list of results.

        Args:
            data : List[Sample]
                The data to be tested.
            test_types : dict
                A dictionary mapping test category names to lists of test scenario names.
            model: ModelAPI
                Model to be tested.

        Returns:
            List[Results]
                A list of results from running the specified tests on the data.
        """
        all_results = []
        all_categories = TestFactory.test_categories()
        test_names = list(test_types.keys())
        TestFactory.task = task if isinstance(task, TaskManager) else TaskManager(task)

        if "defaults" in test_names:
            test_names.pop(test_names.index("defaults"))
        tests = tqdm(
            test_names, desc="Generating testcases...", disable=TestFactory.is_augment
        )
        m_data = kwargs.get("m_data", None)

        # Check test-task supportance
        for test_category in tests:
            if test_category in all_categories.keys():
                sub_test_types = test_types[test_category]
                for sub_test in sub_test_types:
                    supported = (
                        all_categories[test_category]
                        .available_tests()[sub_test]
                        .supported_tasks
                    )
                    if task not in supported:
                        raise ValueError(
                            Errors.E046(sub_test=sub_test, task=task, supported=supported)
                        )
            elif test_category != "defaults":
                raise ValueError(
                    Errors.E047(
                        test_category=test_category,
                        available_categories=list(all_categories.keys()),
                    )
                )

        # Generate testcases
        for each in tests:
            tests.set_description(f"Generating testcases... ({each})")
            if each in all_categories:
                sub_test_types = test_types[each]
                sample_results = (
                    all_categories[each](
                        m_data, sub_test_types, raw_data=data
                    ).transform()
                    if each in ["robustness", "bias"] and m_data
                    else all_categories[each](data, sub_test_types).transform()
                )
                all_results.extend(sample_results)
        tests.close()
        return all_results

    @classmethod
    def test_categories(cls) -> Dict[str, Type["ITests"]]:
        """
        Returns a dictionary mapping test category names to the corresponding test classes.

        Returns:
            Dict:
                A dictionary mapping test category names to the corresponding test classes.
        """
        return {cls.alias_name.lower(): cls for cls in ITests.__subclasses__()}

    @classmethod
    def test_scenarios(cls):
        """
        Returns a dictionary mapping test class names to the available test scenarios for each class.

        Returns:
            Dict:
                A dictionary mapping test class names to the available test scenarios for each class.
        """
        return {
            cls.alias_name.lower(): cls.available_tests()
            for cls in ITests.__subclasses__()
        }

    @staticmethod
    def run(samples_list: List[Sample], model_handler: ModelAPI, **kwargs):
        """
        Runs the specified tests on the given data and returns a list of results.

        Args:
            samples_list : List[Sample]
            model_handler : ModelAPI

        """
        async_tests = TestFactory.async_run(samples_list, model_handler, **kwargs)
        temp_res = asyncio.run(async_tests)
        results = []
        for each in temp_res:
            try:
                if hasattr(each, "_result"):
                    results.extend(each._result)
                elif isinstance(each, list):
                    for i in each:
                        if hasattr(i, "_result"):
                            results.extend(i._result)
                        else:
                            results.append(i)
            except TypeError:
                if hasattr(each, "exception"):
                    raise each.exception()
                raise ValueError(f"Unknown error occurred {each}")

        return results

    @classmethod
    async def async_run(
        cls, samples_list: List[Sample], model_handler: ModelAPI, **kwargs
    ):
        """
        Runs the specified tests on the given data and returns a list of results.

        Args:
            samples_list : List[Sample]
            model_handler : ModelAPI

        """
        hash_samples = {}
        for sample in samples_list:
            if sample.category not in hash_samples:
                hash_samples[sample.category] = {}
            if sample.test_type not in hash_samples[sample.category]:
                hash_samples[sample.category][sample.test_type] = [sample]
            else:
                hash_samples[sample.category][sample.test_type].append(sample)

        # Add hash_samples in kwargs
        kwargs["test_cases"] = hash_samples

        all_categories = TestFactory.test_categories()
        tests = tqdm(
            total=len(samples_list),
            desc="Running testcases... ",
            position=0,
            disable=TestFactory.is_augment,
        )
        all_results = []
        for each in hash_samples:
            values = hash_samples[each]
            category_output = all_categories[each].run(
                values, model_handler, progress_bar=tests, **kwargs
            )
            if type(category_output) == list:
                all_results.extend(category_output)
            else:
                all_results.append(category_output)
        run_results = await asyncio.gather(*all_results)

        return run_results


class ITests(ABC):
    """
    An abstract base class for defining different types of tests.
    """

    alias_name: Union[None, List[str], str] = None
    supported_tasks: List[str] = []

    @abstractmethod
    def transform(self):
        """
        Runs the test and returns the results.

        Returns:
            List[Results]:
                A list of results from running the test.
        """
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def available_tests() -> Dict[str, Type["ITests"]]:
        """
        Returns a list of available test scenarios for the test class.

        Returns:
            List[str]:
                A list of available test scenarios for the test class.
        """
        return NotImplementedError

    @classmethod
    def run(
        cls, sample_list: Dict[str, List[Sample]], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        """
        Runs the specified tests on the given data and returns a list of results.

        Args:
            sample_list (Dict[str, List[Sample]]):
                A dictionary mapping test scenario names to a list of `Sample` objects.
            model (ModelAPI):
                A `ModelAPI` object representing the model to be tested.

        Returns:
            List[Sample]: A list of `Sample` objects with the test results.

        """
        supported_tests = cls.available_tests()
        tasks = []
        for test_name, samples in sample_list.items():
            if len(test_name.split("-")) > 1:
                test_name = "multiple_perturbations"
            test_output = supported_tests[test_name].async_run(samples, model, **kwargs)
            if type(test_output) == list:
                tasks.extend(test_output)
            else:
                tasks.append(test_output)

        return tasks
