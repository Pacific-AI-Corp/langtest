import asyncio
from collections import defaultdict
from ..errors import Errors
from abc import ABC, abstractmethod
from typing import List, Dict, TypedDict, Union

from langtest.modelhandler.modelhandler import ModelAPI
from langtest.transform.base import ITests
from langtest.utils.custom_types import (
    MinScoreOutput,
    MinScoreQASample,
    MinScoreSample,
    Sample,
)
from langtest.utils.custom_types.output import NEROutput, SequenceClassificationOutput
from langtest.utils.gender_classifier import GenderClassifier
from .utils import RepresentationOperation
from .constants import (
    default_ehtnicity_representation,
    default_economic_country_representation,
    default_religion_representation,
)


class RepresentationTestFactory(ITests):
    """
    A class for performing representation tests on a given dataset.
    """

    alias_name = "representation"

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
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
        """
        Runs the representation test and returns the resulting `Sample` objects.

        Returns:
            List[Sample]:
                A list of `Sample` objects representing the resulting dataset after running the representation test.
        """
        all_samples = []

        for test_name, params in self.tests.items():
            data_handler_copy = [x.model_copy() for x in self._data_handler]

            transformed_samples = self.supported_tests[test_name].transform(
                test_name, data_handler_copy, params
            )

            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)

        return all_samples

    @staticmethod
    def available_tests() -> Dict[str, type["BaseRepresentation"]]:
        """
        Get a dictionary of all available tests, with their names as keys and their corresponding classes as values.

        Returns:
            Dict: A dictionary of test names and classes.
        """

        return BaseRepresentation.test_types


class BaseRepresentation(ABC):
    """Abstract base class for implementing representation measures.

    Attributes:
        alias_name (str): A name or list of names that identify the representation measure.
        supported_tasks (List[str]): name of the supported task for the representation measure
    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output
        based on the implemented representation measure.
    """

    test_types = defaultdict(lambda: BaseRepresentation)
    alias_name = None
    supported_tasks = [
        "ner",
        "text-classification",
        "question-answering",
        "summarization",
        "toxicity",
        "translation",
    ]

    # Config Hint for the representation tests
    TestConfig = TypedDict(
        "TestConfig",
        min_count=Union[int, Dict[str, int]],
        min_proportion=Union[float, Dict[str, float]],
    )

    @classmethod
    @abstractmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> Union[List[MinScoreQASample], List[MinScoreSample]]:
        """Abstract method that implements the representation measure.

        Args:
            test (str): name of the test to perform
            data (List[Sample]): The input data to be transformed.
            params (Dict): parameters for tests configuration
        Returns:
            Any: The transformed data based on the implemented representation measure.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    async def run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        """Computes the score for the given data.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for the computation.

        Returns:
            List[Sample]: The transformed samples.
        """
        raise NotImplementedError()

    @classmethod
    async def async_run(cls, sample_list: List[Sample], model: ModelAPI, **kwargs):
        """Creates a task for the run method.

        Args:
            sample_list (List[Sample]): The input data to be evaluated for representation test.
            model (ModelAPI): The model to be used for the computation.

        Returns:
            asyncio.Task: The task for the run method.
        """
        created_task = asyncio.create_task(cls.run(sample_list, model, **kwargs))
        return created_task

    def __init_subclass__(cls) -> None:
        """Registers the subclass in the model_registry dictionary."""
        alias = cls.alias_name if isinstance(cls.alias_name, list) else [cls.alias_name]
        for name in alias:
            BaseRepresentation.test_types[name] = cls


class GenderRepresentation(BaseRepresentation):
    """Subclass of BaseRepresentation that implements the gender representation test.

    Attributes:
        alias_name (List[str]): The list of test names that identify the representation measure.
        supported_tasks (List[str]): name of the supported task for the representation measure
    """

    alias_name = [
        "min_gender_representation_count",
        "min_gender_representation_proportion",
    ]

    min_count = TypedDict("min_count", male=int, female=int, unknown=int)
    min_proportion = TypedDict("min_proportion", male=float, female=float, unknown=float)

    # Config Hint for the representation tests
    TestConfig = TypedDict(
        "TestConfig",
        min_count=Union[min_count, int],
        min_proportion=Union[min_proportion, float],
    )

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> Union[List[MinScoreQASample], List[MinScoreSample]]:
        """Compute the gender representation measure

        Args:
            test (str): name of the test
            data (List[Sample]): The input data to be evaluated for representation test.
            params : parameters specified in config.

        Raises:
            ValueError: If sum of specified proportions in config is greater than 1

        Returns:
            Union[List[MinScoreQASample], List[MinScoreSample]]: Gender Representation test results.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        samples = []
        if test == "min_gender_representation_count":
            if isinstance(params["min_count"], dict):
                min_counts = params["min_count"]
            else:
                min_counts = {
                    "male": params["min_count"],
                    "female": params["min_count"],
                    "unknown": params["min_count"],
                }

            for key, value in min_counts.items():
                if hasattr(data[0], "task") and data[0].task == "question-answering":
                    sample = MinScoreQASample(
                        original_question=key,
                        original_context="-",
                        options="-",
                        perturbed_question="-",
                        perturbed_context="-",
                        category="representation",
                        test_type="min_gender_representation_count",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value),
                    )
                    samples.append(sample)

                else:
                    sample = MinScoreSample(
                        original="-",
                        category="representation",
                        test_type="min_gender_representation_count",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value),
                    )
                    samples.append(sample)
        else:
            min_proportions = {"male": 0.26, "female": 0.26, "unknown": 0.26}

            if isinstance(params["min_proportion"], dict):
                min_proportions = params["min_proportion"]
                if sum(min_proportions.values()) > 1:
                    raise ValueError(
                        Errors.E064(var="min_gender_representation_proportion")
                    )

            for key, value in min_proportions.items():
                if hasattr(data[0], "task") and data[0].task == "question-answering":
                    sample = MinScoreQASample(
                        original_question=key,
                        original_context="-",
                        options="-",
                        perturbed_question="-",
                        perturbed_context="-",
                        category="representation",
                        test_type="min_gender_representation_proportion",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value),
                    )
                    samples.append(sample)

                else:
                    sample = MinScoreSample(
                        original="-",
                        category="representation",
                        test_type="min_gender_representation_proportion",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value),
                    )
                    samples.append(sample)
        return samples

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs) -> List[Sample]:
        """Computes the actual results for the Gender Representation test.

        Args:
            sample_list (List[Sample]): The input data to be evaluated for representation test.
            model (ModelAPI): The model factory object.

        Returns:
            List[Sample]: The list of samples with actual results.

        """

        progress = kwargs.get("progress_bar", False)
        classifier = GenderClassifier()
        for sample in kwargs["raw_data"]:
            if sample.task == "question-answering":
                if "perturbed_context" in sample.__annotations__:
                    genders = [
                        classifier.predict(sample.original_context)
                        for sample in kwargs["raw_data"]
                    ]
                else:
                    genders = [
                        classifier.predict(sample.original_question)
                        for sample in kwargs["raw_data"]
                    ]

            else:
                genders = [
                    classifier.predict(sample.original) for sample in kwargs["raw_data"]
                ]

        gender_counts = {
            "male": len([x for x in genders if x == "male"]),
            "female": len([x for x in genders if x == "female"]),
            "unknown": len([x for x in genders if x == "unknown"]),
        }

        total_samples = len(kwargs["raw_data"])

        for sample in sample_list:
            if progress:
                progress.update(1)

            if sample.test_type == "min_gender_representation_proportion":
                sample.actual_results = MinScoreOutput(
                    min_score=round(gender_counts[sample.test_case] / total_samples, 2)
                )
                sample.state = "done"

            elif sample.test_type == "min_gender_representation_count":
                sample.actual_results = MinScoreOutput(
                    min_score=gender_counts[sample.test_case]
                )
                sample.state = "done"
        return sample_list


class EthnicityRepresentation(BaseRepresentation):
    """Subclass of BaseRepresentation that implements the ethnicity representation test.

    Attributes:
        alias_name (List[str]): The list of test names that identify the representation measure.
        supported_tasks (List[str]): name of the supported task for the representation measure
    """

    alias_name = [
        "min_ethnicity_name_representation_count",
        "min_ethnicity_name_representation_proportion",
    ]

    min_count = TypedDict(
        "min_count",
        black=int,
        asian=int,
        white=int,
        native_american=int,
        hispanic=int,
        inter_racial=int,
    )

    min_proportion = TypedDict(
        "min_proportion",
        black=float,
        asian=float,
        white=float,
        native_american=float,
        hispanic=float,
        inter_racial=float,
    )

    TestConfig = TypedDict(
        "TestConfig",
        min_count=Union[min_count, int],
        min_proportion=Union[min_proportion, float],
    )

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> Union[List[MinScoreQASample], List[MinScoreSample]]:
        """Compute the ethnicity representation measure

        Args:
            test (str): name of the test
            data (List[Sample]): The input data to be evaluated for representation test.
            params : parameters specified in config.

        Raises:
            ValueError: If sum of specified proportions in config is greater than 1

        Returns:
            Union[List[MinScoreQASample], List[MinScoreSample]]: Ethnicity Representation test results.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        sample_list = []
        if test == "min_ethnicity_name_representation_count":
            if not params:
                expected_representation = {
                    "black": 10,
                    "asian": 10,
                    "white": 10,
                    "native_american": 10,
                    "hispanic": 10,
                    "inter_racial": 10,
                }

            else:
                if isinstance(params["min_count"], dict):
                    expected_representation = params["min_count"]

                elif isinstance(params["min_count"], int):
                    expected_representation = {
                        key: params["min_count"]
                        for key in default_ehtnicity_representation
                    }

            for key, value in expected_representation.items():
                if hasattr(data[0], "task") and data[0].task == "question-answering":
                    sample = MinScoreQASample(
                        original_question=key,
                        original_context="-",
                        options="-",
                        perturbed_question="-",
                        perturbed_context="-",
                        category="representation",
                        test_type="min_ethnicity_name_representation_count",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value),
                    )
                    sample_list.append(sample)

                else:
                    sample = MinScoreSample(
                        original="-",
                        category="representation",
                        test_type="min_ethnicity_name_representation_count",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value),
                    )
                    sample_list.append(sample)

        else:
            if not params:
                expected_representation = {
                    "black": 0.13,
                    "asian": 0.13,
                    "white": 0.13,
                    "native_american": 0.13,
                    "hispanic": 0.13,
                    "inter_racial": 0.13,
                }

            else:
                if isinstance(params["min_proportion"], dict):
                    expected_representation = params["min_proportion"]

                    if sum(expected_representation.values()) > 1:
                        raise ValueError(
                            Errors.E064(
                                var="min_ethnicity_name_representation_proportion"
                            )
                        )

                elif isinstance(params["min_proportion"], float):
                    expected_representation = {
                        key: params["min_proportion"]
                        for key in default_ehtnicity_representation
                    }
                    if sum(expected_representation.values()) > 1:
                        raise ValueError(
                            Errors.E064(
                                var="min_ethnicity_name_representation_proportion"
                            )
                        )
            for key, value in expected_representation.items():
                if hasattr(data[0], "task") and data[0].task == "question-answering":
                    sample = MinScoreQASample(
                        original_question=key,
                        original_context="-",
                        options="-",
                        perturbed_question="-",
                        perturbed_context="-",
                        category="representation",
                        test_type="min_ethnicity_name_representation_proportion",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value),
                    )
                    sample_list.append(sample)

                else:
                    sample = MinScoreSample(
                        original="-",
                        category="representation",
                        test_type="min_ethnicity_name_representation_proportion",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value),
                    )
                    sample_list.append(sample)

        return sample_list

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs) -> List[Sample]:
        """Computes the actual results for the ethnicity representation test.

        Args:
            sample_list (List[Sample]): The input data to be evaluated for representation test.
            model (ModelAPI): The model to be used for evaluation.

        Returns:
            List[Sample]: The list of samples with actual results.
        """
        progress = kwargs.get("progress_bar", False)

        entity_representation = RepresentationOperation.get_ethnicity_representation_dict(
            kwargs["raw_data"]
        )

        for sample in sample_list:
            if sample.test_type == "min_ethnicity_name_representation_proportion":
                entity_representation_proportion = (
                    RepresentationOperation.get_entity_representation_proportions(
                        entity_representation
                    )
                )
                actual_representation = {
                    **default_ehtnicity_representation,
                    **entity_representation_proportion,
                }

                sample.actual_results = MinScoreOutput(
                    min_score=round(actual_representation[sample.test_case], 2)
                )
                sample.state = "done"

            elif sample.test_type == "min_ethnicity_name_representation_count":
                actual_representation = {
                    **default_ehtnicity_representation,
                    **entity_representation,
                }
                sample.actual_results = MinScoreOutput(
                    min_score=round(actual_representation[sample.test_case], 2)
                )
                sample.state = "done"

            if progress:
                progress.update(1)

        return sample_list


class LabelRepresentation(BaseRepresentation):
    """Subclass of BaseRepresentation that implements the label representation test.

    Attributes:
        alias_name (List[str]): The list of test names that identify the representation measure.
        supported_tasks (List[str]): name of the supported task for the representation measure
    """

    alias_name = ["min_label_representation_count", "min_label_representation_proportion"]

    supported_tasks = ["ner", "text-classification"]

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> Union[List[MinScoreQASample], List[MinScoreSample]]:
        """Compute the label representation measure

        Args:
            test (str): name of the test
            data (List[Sample]): The input data to be evaluated for representation test.
            params (Dict): parameters specified in config.

        Raises:
            ValueError: If sum of specified proportions in config is greater than 1

        Returns:
            Union[List[MinScoreQASample], List[MinScoreSample]]: Label Representation test results.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        sample_list = []
        labels = [s.expected_results.predictions for s in data]
        if isinstance(data[0].expected_results, NEROutput):
            labels = [x.entity.split("-")[-1] for sentence in labels for x in sentence]
        elif isinstance(data[0].expected_results, SequenceClassificationOutput):
            labels = [x.label for sentence in labels for x in sentence]
        labels = set(labels)

        if test == "min_label_representation_count":
            if not params:
                expected_representation = {k: 10 for k in labels}

            else:
                if isinstance(params["min_count"], dict):
                    expected_representation = params["min_count"]

                elif isinstance(params["min_count"], int):
                    expected_representation = {key: params["min_count"] for key in labels}

            for key, value in expected_representation.items():
                sample = MinScoreSample(
                    original="-",
                    category="representation",
                    test_type="min_label_representation_count",
                    test_case=key,
                    expected_results=MinScoreOutput(min_score=value),
                )
                sample_list.append(sample)

        else:
            if not params:
                expected_representation = {k: (1 / len(k)) * 0.8 for k in labels}

            else:
                if isinstance(params["min_proportion"], dict):
                    expected_representation = params["min_proportion"]

                    if sum(expected_representation.values()) > 1:
                        raise ValueError(
                            Errors.E064(var="min_label_representation_proportion")
                        )

                elif isinstance(params["min_proportion"], float):
                    expected_representation = {
                        key: params["min_proportion"] for key in labels
                    }
                    if sum(expected_representation.values()) > 1:
                        raise ValueError(
                            Errors.E064(var="min_label_representation_proportion")
                        )

            for key, value in expected_representation.items():
                sample = MinScoreSample(
                    original="-",
                    category="representation",
                    test_type=test,
                    test_case=key,
                    expected_results=MinScoreOutput(min_score=value),
                )
                sample_list.append(sample)

        return sample_list

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs) -> List[Sample]:
        """Computes the actual representation of the labels in the dataset.

        Args:
            sample_list (List[Sample]): The input data to be evaluated for representation test.
            model (ModelAPI): The model to be evaluated.

        Returns:
            List[Sample]: Label Representation test results.
        """
        progress = kwargs.get("progress_bar", False)

        entity_representation = RepresentationOperation.get_label_representation_dict(
            kwargs["raw_data"]
        )

        for sample in sample_list:
            if progress:
                progress.update(1)

            if sample.test_type == "min_label_representation_proportion":
                entity_representation_proportion = (
                    RepresentationOperation.get_entity_representation_proportions(
                        entity_representation
                    )
                )
                actual_representation = {**entity_representation_proportion}
                sample.actual_results = MinScoreOutput(
                    min_score=round(actual_representation[sample.test_case], 2)
                )
                sample.state = "done"
            elif sample.test_type == "min_label_representation_count":
                actual_representation = {**entity_representation}
                sample.actual_results = MinScoreOutput(
                    min_score=round(actual_representation[sample.test_case], 2)
                )
                sample.state = "done"

        return sample_list


class ReligionRepresentation(BaseRepresentation):
    """Subclass of BaseRepresentation that implements the religion representation test.

    Attributes:
        alias_name (List[str]): The list of test names that identify the representation measure.
        supported_tasks (List[str]): name of the supported task for the representation measure
    """

    alias_name = [
        "min_religion_name_representation_count",
        "min_religion_name_representation_proportion",
    ]
    supported_tasks = [
        "ner",
        "text-classification",
        "question-answering",
        "summarization",
    ]

    min_count = TypedDict(
        "min_count",
        muslim=int,
        hindu=int,
        sikh=int,
        christian=int,
        jain=int,
        buddhist=int,
        parsi=int,
    )

    min_proportion = TypedDict(
        "min_proportion",
        muslim=float,
        hindu=float,
        sikh=float,
        christian=float,
        jain=float,
        buddhist=float,
        parsi=float,
    )

    TestConfig = TypedDict(
        "TestConfig",
        min_count=Union[min_count, int],
        min_proportion=Union[min_proportion, float],
    )

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> Union[List[MinScoreQASample], List[MinScoreSample]]:
        """Compute the religion representation measure

        Args:
            test (str): name of the test
            data (List[Sample]): The input data to be evaluated for representation test.
            params : parameters specified in config.

        Raises:
            ValueError: If sum of specified proportions in config is greater than 1

        Returns:
            Union[List[MinScoreQASample], List[MinScoreSample]]: Religion Representation test results.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        sample_list = []
        if test == "min_religion_name_representation_count":
            if not params:
                expected_representation = {
                    "muslim": 5,
                    "hindu": 5,
                    "sikh": 5,
                    "christian": 5,
                    "jain": 5,
                    "buddhist": 5,
                    "parsi": 5,
                }

            else:
                if isinstance(params["min_count"], dict):
                    expected_representation = params["min_count"]

                elif isinstance(params["min_count"], int):
                    expected_representation = {
                        key: params["min_count"]
                        for key in default_religion_representation
                    }

            for key, value in expected_representation.items():
                if hasattr(data[0], "task") and data[0].task == "question-answering":
                    sample = MinScoreQASample(
                        original_question=key,
                        original_context="-",
                        options="-",
                        perturbed_question="-",
                        perturbed_context="-",
                        category="representation",
                        test_type="min_religion_name_representation_count",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value),
                    )
                    sample_list.append(sample)
                else:
                    sample = MinScoreSample(
                        original="-",
                        category="representation",
                        test_type="min_religion_name_representation_count",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value),
                    )
                    sample_list.append(sample)

        else:
            if not params:
                expected_representation = {
                    "muslim": 0.11,
                    "hindu": 0.11,
                    "sikh": 0.11,
                    "christian": 0.11,
                    "jain": 0.11,
                    "buddhist": 0.11,
                    "parsi": 0.11,
                }

            else:
                if isinstance(params["min_proportion"], dict):
                    expected_representation = params["min_proportion"]

                    if sum(expected_representation.values()) > 1:
                        raise ValueError(
                            Errors.E064(var="min_religion_name_representation_proportion")
                        )

                elif isinstance(params["min_proportion"], float):
                    expected_representation = {
                        key: params["min_proportion"]
                        for key in default_religion_representation
                    }
                    if sum(expected_representation.values()) > 1:
                        raise ValueError(
                            Errors.E064(var="min_religion_name_representation_proportion")
                        )

            entity_representation = (
                RepresentationOperation.get_religion_name_representation_dict(data)
            )
            entity_representation_proportion = (
                RepresentationOperation.get_entity_representation_proportions(
                    entity_representation
                )
            )
            actual_representation = {
                **default_religion_representation,
                **entity_representation_proportion,
            }
            for key, value in expected_representation.items():
                if hasattr(data[0], "task") and data[0].task == "question-answering":
                    sample = MinScoreQASample(
                        original_question=key,
                        original_context="-",
                        options="-",
                        perturbed_question="-",
                        perturbed_context="-",
                        category="representation",
                        test_type="min_religion_name_representation_proportion",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value),
                    )
                    sample_list.append(sample)
                else:
                    sample = MinScoreSample(
                        original="-",
                        category="representation",
                        test_type="min_religion_name_representation_proportion",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value),
                        actual_results=MinScoreOutput(
                            min_score=actual_representation[key]
                        ),
                        state="done",
                    )
                    sample_list.append(sample)

        return sample_list

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs) -> List[Sample]:
        """Computes the actual representation of religion names in the data.

        Args:
            sample_list (List[Sample]): The input data to be evaluated for representation test.
            model (ModelAPI): The model to be evaluated.

        Returns:
            List[Sample]: Religion Representation test results.

        """
        progress = kwargs.get("progress_bar", False)

        entity_representation = (
            RepresentationOperation.get_religion_name_representation_dict(
                kwargs["raw_data"]
            )
        )

        for sample in sample_list:
            if sample.test_type == "min_religion_name_representation_proportion":
                entity_representation_proportion = (
                    RepresentationOperation.get_entity_representation_proportions(
                        entity_representation
                    )
                )
                actual_representation = {
                    **default_religion_representation,
                    **entity_representation_proportion,
                }

                sample.actual_results = MinScoreOutput(
                    min_score=round(actual_representation[sample.test_case], 2)
                )
                sample.state = "done"

            elif sample.test_type == "min_religion_name_representation_count":
                actual_representation = {
                    **default_religion_representation,
                    **entity_representation,
                }

                sample.actual_results = MinScoreOutput(
                    min_score=round(actual_representation[sample.test_case], 2)
                )
                sample.state = "done"

            if progress:
                progress.update(1)

        return sample_list


class CountryEconomicRepresentation(BaseRepresentation):
    """Subclass of BaseRepresentation that implements the country economic representation test.

    Attributes:
        alias_name (List[str]): The list of test names that identify the representation measure.
        supported_tasks (List[str]): name of the supported task for the representation measure
    """

    alias_name = [
        "min_country_economic_representation_count",
        "min_country_economic_representation_proportion",
    ]

    min_count = TypedDict(
        "min_count",
        high_income=int,
        low_income=int,
        lower_middle_income=int,
        upper_middle_income=int,
    )

    min_proportion = TypedDict(
        "min_proportion",
        high_income=float,
        low_income=float,
        lower_middle_income=float,
        upper_middle_income=float,
    )

    TestConfig = TypedDict(
        "TestConfig",
        min_count=Union[min_count, int],
        min_proportion=Union[min_proportion, float],
    )

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> Union[List[MinScoreQASample], List[MinScoreSample]]:
        """Compute the country economic representation measure

        Args:
            test (str): name of the test
            data (List[Sample]): The input data to be evaluated for representation test.
            params : parameters specified in config.

        Raises:
            ValueError: If sum of specified proportions in config is greater than 1

        Returns:
            Union[List[MinScoreQASample], List[MinScoreSample]]: Country Economic Representation test results.
        """
        sample_list = []

        if test == "min_country_economic_representation_count":
            if not params:
                expected_representation = {
                    "high_income": 10,
                    "low_income": 10,
                    "lower_middle_income": 10,
                    "upper_middle_income": 10,
                }

            else:
                if isinstance(params["min_count"], dict):
                    expected_representation = params["min_count"]

                elif isinstance(params["min_count"], int):
                    expected_representation = {
                        key: params["min_count"]
                        for key in default_economic_country_representation
                    }

            for key, value in expected_representation.items():
                if hasattr(data[0], "task") and data[0].task == "question-answering":
                    sample = MinScoreQASample(
                        original_question=key,
                        original_context="-",
                        options="-",
                        perturbed_question="-",
                        perturbed_context="-",
                        category="representation",
                        test_type="min_country_economic_representation_count",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value),
                    )
                    sample_list.append(sample)
                else:
                    sample = MinScoreSample(
                        original="-",
                        category="representation",
                        test_type="min_country_economic_representation_count",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value),
                    )
                    sample_list.append(sample)

        else:
            if not params:
                expected_representation = {
                    "high_income": 0.20,
                    "low_income": 0.20,
                    "lower_middle_income": 0.20,
                    "upper_middle_income": 0.20,
                }

            else:
                if isinstance(params["min_proportion"], dict):
                    expected_representation = params["min_proportion"]

                    if sum(expected_representation.values()) > 1:
                        raise ValueError(
                            Errors.E064(
                                var="min_country_economic_representation_proportion"
                            )
                        )

                elif isinstance(params["min_proportion"], float):
                    expected_representation = {
                        key: params["min_proportion"]
                        for key in default_economic_country_representation
                    }
                    if sum(expected_representation.values()) > 1:
                        raise ValueError(
                            Errors.E064(
                                var="min_country_economic_representation_proportion"
                            )
                        )

            for key, value in expected_representation.items():
                if hasattr(data[0], "task") and data[0].task == "question-answering":
                    sample = MinScoreQASample(
                        original_question=key,
                        original_context="-",
                        options="-",
                        perturbed_question="-",
                        perturbed_context="-",
                        category="representation",
                        test_type="min_country_economic_representation_proportion",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value),
                    )
                    sample_list.append(sample)

                else:
                    sample = MinScoreSample(
                        original="-",
                        category="representation",
                        test_type="min_country_economic_representation_proportion",
                        test_case=key,
                        expected_results=MinScoreOutput(min_score=value),
                    )
                    sample_list.append(sample)

        return sample_list

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs) -> List[Sample]:
        """Computes the actual results for the country economic representation test.

        Args:
            sample_list (List[Sample]): The input data to be evaluated for representation test.
            model (ModelAPI): The model to be used for evaluation.

        Returns:
            List[Sample]: Country Economic Representation test results.

        """
        progress = kwargs.get("progress_bar", False)

        entity_representation = (
            RepresentationOperation.get_country_economic_representation_dict(
                kwargs["raw_data"]
            )
        )

        for sample in sample_list:
            if sample.test_type == "min_country_economic_representation_proportion":
                entity_representation_proportion = (
                    RepresentationOperation.get_entity_representation_proportions(
                        entity_representation
                    )
                )
                actual_representation = {
                    **default_economic_country_representation,
                    **entity_representation_proportion,
                }

                sample.actual_results = MinScoreOutput(
                    min_score=round(actual_representation[sample.test_case], 2)
                )
                sample.state = "done"

            elif sample.test_type == "min_country_economic_representation_count":
                actual_representation = {
                    **default_economic_country_representation,
                    **entity_representation,
                }

                sample.actual_results = MinScoreOutput(
                    min_score=round(actual_representation[sample.test_case], 2)
                )
                sample.state = "done"

            if progress:
                progress.update(1)

        return sample_list
