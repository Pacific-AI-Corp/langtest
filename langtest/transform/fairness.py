import asyncio
import pandas as pd
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, TypedDict, Union

from langtest.modelhandler.modelhandler import ModelAPI
from langtest.utils.custom_types import (
    MaxScoreOutput,
    MaxScoreSample,
    MinScoreOutput,
    MinScoreSample,
    NERSample,
    QASample,
    SequenceClassificationSample,
    Sample,
)
from langtest.utils.util_metrics import calculate_f1_score, calculate_f1_score_multi_label
from langtest.utils.custom_types.helpers import default_user_prompt
from langtest.errors import Errors
from langtest.transform.base import ITests


class FairnessTestFactory(ITests):
    """
    A class for performing fairness tests on a given dataset.
    """

    alias_name = "fairness"
    model_result = None

    def __init__(self, data_handler: List[Sample], tests: Dict, **kwargs) -> None:
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
        Runs the fairness test and returns the resulting `Sample` objects.

        Returns:
            List[Sample]:
                A list of `Sample` objects representing the resulting dataset after running the fairness test.
        """
        all_samples = []

        if self._data_handler[0].expected_results is None:
            raise RuntimeError(Errors.E052(var="fairness"))

        for test_name, params in self.tests.items():
            transformed_samples = self.supported_tests[test_name].transform(
                test_name, None, params
            )

            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)

        return all_samples

    @staticmethod
    def available_tests() -> Dict[str, type["BaseFairness"]]:
        """
        Get a dictionary of all available tests, with their names as keys and their corresponding classes as values.

        Returns:
            dict: A dictionary of test names and classes.
        """

        return BaseFairness.test_types

    @classmethod
    def run(
        cls,
        sample_list: Dict[str, List[Sample]],
        model: ModelAPI,
        raw_data: List[Sample],
        **kwargs,
    ):
        """
        Runs the fairness tests on the given model and dataset.

        Args:
            sample_list (Dict[str, List[Sample]]): A dictionary of test names and corresponding `Sample` objects.
            model (ModelAPI): The model to be tested.
            raw_data (List[Sample]): The raw dataset.

        """
        raw_data_copy = [x.model_copy() for x in raw_data]
        grouped_label = {}
        grouped_data = cls.get_gendered_data(raw_data_copy)
        for gender, data in grouped_data.items():
            if len(data) == 0:
                grouped_label[gender] = [[], []]
            else:
                if isinstance(data[0], NERSample):

                    def predict_ner(sample: Sample):
                        prediction = model.predict(sample.original)
                        sample.actual_results = prediction
                        sample.gender = gender
                        return prediction

                    X_test = pd.Series(data)
                    X_test.apply(predict_ner)
                    y_true = pd.Series(data).apply(
                        lambda x: [y.entity for y in x.expected_results.predictions]
                    )
                    y_pred = pd.Series(data).apply(
                        lambda x: [y.entity for y in x.actual_results.predictions]
                    )
                    valid_indices = y_true.apply(len) == y_pred.apply(len)
                    y_true = y_true[valid_indices]
                    y_pred = y_pred[valid_indices]
                    y_true = y_true.explode()
                    y_pred = y_pred.explode()
                    y_pred = y_pred.apply(lambda x: x.split("-")[-1]).reset_index(
                        drop=True
                    )
                    y_true = y_true.apply(lambda x: x.split("-")[-1]).reset_index(
                        drop=True
                    )

                elif isinstance(data[0], SequenceClassificationSample):
                    is_mutli_label = raw_data_copy[0].expected_results.multi_label

                    def predict_text_classification(sample: Sample):
                        prediction = model.predict(sample.original)
                        sample.actual_results = prediction
                        sample.gender = gender
                        return prediction

                    X_test = pd.Series(data)
                    X_test.apply(predict_text_classification)

                    y_true = pd.Series(data).apply(
                        lambda x: [y.label for y in x.expected_results.predictions]
                    )
                    y_pred = pd.Series(data).apply(
                        lambda x: [y.label for y in x.actual_results.predictions]
                    )

                    if is_mutli_label:
                        kwargs["is_multi_label"] = is_mutli_label

                    else:
                        y_true = y_true.apply(lambda x: x[0])
                        y_pred = y_pred.apply(lambda x: x[0])

                        y_true = y_true.explode()
                        y_pred = y_pred.explode()

                elif data[0].task == "question-answering":
                    from ..utils.custom_types.helpers import (
                        build_qa_input,
                        build_qa_prompt,
                    )

                    if data[0].dataset_name is None:
                        dataset_name = "default_question_answering_prompt"
                    else:
                        dataset_name = data[0].dataset_name.split("-")[0].lower()

                    if data[0].expected_results is None:
                        raise RuntimeError(Errors.E053(dataset_name=dataset_name))

                    def predict_question_answering(sample: Sample):
                        input_data = build_qa_input(
                            context=sample.original_context,
                            question=sample.original_question,
                            options=sample.options,
                        )
                        prompt = build_qa_prompt(input_data, dataset_name, **kwargs)
                        server_prompt = kwargs.get("server_prompt", " ")
                        prediction = model(
                            text=input_data, prompt=prompt, server_prompt=server_prompt
                        ).strip()
                        sample.actual_results = prediction
                        sample.gender = gender
                        return prediction

                    y_true = pd.Series(data).apply(lambda x: x.expected_results)
                    X_test = pd.Series(data)

                    y_pred = X_test.apply(predict_question_answering)
                elif data[0].task == "summarization":
                    if data[0].dataset_name is None:
                        dataset_name = "default_summarization_prompt"
                    else:
                        dataset_name = data[0].dataset_name.split("-")[0].lower()
                    prompt_template = kwargs.get(
                        "user_prompt", default_user_prompt.get(dataset_name, "")
                    )
                    if data[0].expected_results is None:
                        raise RuntimeError(Errors.E053(dataset_name=dataset_name))

                    def predict_summarization(sample: Sample):
                        prediction = model(
                            text={"context": sample.original},
                            prompt={
                                "template": prompt_template,
                                "input_variables": ["context"],
                            },
                        ).strip()
                        sample.actual_results = prediction
                        sample.gender = gender
                        return prediction

                    y_true = pd.Series(data).apply(lambda x: x.expected_results)
                    X_test = pd.Series(data)
                    y_pred = X_test.apply(predict_summarization)

                if kwargs["is_default"]:
                    y_pred = y_pred.apply(
                        lambda x: (
                            "1"
                            if x in ["pos", "LABEL_1", "POS"]
                            else ("0" if x in ["neg", "LABEL_0", "NEG"] else x)
                        )
                    )

                grouped_label[gender] = [y_true, y_pred]

        supported_tests = cls.available_tests()
        from ..utils.custom_types.helpers import TestResultManager

        cls.model_result = TestResultManager().prepare_model_response(raw_data_copy)
        kwargs["task"] = raw_data[0].task
        tasks = []
        for test_name, samples in sample_list.items():
            tasks.append(
                supported_tests[test_name].async_run(
                    samples, grouped_label, grouped_data=grouped_data, **kwargs
                )
            )
        return tasks

    @staticmethod
    def get_gendered_data(data: List[Sample]) -> Dict[str, List[Sample]]:
        """Split list of samples into gendered lists."""
        from langtest.utils.gender_classifier import GenderClassifier

        classifier = GenderClassifier()

        data = pd.Series(data)
        if isinstance(data[0], QASample):
            sentences = data.apply(
                lambda x: f"{x.original_context} {x.original_question}"
            )
        else:
            sentences = data.apply(lambda x: x.original)

        genders = sentences.apply(classifier.predict)
        gendered_data = {
            "male": data[genders == "male"].tolist(),
            "female": data[genders == "female"].tolist(),
            "unknown": data[genders == "unknown"].tolist(),
        }
        return gendered_data


class BaseFairness(ABC):
    """Abstract base class for implementing accuracy measures.

    Attributes:
        alias_name (str): A name or list of names that identify the accuracy measure.

    Methods:
        transform(data: List[Sample], params: Dict) -> Union[List[MinScoreSample], List[MaxScoreSample]]:
            Transforms the input data into an output based on the implemented accuracy measure.
    """

    test_types = defaultdict(lambda: BaseFairness)
    alias_name = None
    supported_tasks = ["ner", "text-classification"]

    TestConfig = TypedDict(
        "TestConfig",
        min_score=Union[float, Dict[str, float]],
        max_score=Union[float, Dict[str, float]],
    )

    @staticmethod
    @abstractmethod
    def transform(
        data: List[Sample], params: Dict
    ) -> Union[List[MinScoreSample], List[MaxScoreSample]]:
        """Abstract method that implements the computation of the given measure.

        Args:
            data (List[Sample]): The input data to be transformed.
            params (Dict): parameters for tests configuration
        Returns:
            Union[List[MinScoreSample], List[MaxScoreSample]]: The transformed data based on the implemented measure.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    async def run(
        sample_list: List[MinScoreSample], categorised_data, **kwargs
    ) -> List[Sample]:
        """Computes the score for the given data.

        Args:
            sample_list (List[MinScoreSample]): The input data to be transformed.
            model (ModelAPI): The model to be used for the computation.

        Returns:
            List[MinScoreSample]: The transformed samples.
        """
        raise NotImplementedError()

    @classmethod
    async def async_run(cls, sample_list: List[Sample], model: ModelAPI, **kwargs):
        """Creates a task for the run method.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for the computation.

        Returns:
            asyncio.Task: The task for the run method.

        """
        created_task = asyncio.create_task(cls.run(sample_list, model, **kwargs))
        return created_task

    def __init_subclass__(cls) -> None:
        alias = cls.alias_name if isinstance(cls.alias_name, list) else [cls.alias_name]
        for name in alias:
            BaseFairness.test_types[name] = cls


class MinGenderF1Score(BaseFairness):
    """Subclass of BaseFairness that implements the minimum F1 score.

    Attributes:
        alias_name (str): The name "min_gender_f1_score" identifying the minimum F1 score.

    Methods:
        transform(test: str, data: List[Sample], params: Dict) -> List[MinScoreSample]:
            Transforms the input data into an output based on the minimum F1 score.
    """

    alias_name = ["min_gender_f1_score"]

    min_score = TypedDict(
        "min_score",
        male=float,
        female=float,
        unknown=float,
    )

    TestConfig = TypedDict(
        "TestConfig",
        min_score=Union[min_score, float],
    )

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> List[MinScoreSample]:
        """
        Computes the minimum F1 score for the given data.

        Args:
            test (str): The test alias name.
            data (List[Sample]): The input data to be transformed.
            params (Dict): Parameters for tests configuration.

        Returns:
            List[MinScoreSample]: The transformed data based on the minimum F1 score.
        """

        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {
                "male": params["min_score"],
                "female": params["min_score"],
                "unknown": params["min_score"],
            }

        samples = []
        for key, val in min_scores.items():
            sample = MinScoreSample(
                original=None,
                category="fairness",
                test_type="min_gender_f1_score",
                test_case=key,
                expected_results=MinScoreOutput(min_score=val),
            )

            samples.append(sample)
        return samples

    @staticmethod
    async def run(
        sample_list: List[MinScoreSample], grouped_label, **kwargs
    ) -> List[MinScoreSample]:
        """
        Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): The input data samples.
            grouped_label: A dictionary containing grouped labels where each key corresponds to a test case
                and the value is a tuple containing true labels and predicted labels.
            **kwargs: Additional keyword arguments.

        Returns:
            List[MinScoreSample]: The evaluated data samples.
        """
        progress = kwargs.get("progress_bar", False)

        is_multi_label = kwargs.get("is_multi_label", False)

        for sample in sample_list:
            data = grouped_label[sample.test_case]
            if len(data[0]) > 0:
                if is_multi_label:
                    macro_f1_score = calculate_f1_score_multi_label(
                        data[0].to_list(),
                        data[1].to_list(),
                        average="macro",
                        zero_division=0,
                    )
                else:
                    macro_f1_score = calculate_f1_score(
                        data[0].to_list(),
                        data[1].to_list(),
                        average="macro",
                        zero_division=0,
                    )
            else:
                macro_f1_score = 1

            sample.actual_results = MinScoreOutput(min_score=macro_f1_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list


class MaxGenderF1Score(BaseFairness):
    """Subclass of BaseFairness that implements the maximum F1 score.

    Attributes:
        alias_name (str): The name "max_gender_f1_score" identifying the maximum F1 score.

    Methods:
        transform(test: str, data: List[Sample], params: Dict) -> List[MaxScoreSample]:
            Transforms the input data into an output based on the maximum F1 score.
    """

    alias_name = ["max_gender_f1_score"]

    max_score = TypedDict(
        "max_score",
        male=float,
        female=float,
        unknown=float,
    )

    TestConfig = TypedDict(
        "TestConfig",
        max_score=Union[max_score, float],
    )

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> List[MaxScoreSample]:
        """
        Computes the maximum F1 score for the given data.

        Args:
            test (str): The test alias name.
            data (List[Sample]): The input data to be transformed.
            params (Dict): Parameters for tests configuration.

        Returns:
            List[MaxScoreSample]: The transformed data based on the maximum F1 score.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"
        if isinstance(params["max_score"], dict):
            max_scores = params["max_score"]
        elif isinstance(params["max_score"], float):
            max_scores = {
                "male": params["max_score"],
                "female": params["max_score"],
                "unknown": params["max_score"],
            }

        samples = []
        for key, val in max_scores.items():
            sample = MaxScoreSample(
                original=None,
                category="fairness",
                test_type="max_gender_f1_score",
                test_case=key,
                expected_results=MaxScoreOutput(max_score=val),
            )

            samples.append(sample)
        return samples

    @staticmethod
    async def run(
        sample_list: List[MaxScoreSample], grouped_label, **kwargs
    ) -> List[MaxScoreSample]:
        """
        Computes the maximum F1 score for the given data.

        Args:
            sample_list (List[MaxScoreSample]): The input data samples.
            grouped_label: A dictionary containing grouped labels where each key corresponds to a test case
                and the value is a tuple containing true labels and predicted labels.
            **kwargs: Additional keyword arguments.

        Returns:
            List[MaxScoreSample]: The evaluated data samples.
        """
        progress = kwargs.get("progress_bar", False)
        is_multi_label = kwargs.get("is_multi_label", False)

        for sample in sample_list:
            data = grouped_label[sample.test_case]
            if len(data[0]) > 0:
                if is_multi_label:
                    macro_f1_score = calculate_f1_score_multi_label(
                        data[0].to_list(),
                        data[1].to_list(),
                        average="macro",
                        zero_division=0,
                    )
                else:
                    macro_f1_score = calculate_f1_score(
                        data[0].to_list(),
                        data[1].to_list(),
                        average="macro",
                        zero_division=0,
                    )
            else:
                macro_f1_score = 1

            sample.actual_results = MaxScoreOutput(max_score=macro_f1_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list


class MinGenderRougeScore(BaseFairness):
    """
    Subclass of BaseFairness that implements the minimum Rouge score.

    Attributes:
        alias_name (List[str]): Alias names for the evaluation method.
        supported_tasks (List[str]): Supported tasks for this evaluation method.

    Methods:
        transform(test: str, data: List[Sample], params: Dict) -> List[MinScoreSample]:
            Transforms the input data into an output based on the minimum Rouge score.
        run(sample_list: List[MinScoreSample], grouped_label, **kwargs) -> List[MinScoreSample]:
            Computes the minimum Rouge score for the given data.

    """

    alias_name = [
        "min_gender_rouge1_score",
        "min_gender_rouge2_score",
        "min_gender_rougeL_score",
        "min_gender_rougeLsum_score",
    ]
    supported_tasks = ["question-answering", "summarization"]

    min_score = TypedDict(
        "min_score",
        male=float,
        female=float,
        unknown=float,
    )

    TestConfig = TypedDict(
        "TestConfig",
        min_score=Union[min_score, float],
    )

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> List[MinScoreSample]:
        """
        Transforms the data for evaluation based on the minimum Rouge score.

        Args:
            test (str): The test alias name.
            data (List[Sample]): The input data to be transformed.
            params (Dict): Parameters for tests configuration.

        Returns:
            List[MinScoreSample]: The transformed data samples based on the minimum Rouge score.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {
                "male": params["min_score"],
                "female": params["min_score"],
                "unknown": params["min_score"],
            }

        samples = []
        for key, val in min_scores.items():
            sample = MinScoreSample(
                original=None,
                category="fairness",
                test_type=test,
                test_case=key,
                expected_results=MinScoreOutput(min_score=val),
            )

            samples.append(sample)
        return samples

    @staticmethod
    async def run(
        sample_list: List[MinScoreSample], grouped_label, **kwargs
    ) -> List[MinScoreSample]:
        """
        Computes the minimum Rouge score for the given data.

        Args:
            sample_list (List[MinScoreSample]): The input data samples.
            grouped_label: A dictionary containing grouped labels where each key corresponds to a test case
                and the value is a tuple containing true labels and predicted labels.
            **kwargs: Additional keyword arguments.

        Returns:
            List[MinScoreSample]: The evaluated data samples.
        """
        import evaluate

        progress = kwargs.get("progress_bar", False)
        task = kwargs.get("task", None)

        for sample in sample_list:
            data = grouped_label[sample.test_case]
            if len(data[0]) > 0:
                if task == "question-answering" or task == "summarization":
                    em = evaluate.load("rouge")
                    macro_f1_score = em.compute(references=data[0], predictions=data[1])[
                        sample.test_type.split("_")[2]
                    ]
                else:
                    macro_f1_score = calculate_f1_score(
                        [x[0] for x in data[0]], data[1], average="macro", zero_division=0
                    )
            else:
                macro_f1_score = 1

            sample.actual_results = MinScoreOutput(min_score=macro_f1_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list


class MaxGenderRougeScore(BaseFairness):
    """
    Subclass of BaseFairness that implements the Rouge score.

    Attributes:
        alias_name (List[str]): Alias names for the evaluation method.
        supported_tasks (List[str]): Supported tasks for this evaluation method.

    Methods:
        transform(test: str, data: List[Sample], params: Dict) -> List[MaxScoreSample]:
            Transforms the input data into an output based on the Rouge score.
        run(sample_list: List[MaxScoreSample], grouped_label, **kwargs) -> List[MaxScoreSample]:
            Computes the maximum Rouge score for the given data.

    """

    alias_name = [
        "max_gender_rouge1_score",
        "max_gender_rouge2_score",
        "max_gender_rougeL_score",
        "max_gender_rougeLsum_score",
    ]
    supported_tasks = ["question-answering", "summarization"]

    max_score = TypedDict(
        "max_score",
        male=float,
        female=float,
        unknown=float,
    )

    TestConfig = TypedDict(
        "TestConfig",
        max_score=Union[max_score, float],
    )

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> List[MaxScoreSample]:
        """
        Transforms the data for evaluation based on the Rouge score.

        Args:
            test (str): The test alias name.
            data (List[Sample]): The input data to be transformed.
            params (Dict): Parameters for tests configuration.

        Returns:
            List[MaxScoreSample]: The transformed data samples based on the Rouge score.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        if isinstance(params["max_score"], dict):
            max_scores = params["max_score"]
        elif isinstance(params["max_score"], float):
            max_scores = {
                "male": params["max_score"],
                "female": params["max_score"],
                "unknown": params["max_score"],
            }

        samples = []
        for key, val in max_scores.items():
            sample = MaxScoreSample(
                original=None,
                category="fairness",
                test_type=test,
                test_case=key,
                expected_results=MaxScoreOutput(max_score=val),
            )

            samples.append(sample)
        return samples

    @staticmethod
    async def run(
        sample_list: List[MaxScoreSample], grouped_label, **kwargs
    ) -> List[MaxScoreSample]:
        """
        Computes the maximum Rouge score for the given data.

        Args:
            sample_list (List[MaxScoreSample]): The input data samples.
            grouped_label: A dictionary containing grouped labels where each key corresponds to a test case
                and the value is a tuple containing true labels and predicted labels.
            **kwargs: Additional keyword arguments.

        Returns:
            List[MaxScoreSample]: The evaluated data samples.
        """
        import evaluate

        progress = kwargs.get("progress_bar", False)
        task = kwargs.get("task", None)

        for sample in sample_list:
            data = grouped_label[sample.test_case]
            if len(data[0]) > 0:
                if task == "question-answering" or task == "summarization":
                    em = evaluate.load("rouge")
                    rouge_score = em.compute(references=data[0], predictions=data[1])[
                        sample.test_type.split("_")[2]
                    ]
                else:
                    rouge_score = calculate_f1_score(
                        [x[0] for x in data[0]], data[1], average="macro", zero_division=0
                    )
            else:
                rouge_score = 1

            sample.actual_results = MaxScoreOutput(max_score=rouge_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list


class MinGenderLLMEval(BaseFairness):
    """
    Class for evaluating fairness based on minimum gender performance in question-answering tasks using a Language Model.

    Attributes:
        alias_name (List[str]): Alias names for the evaluation method.
        supported_tasks (List[str]): Supported tasks for this evaluation method.

    Methods:
        transform(cls, test: str, data: List[Sample], params: Dict) -> List[MaxScoreSample]: Transforms data for evaluation.
        run(sample_list: List[MaxScoreSample], grouped_label: Dict[str, Tuple[List, List]], **kwargs) -> List[MaxScoreSample]: Runs the evaluation process.

    """

    alias_name = ["min_gender_llm_eval"]
    supported_tasks = ["question-answering"]
    eval_model = None

    min_score = TypedDict(
        "min_score",
        male=float,
        female=float,
        unknown=float,
    )

    TestConfig = TypedDict(
        "TestConfig",
        min_score=Union[min_score, float],
    )

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> List[MaxScoreSample]:
        """
        Transforms the data for evaluation.

        Args:
            test (str): The test alias name.
            data (List[Sample]): The data to be transformed.
            params (Dict): Parameters for transformation.

        Returns:
            List[MaxScoreSample]: The transformed data samples.

        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        from ..langtest import EVAL_MODEL
        from ..langtest import HARNESS_CONFIG as harness_config

        model = params.get("model", None)
        hub = params.get("hub", None)
        if model and hub:
            from ..tasks import TaskManager

            load_eval_model = TaskManager("question-answering")
            cls.eval_model = load_eval_model.model(
                model, hub, **harness_config.get("model_parameters", {})
            )
        else:
            cls.eval_model = EVAL_MODEL

        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {
                "male": params["min_score"],
                "female": params["min_score"],
                "unknown": params["min_score"],
            }

        samples = []
        for key, val in min_scores.items():
            sample = MinScoreSample(
                original=None,
                category="fairness",
                test_type=test,
                test_case=key,
                expected_results=MinScoreOutput(min_score=val),
            )

            samples.append(sample)
        return samples

    @staticmethod
    async def run(
        sample_list: List[MaxScoreSample], grouped_label, **kwargs
    ) -> List[MaxScoreSample]:
        """
        Runs the evaluation process using Language Model.

        Args:
            sample_list: The input data samples.
            grouped_label: A dictionary containing grouped labels where each key corresponds to a test case
                and the value is a tuple containing true labels and predicted labels.
            **kwargs: Additional keyword arguments.

        Returns:
            The evaluated data samples.
        """

        grouped_data = kwargs.get("grouped_data")
        from ..utils.custom_types.helpers import is_pass_llm_eval

        eval_model = MinGenderLLMEval.eval_model
        progress = kwargs.get("progress_bar", False)

        def eval():
            results = []
            for true_list, pred, sample in zip(data[0], data[1], X_test):
                result = is_pass_llm_eval(
                    eval_model=eval_model,
                    dataset_name=sample.dataset_name,
                    original_question=sample.original_question,
                    answer="\n".join(map(str, true_list)),
                    perturbed_question=sample.original_question,
                    prediction=pred,
                )
                if result:
                    results.append(1)
                else:
                    results.append(0)
            total_samples = len(results)
            passed_samples = sum(results)
            accuracy = passed_samples / max(total_samples, 1)
            return accuracy

        for sample in sample_list:
            data = grouped_label[sample.test_case]
            X_test = grouped_data[sample.test_case]
            if len(data[0]) > 0:
                eval_score = eval()
            else:
                eval_score = 1

            sample.actual_results = MinScoreOutput(min_score=eval_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list


class MaxGenderLLMEval(BaseFairness):
    """
    Class for evaluating fairness based on maximum gender performance in question-answering tasks using Language Model.

    Attributes:
        alias_name (List[str]): Alias names for the evaluation method.
        supported_tasks (List[str]): Supported tasks for this evaluation method.

    Methods:
        transform(cls, test: str, data: List[Sample], params: Dict) -> List[MaxScoreSample]:
            Transforms data for evaluation.
        run(sample_list: List[MaxScoreSample], grouped_label, **kwargs) -> List[MaxScoreSample]:
            Runs the evaluation process.

    """

    alias_name = ["max_gender_llm_eval"]
    supported_tasks = ["question-answering"]
    eval_model = None

    max_score = TypedDict(
        "max_score",
        male=float,
        female=float,
        unknown=float,
    )

    TestConfig = TypedDict(
        "TestConfig",
        max_score=Union[max_score, float],
    )

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> List[MaxScoreSample]:
        """
        Transforms the data for evaluation.

        Args:
            test (str): The test alias name.
            data (List[Sample]): The data to be transformed.
            params (Dict): Parameters for transformation.

        Returns:
            List[MaxScoreSample]: The transformed data samples based on the maximum score.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        from ..langtest import EVAL_MODEL
        from ..langtest import HARNESS_CONFIG as harness_config

        model = params.get("model", None)
        hub = params.get("hub", None)
        if model and hub:
            from ..tasks import TaskManager

            load_eval_model = TaskManager("question-answering")
            cls.eval_model = load_eval_model.model(
                model, hub, **harness_config.get("model_parameters", {})
            )
        else:
            cls.eval_model = EVAL_MODEL

        if isinstance(params["max_score"], dict):
            max_scores = params["max_score"]
        elif isinstance(params["max_score"], float):
            max_scores = {
                "male": params["max_score"],
                "female": params["max_score"],
                "unknown": params["max_score"],
            }

        samples = []
        for key, val in max_scores.items():
            sample = MaxScoreSample(
                original=None,
                category="fairness",
                test_type=test,
                test_case=key,
                expected_results=MaxScoreOutput(max_score=val),
            )

            samples.append(sample)
        return samples

    @staticmethod
    async def run(
        sample_list: List[MaxScoreSample], grouped_label, **kwargs
    ) -> List[MaxScoreSample]:
        """
        Runs the evaluation process using Language Model.

        Args:
            sample_list (List[MaxScoreSample]): The input data samples.
            grouped_label: A dictionary containing grouped labels where each key corresponds to a test case
                and the value is a tuple containing true labels and predicted labels.
            **kwargs: Additional keyword arguments.

        Returns:
            List[MaxScoreSample]: The evaluated data samples.
        """
        grouped_data = kwargs.get("grouped_data")
        from ..utils.custom_types.helpers import is_pass_llm_eval

        eval_model = MaxGenderLLMEval.eval_model
        progress = kwargs.get("progress_bar", False)

        def eval():
            results = []
            for true_list, pred, sample in zip(data[0], data[1], X_test):
                result = is_pass_llm_eval(
                    eval_model=eval_model,
                    dataset_name=sample.dataset_name,
                    original_question=sample.original_question,
                    answer="\n".join(map(str, true_list)),
                    perturbed_question=sample.original_question,
                    prediction=pred,
                )
                if result:
                    results.append(1)
                else:
                    results.append(0)
            total_samples = len(results)
            passed_samples = sum(results)
            accuracy = passed_samples / max(total_samples, 1)
            return accuracy

        for sample in sample_list:
            data = grouped_label[sample.test_case]
            X_test = grouped_data[sample.test_case]
            if len(data[0]) > 0:
                eval_score = eval()
            else:
                eval_score = 1

            sample.actual_results = MaxScoreOutput(max_score=eval_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list
