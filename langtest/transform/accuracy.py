import asyncio
from collections import defaultdict
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, DefaultDict, Dict, List, Type, TypedDict, Union

from langtest.modelhandler.modelhandler import ModelAPI
from langtest.transform.base import ITests
from langtest.utils.custom_types import (
    MinScoreOutput,
    MinScoreSample,
    NERSample,
    SequenceClassificationSample,
    Sample,
)
from langtest.utils.custom_types.helpers import default_user_prompt
from langtest.errors import Errors
from langtest.utils.custom_types.sample import DegradationSample, QASample
from langtest.utils.util_metrics import (
    calculate_f1_score,
    calculate_f1_score_multi_label,
    classification_report,
    classification_report_multi_label,
    combine_labels,
)


class AccuracyTestFactory(ITests):
    """
    A class for performing accuracy tests on a given dataset.
    """

    alias_name = "accuracy"
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
        Runs the accuracy test and returns the resulting `Sample` objects.

        Returns:
            List[Sample]:
                A list of `Sample` objects representing the resulting dataset after running the accuracy test.
        """
        all_samples = []

        if self._data_handler[0].expected_results is None:
            raise RuntimeError(Errors.E052(var="accuracy"))

        for test_name, params in self.tests.items():
            data_handler_copy = [x.model_copy() for x in self._data_handler]

            if data_handler_copy[0].task == "ner":
                y_true = pd.Series(data_handler_copy).apply(
                    lambda x: [y.entity for y in x.expected_results.predictions]
                )
                y_true = y_true.explode().apply(
                    lambda x: x.split("-")[-1] if isinstance(x, str) else x
                )
            elif data_handler_copy[0].task == "text-classification":
                y_true = (
                    pd.Series(data_handler_copy)
                    .apply(lambda x: [y.label for y in x.expected_results.predictions])
                    .explode()
                )
            elif (
                data_handler_copy[0].task == "question-answering"
                or data_handler_copy[0].task == "summarization"
            ):
                y_true = (
                    pd.Series(data_handler_copy)
                    .apply(lambda x: x.expected_results)
                    .explode()
                )

            y_true = y_true.dropna()
            transformed_samples = self.supported_tests[test_name].transform(
                test_name, y_true, params
            )

            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)

        return all_samples

    @staticmethod
    def available_tests() -> DefaultDict[str, Type["BaseAccuracy"]]:
        """
        Get a dictionary of all available tests, with their names as keys and their corresponding classes as values.

        Returns:
            dict: A dictionary of test names and classes.
        """

        return BaseAccuracy.test_types

    @classmethod
    def run(
        cls,
        sample_list: Dict[str, List[Sample]],
        model: ModelAPI,
        raw_data: List[Sample],
        **kwargs,
    ):
        """
        Runs the accuracy tests on the given model and dataset.

        Args:
            sample_list (Dict[str, List[Sample]]): A dictionary of test names and corresponding `Sample` objects.
            model (ModelAPI): The model to be tested.
            raw_data (List[Sample]): The raw dataset.

        """
        raw_data_copy = [x.model_copy() for x in raw_data]

        if isinstance(raw_data_copy[0], NERSample):

            def predict_ner(sample: NERSample):
                prediction = model.predict(sample.original)
                sample.actual_results = prediction
                return prediction

            X_test = pd.Series(raw_data_copy)
            X_test.apply(predict_ner)
            y_true = pd.Series(raw_data_copy).apply(
                lambda x: [y.entity for y in x.expected_results.predictions]
            )
            y_pred = pd.Series(raw_data_copy).apply(
                lambda x: [y.entity for y in x.actual_results.predictions]
            )
            valid_indices = y_true.apply(len) == y_pred.apply(len)
            y_true = y_true[valid_indices]
            y_pred = y_pred[valid_indices]
            y_true = y_true.explode()
            y_pred = y_pred.explode()
            y_pred = y_pred.apply(lambda x: x.split("-")[-1])
            y_true = y_true.apply(lambda x: x.split("-")[-1])

        elif isinstance(raw_data_copy[0], SequenceClassificationSample):
            is_mutli_label = raw_data_copy[0].expected_results.multi_label

            def predict_text_classification(sample):
                prediction = model.predict(sample.original)
                sample.actual_results = prediction
                return prediction

            X_test = pd.Series(raw_data_copy)
            X_test.apply(predict_text_classification)

            y_true = pd.Series(raw_data_copy).apply(
                lambda x: [y.label for y in x.expected_results.predictions]
            )
            y_pred = pd.Series(raw_data_copy).apply(
                lambda x: [y.label for y in x.actual_results.predictions]
            )

            if is_mutli_label:
                kwargs["is_multi_label"] = is_mutli_label

            else:
                y_true = y_true.apply(lambda x: x[0])
                y_pred = y_pred.apply(lambda x: x[0])

                y_true = y_true.explode()
                y_pred = y_pred.explode()

        elif raw_data_copy[0].task == "question-answering":
            from ..utils.custom_types.helpers import build_qa_input, build_qa_prompt

            if raw_data_copy[0].dataset_name is None:
                dataset_name = "default_question_answering_prompt"
            else:
                dataset_name = raw_data_copy[0].dataset_name.split("-")[0].lower()

            def predict_question_answering(sample):
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
                return prediction

            y_true = pd.Series(raw_data_copy).apply(lambda x: x.expected_results)
            X_test = pd.Series(raw_data_copy)

            y_pred = X_test.apply(predict_question_answering)

        elif raw_data_copy[0].task == "summarization":
            if raw_data_copy[0].dataset_name is None:
                dataset_name = "default_summarization_prompt"
            else:
                dataset_name = raw_data_copy[0].dataset_name.split("-")[0].lower()
            prompt_template = kwargs.get(
                "user_prompt", default_user_prompt.get(dataset_name, "")
            )

            def predict_summarization(sample):
                prediction = model(
                    text={"context": sample.original},
                    prompt={"template": prompt_template, "input_variables": ["context"]},
                ).strip()
                sample.actual_results = prediction
                return prediction

            y_true = pd.Series(raw_data_copy).apply(lambda x: x.expected_results)
            X_test = pd.Series(raw_data_copy)
            y_pred = X_test.apply(predict_summarization)

        if kwargs["is_default"]:
            y_pred = y_pred.apply(
                lambda x: (
                    "1"
                    if x in ["pos", "LABEL_1", "POS"]
                    else ("0" if x in ["neg", "LABEL_0", "NEG"] else x)
                )
            )

        supported_tests = cls.available_tests()

        tasks = []

        from ..utils.custom_types.helpers import TestResultManager

        cls.model_result = TestResultManager().prepare_model_response(raw_data_copy)

        for test_name, samples in sample_list.items():
            tasks.append(
                supported_tests[test_name].async_run(
                    samples, y_true, y_pred, X_test=X_test, **kwargs
                )
            )
        return tasks


class BaseAccuracy(ABC):
    """Abstract base class for implementing accuracy measures.

    Attributes:
        alias_name (str): A name or list of names that identify the accuracy measure.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented accuracy measure.
    """

    test_types: DefaultDict[str, Type["BaseAccuracy"]] = defaultdict(lambda: BaseAccuracy)

    alias_name = None
    supported_tasks = ["ner", "text-classification"]

    TestConfig = TypedDict(
        "TestConfig",
        min_score=Union[Dict[str, float], float],
    )

    @classmethod
    @abstractmethod
    def transform(y_true: List[Any], params: Dict) -> List[MinScoreSample]:
        """Abstract method that implements the accuracy measure.

        Args:
            y_true (List[Any]): True values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: The transformed data based on the implemented accuracy measure.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    async def run(
        sample_list: List[MinScoreSample], y_true: List[Any], y_pred: List[Any], **kwargs
    ) -> List[MinScoreSample]:
        """Computes the accuracy score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values
        """
        raise NotImplementedError()

    @classmethod
    async def async_run(
        cls,
        sample_list: List[MinScoreSample],
        y_true: List[Any],
        y_pred: List[Any],
        **kwargs,
    ):
        """Creates a task to run the accuracy measure.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        created_task = asyncio.create_task(cls.run(sample_list, y_true, y_pred, **kwargs))
        return created_task

    def __init_subclass__(cls) -> None:
        """Registers subclasses of BaseAccuracy in the test_types dictionary."""
        alias = cls.alias_name if isinstance(cls.alias_name, list) else [cls.alias_name]
        for name in alias:
            BaseAccuracy.test_types[name] = cls


class MinPrecisionScore(BaseAccuracy):
    """Subclass of BaseAccuracy that implements the minimum precision score.

    Attributes:
        alias_name (str): The name for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = ["min_precision_score"]

    @classmethod
    def transform(
        cls, test: str, y_true: List[Any], params: Dict
    ) -> List[MinScoreSample]:
        """Computes the minimum precision score for the given data.

        Args:
            y_true (List[Any]): True values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: Precision test results.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        labels = set(y_true)  # .union(set(y_pred))

        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {label: params["min_score"] for label in labels}

        precision_samples = []
        for k in labels:
            if k not in min_scores.keys():
                continue
            sample = MinScoreSample(
                original="-",
                category="accuracy",
                test_type="min_precision_score",
                test_case=k,
                expected_results=MinScoreOutput(min_score=min_scores[k]),
            )
            precision_samples.append(sample)
        return precision_samples

    async def run(
        sample_list: List[MinScoreSample], y_true: List[Any], y_pred: List[Any], **kwargs
    ):
        """Computes the minimum precision score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values
        """
        progress = kwargs.get("progress_bar", False)
        is_multi_label = kwargs.get("is_multi_label", False)
        if is_multi_label:
            df_metrics = classification_report_multi_label(
                y_true, y_pred, zero_division=0
            )
        else:
            df_metrics = classification_report(y_true, y_pred, zero_division=0)
        df_metrics.pop("macro avg")

        for idx, sample in enumerate(sample_list):
            if progress:
                progress.update(1)
            if sample.test_case not in df_metrics:
                sample_list.pop(idx)

                continue
            precision = df_metrics.get(sample.test_case)
            sample.actual_results = MinScoreOutput(min_score=precision["precision"])
            sample.state = "done"

        return sample_list


class MinRecallScore(BaseAccuracy):
    """Subclass of BaseAccuracy that implements the minimum recall score.

    Attributes:
        alias_name (str): The name for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = ["min_recall_score"]

    @classmethod
    def transform(
        cls, test: str, y_true: List[Any], params: Dict
    ) -> List[MinScoreSample]:
        """Computes the minimum recall score for the given data.

        Args:
            y_true (List[Any]): True values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: minimum recall results.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        labels = set(y_true)  # .union(set(y_pred))

        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {label: params["min_score"] for label in labels}

        rec_samples = []
        for k in labels:
            if k not in min_scores.keys():
                continue
            sample = MinScoreSample(
                original="-",
                category="accuracy",
                test_type="min_recall_score",
                test_case=k,
                expected_results=MinScoreOutput(min_score=min_scores[k]),
            )
            rec_samples.append(sample)
        return rec_samples

    async def run(
        sample_list: List[MinScoreSample], y_true: List[Any], y_pred: List[Any], **kwargs
    ):
        """Computes the minimum recall score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        progress = kwargs.get("progress_bar", False)

        is_multi_label = kwargs.get("is_multi_label", False)
        if is_multi_label:
            df_metrics = classification_report_multi_label(
                y_true, y_pred, zero_division=0
            )
        else:
            df_metrics = classification_report(y_true, y_pred, zero_division=0)
        df_metrics.pop("macro avg")

        for idx, sample in enumerate(sample_list):
            if progress:
                progress.update(1)
            if sample.test_case not in df_metrics:
                sample_list.pop(idx)

                continue
            precision = df_metrics.get(sample.test_case)
            sample.actual_results = MinScoreOutput(min_score=precision["recall"])
            sample.state = "done"

        return sample_list


class MinF1Score(BaseAccuracy):
    """Subclass of BaseAccuracy that implements the minimum F1 score.

    Attributes:
        alias_name (str): The name for config.

    """

    alias_name = ["min_f1_score"]

    @classmethod
    def transform(
        cls, test: str, y_true: List[Any], params: Dict
    ) -> List[MinScoreSample]:
        """Computes the minimum F1 score for the given data.

        Args:
            y_true (List[Any]): True values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: F1 score test results.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        labels = set(y_true)

        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {label: params["min_score"] for label in labels}

        f1_samples = []
        for k in labels:
            if k not in min_scores.keys():
                continue
            sample = MinScoreSample(
                original="-",
                category="accuracy",
                test_type="min_f1_score",
                test_case=k,
                expected_results=MinScoreOutput(min_score=min_scores[k]),
            )
            f1_samples.append(sample)
        return f1_samples

    async def run(
        sample_list: List[MinScoreSample], y_true: List[Any], y_pred: List[Any], **kwargs
    ):
        """Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        progress = kwargs.get("progress_bar", False)
        is_multi_label = kwargs.get("is_multi_label", False)
        if is_multi_label:
            df_metrics = classification_report_multi_label(
                y_true, y_pred, zero_division=0
            )
        else:
            df_metrics = classification_report(y_true, y_pred, zero_division=0)
        df_metrics.pop("macro avg")

        for idx, sample in enumerate(sample_list):
            if progress:
                progress.update(1)

            if sample.test_case not in df_metrics:
                sample_list.pop(idx)
                continue
            f1_scores = df_metrics.get(sample.test_case)
            sample.actual_results = MinScoreOutput(min_score=f1_scores["f1-score"])
            sample.state = "done"

        return sample_list


class MinMicroF1Score(BaseAccuracy):
    """Subclass of BaseAccuracy that implements the minimum micro f1 score.

    Attributes:
        alias_name (str): The name for config.
    """

    alias_name = ["min_micro_f1_score"]

    @classmethod
    def transform(
        cls, test: str, y_true: List[Any], params: Dict
    ) -> List[MinScoreSample]:
        """Computes the minimum micro F1 score for the given data.

        Args:
            y_true (List[Any]): True values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: The transformed data based on the minimum micro F1 score.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        min_score = params["min_score"]

        sample = MinScoreSample(
            original="-",
            category="accuracy",
            test_type="min_micro_f1_score",
            test_case="micro",
            expected_results=MinScoreOutput(min_score=min_score),
        )

        return [sample]

    async def run(
        sample_list: List[MinScoreSample], y_true: List[Any], y_pred: List[Any], **kwargs
    ):
        """Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        progress = kwargs.get("progress_bar", False)
        is_multi_label = kwargs.get("is_multi_label", False)

        if is_multi_label:
            f1 = calculate_f1_score_multi_label(
                y_true, y_pred, average="micro", zero_division=0
            )
        else:
            f1 = calculate_f1_score(y_true, y_pred, average="micro", zero_division=0)

        for sample in sample_list:
            sample.actual_results = MinScoreOutput(min_score=f1)
            sample.state = "done"
            if progress:
                progress.update(1)

        return sample_list


class MinMacroF1Score(BaseAccuracy):
    """Subclass of BaseAccuracy that implements the minimum macro score.

    Attributes:
        alias_name (str): The name for config.

    Methods:
        transform(y_true, params) -> Any: Creates accuracy test results.
    """

    alias_name = ["min_macro_f1_score"]

    @classmethod
    def transform(
        cls, test: str, y_true: List[Any], params: Dict
    ) -> List[MinScoreSample]:
        """Computes the minimum macro F1 score for the given data.

        Args:
            y_true (List[Any]): True values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: The transformed data based on the minimum macro F1 score.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"
        min_score = params["min_score"]

        sample = MinScoreSample(
            original="-",
            category="accuracy",
            test_type="min_macro_f1_score",
            test_case="macro",
            expected_results=MinScoreOutput(min_score=min_score),
        )

        return [sample]

    async def run(
        sample_list: List[MinScoreSample], y_true: List[Any], y_pred: List[Any], **kwargs
    ):
        """Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        progress = kwargs.get("progress_bar", False)

        is_multi_label = kwargs.get("is_multi_label", False)

        if is_multi_label:
            f1 = calculate_f1_score_multi_label(
                y_true, y_pred, average="macro", zero_division=0
            )
        else:
            f1 = calculate_f1_score(y_true, y_pred, average="macro", zero_division=0)

        for sample in sample_list:
            sample.actual_results = MinScoreOutput(min_score=f1)
            sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list


class MinWeightedF1Score(BaseAccuracy):
    """Subclass of BaseAccuracy that implements the minimum weighted f1 score.

    Attributes:
        alias_name (str): The name for config.

    Methods:
        transform(y_true, params) -> Any: Creates accuracy test results.
    """

    alias_name = ["min_weighted_f1_score"]

    @classmethod
    def transform(
        cls, test: str, y_true: List[Any], params: Dict
    ) -> List[MinScoreSample]:
        """Computes the minimum weighted F1 score for the given data.

        Args:
            y_true (List[Any]): True values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: The transformed data based on the minimum F1 score.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"
        min_score = params["min_score"]

        sample = MinScoreSample(
            original="-",
            category="accuracy",
            test_type="min_weighted_f1_score",
            test_case="weighted",
            expected_results=MinScoreOutput(min_score=min_score),
        )

        return [sample]

    async def run(
        sample_list: List[MinScoreSample], y_true: List[Any], y_pred: List[Any], **kwargs
    ):
        """Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        progress = kwargs.get("progress_bar", False)
        is_multi_label = kwargs.get("is_multi_label", False)

        if is_multi_label:
            f1 = calculate_f1_score_multi_label(
                y_true, y_pred, average="weighted", zero_division=0
            )
        else:
            f1 = calculate_f1_score(y_true, y_pred, average="weighted", zero_division=0)

        for sample in sample_list:
            sample.actual_results = MinScoreOutput(min_score=f1)
            sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list


class MinEMcore(BaseAccuracy):
    """Subclass of BaseAccuracy that implements the minimum precision score.

    Attributes:
        alias_name (str): The name for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = ["min_exact_match_score"]
    supported_tasks = ["question-answering", "summarization"]

    @classmethod
    def transform(
        cls, test: str, y_true: List[Any], params: Dict
    ) -> List[MinScoreSample]:
        """Computes the minimum F1 score for the given data.

        Args:
            y_true (List[Any]): True values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: The transformed data based on the minimum F1 score.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"
        min_score = params["min_score"]

        sample = MinScoreSample(
            category="accuracy",
            test_type="min_macro_f1_score",
            expected_results=MinScoreOutput(min_score=min_score),
        )

        return [sample]

    @staticmethod
    async def run(
        sample_list: List[MinScoreSample], y_true: List[Any], y_pred: List[Any], **kwargs
    ):
        """Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        import evaluate

        progress = kwargs.get("progress_bar", False)

        em = evaluate.load("exact_match")
        y_true = [x[0] for x in y_true]
        result = em.compute(references=y_true, predictions=y_pred)["exact_match"]
        for sample in sample_list:
            sample.actual_results = MinScoreOutput(min_score=result)
            sample.state = "done"
            if progress:
                progress.update(1)

        return sample_list


class MinBLEUcore(BaseAccuracy):
    """Subclass of BaseAccuracy that implements the minimum precision score.

    Attributes:
        alias_name (str): The name for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = ["min_bleu_score"]
    supported_tasks = ["question-answering", "summarization"]

    @classmethod
    def transform(
        cls, test: str, y_true: List[Any], params: Dict
    ) -> List[MinScoreSample]:
        """Computes the minimum F1 score for the given data.

        Args:
            y_true (List[Any]): True values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: The transformed data based on the minimum F1 score.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"
        min_score = params["min_score"]

        sample = MinScoreSample(
            category="accuracy",
            test_type="min_bleu_score",
            expected_results=MinScoreOutput(min_score=min_score),
        )

        return [sample]

    @staticmethod
    async def run(
        sample_list: List[MinScoreSample], y_true: List[Any], y_pred: List[Any], **kwargs
    ):
        """Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        try:
            progress = kwargs.get("progress_bar", False)
            import evaluate

            em = evaluate.load("bleu")
            result = em.compute(references=y_true, predictions=y_pred)
        except Exception as e:
            print(f"Error in BLEU evaluation: {e}. Setting BLEU score to 0")
            result = {"bleu": 0}

        y_true = [[f"The answer is {y}" for y in x] for x in y_true]
        y_pred = [f"The answer is {x}" for x in y_pred]

        for sample in sample_list:
            sample.actual_results = MinScoreOutput(min_score=result["bleu"])
            sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list


class MinROUGEcore(BaseAccuracy):
    """Subclass of BaseAccuracy that implements the minimum precision score.

    Attributes:
        alias_name (str): The name for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = [
        "min_rouge1_score",
        "min_rouge2_score",
        "min_rougeL_score",
        "min_rougeLsum_score",
    ]
    supported_tasks = ["question-answering", "summarization"]

    @classmethod
    def transform(
        cls, test: str, y_true: List[Any], params: Dict
    ) -> List[MinScoreSample]:
        """Computes the minimum F1 score for the given data.

        Args:
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: The transformed data based on the minimum F1 score.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"
        min_score = params["min_score"]

        sample = MinScoreSample(
            category="accuracy",
            test_type=test,
            expected_results=MinScoreOutput(min_score=min_score),
        )

        return [sample]

    @staticmethod
    async def run(
        sample_list: List[MinScoreSample], y_true: List[Any], y_pred: List[Any], **kwargs
    ):
        """Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        progress = kwargs.get("progress_bar", False)
        import evaluate

        em = evaluate.load("rouge")
        result = em.compute(references=y_true, predictions=y_pred)
        for sample in sample_list:
            sample.actual_results = MinScoreOutput(
                min_score=result[sample.test_type.split("_")[1]]
            )
            sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list


class LLMEval(BaseAccuracy):
    """
    Evaluation class for Language Model performance on question-answering tasks using the Language Model Metric (LLM).

    Attributes:
        alias_name (List[str]): Alias names for the evaluation class, should include "llm_eval".
        supported_tasks (List[str]): Supported tasks for evaluation, includes "question-answering".

    Methods:
        transform(cls, test: str, y_true: List[Any], params: Dict) -> List[MinScoreSample]:
            Transforms evaluation parameters and initializes the evaluation model.

        run(cls, sample_list: List[MinScoreSample], *args, **kwargs) -> List[MinScoreSample]:
            Runs the evaluation on a list of samples using the Language Model Metric (LLM).

    """

    alias_name = ["llm_eval"]

    supported_tasks = ["question-answering"]

    eval_model = None

    TestConfig = TypedDict(
        "TestConfig",
        model=str,
        hub=str,
        min_score=float,
    )

    @classmethod
    def transform(
        cls, test: str, y_true: List[Any], params: Dict
    ) -> List[MinScoreSample]:
        """
        Transforms evaluation parameters and initializes the evaluation model.

        Args:
            test (str): The alias name for the evaluation class.
            y_true (List[Any]): List of true labels (not used in this method).
            params (Dict): Additional parameters for evaluation, including 'model', 'hub', and 'min_score'.

        Returns:
            List[MinScoreSample]: List containing a MinScoreSample instance with evaluation information.

        Raises:
            AssertionError: If the 'test' parameter is not in the alias_name list.

        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        from ..langtest import EVAL_MODEL
        from ..langtest import HARNESS_CONFIG as harness_config

        model = params.get("model", None)
        hub = params.get("hub", None)

        # get model parameters from tests config or harness config if not provided
        # if not provided, default to the default model parameters
        parameters = params.get(
            "model_parameters", harness_config.get("model_parameters", {})
        )
        if model and hub:
            from ..tasks import TaskManager

            load_eval_model = TaskManager("question-answering")
            cls.eval_model = load_eval_model.model(model, hub, **parameters)

        else:
            cls.eval_model = EVAL_MODEL

        min_score = params["min_score"]

        sample = MinScoreSample(
            category="accuracy",
            test_type=test,
            expected_results=MinScoreOutput(min_score=min_score),
        )

        return [sample]

    @staticmethod
    async def run(
        sample_list: List[MinScoreSample], y_true: List[Any], y_pred: List[Any], **kwargs
    ):
        """
        Runs the evaluation on a list of samples using the Language Model Metric (LLM).

        Args:
            sample_list (List[MinScoreSample]): List of MinScoreSample instances containing evaluation information.
            y_true (List[Any]): List of true values for the model's predictions.
            y_pred (List[Any]): List of predicted values by the model.
            X_test (Optional): Additional keyword argument representing the test data.
            progress_bar (Optional): Additional keyword argument indicating whether to display a progress bar.
            **kwargs: Additional keyword arguments.

        Returns:
            List[MinScoreSample]: List containing updated MinScoreSample instances after evaluation.

        """
        X_test = kwargs.get("X_test")

        progress = kwargs.get("progress_bar", False)
        from ..utils.custom_types.helpers import is_pass_llm_eval

        eval_model = LLMEval.eval_model

        if not eval_model:
            from ..langtest import EVAL_MODEL

            eval_model = EVAL_MODEL

        def eval():
            results = []
            for true_list, pred, sample in zip(y_true, y_pred, X_test):
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
            sample.actual_results = MinScoreOutput(min_score=eval())
            sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list


class DegradationAnalysis(BaseAccuracy):
    """
    Evaluation class for model performance degradation analysis.

    Attributes:
        alias_name (List[str]): Alias names for the evaluation class, should
            include "degradation_analysis".
        supported_tasks (List[str]): Supported tasks for evaluation,
    Methods:
    """

    alias_name = ["degradation_analysis"]

    supported_tasks = ["ner", "text-classification", "question-answering"]

    result_data = defaultdict(dict)

    @classmethod
    def transform(cls, test: str, y_true: List[Any], params: Dict):
        # reset the result data
        DegradationAnalysis.result_data.clear()

        return [
            DegradationSample(
                category="accuracy",
                test_type="degradation_analysis",
            )
        ]

    @staticmethod
    async def run(
        sample_list: List[DegradationSample],
        y_true: List[Any],
        y_pred: List[Any],
        **kwargs,
    ):
        test_cases: Dict[str, Dict[str, List[Sample]]] = kwargs.get("test_cases", [])
        X_test = kwargs.get("X_test", [])

        if len(X_test) and isinstance(X_test[0], QASample):
            X_test = pd.DataFrame(
                {
                    "original_content": [
                        sample.original_context if sample.original_context else ""
                        for sample in X_test
                    ],
                    "original_question": [sample.original_question for sample in X_test],
                    "expected_results": [sample.expected_results for sample in X_test],
                }
            )
            X_test["index"] = (
                X_test["original_content"] + "\n" + X_test["original_question"]
            )
            X_test.set_index("index", inplace=True)

        elif isinstance(X_test, pd.Series) or isinstance(X_test, list):
            X_test = pd.DataFrame(
                {
                    "index": [sample.original for sample in X_test],
                    "expected_results": [sample.expected_results for sample in X_test],
                }
            )

            X_test.set_index("index", inplace=True)

        else:
            raise ValueError("X_test should be either a pd.Series or a list of samples.")

        # ground_truth = X_test["expected_results"].to_list()

        # # if ground_truth is having None values, raise an error
        if X_test["expected_results"].isnull().any():
            raise ValueError("Ground truth values cannot be None.")

        progress = kwargs.get("progress_bar", False)

        for category, data in test_cases.items():
            if category not in ["robustness", "bias"]:
                continue
            for test_type, samples in data.items():
                if len(samples) and isinstance(samples[0], QASample):
                    accuracy_score1, accuracy_score2 = DegradationAnalysis.qa_evaluation(
                        samples, X_test
                    )

                else:
                    ground_truth = X_test[
                        X_test.index.isin([i.original for i in samples])
                    ]["expected_results"].to_list()

                    expected_results = [x.expected_results for x in samples]
                    actual_results = [x.actual_results for x in samples]

                    accuracy_score1 = calculate_f1_score(
                        *DegradationAnalysis.preprocess(ground_truth, expected_results)
                    )
                    accuracy_score2 = calculate_f1_score(
                        *DegradationAnalysis.preprocess(ground_truth, actual_results)
                    )

                degradation = accuracy_score2 - accuracy_score1

                DegradationAnalysis.result_data[category][test_type] = {
                    "before": accuracy_score1,
                    "after": accuracy_score2,
                    "difference": degradation,
                }

        if len(sample_list) == 1:
            for sample in sample_list:
                sample.f1_scores = DegradationAnalysis.result_data
                sample.state = "done"
                if progress:
                    progress.update(1)
        else:
            sample_list = [
                DegradationSample(
                    category="accuracy",
                    test_type="degradation_analysis",
                    f1_scores=DegradationAnalysis.result_data,
                    state="done",
                )
            ]

        return sample_list

    @staticmethod
    def preprocess(y_true: Union[list, pd.Series], y_pred: Union[list, pd.Series]):
        """
        Preprocesses the input data for the degradation analysis.

        Args:

            y_true (List): The true labels.
            y_pred (List): The predicted labels.

        Returns:

            y_true, y_pred (Tuple[pd.Series, pd.Series]):
            The preprocessed true and predicted labels.
        """

        if isinstance(y_true, list):
            y_true = pd.Series(y_true).apply(lambda x: [y.entity for y in x])
        else:
            y_true = pd.Series(y_true).apply(lambda x: [y.entity for y in x.predictions])

        y_pred = pd.Series(y_pred).apply(lambda x: [y.entity for y in x.predictions])

        # remove the B- and I- prefixes from the entity tags and merge it
        y_pred = y_pred.apply(lambda x: combine_labels(x))
        y_true = y_true.apply(lambda x: combine_labels(x))

        valid_indices = y_true.apply(len) == y_pred.apply(len)
        y_true = y_true[valid_indices]
        y_pred = y_pred[valid_indices]

        y_true = y_true.explode()
        y_pred = y_pred.explode()
        y_pred = y_pred.apply(lambda x: x.split("-")[-1])
        y_true = y_true.apply(lambda x: x.split("-")[-1])

        return y_true, y_pred

    @staticmethod
    def qa_evaluation(samples: List[QASample], X_test: pd.DataFrame):
        """
        Evaluates the model performance on question-answering tasks.

        Args:

            samples (List[QASample]): The list of QASample instances.
            X_test (pd.DataFrame): The test data.

        Returns:

            Tuple[float, float]: The accuracy scores for the original and perturbed samples.

        """

        results = {
            "original": [],
            "perturbed": [],
            "total": len(samples),
        }
        for sample in samples:
            if sample.original_context is None:
                context = ""
            else:
                context = sample.original_context
            index = context + "\n" + sample.original_question

            g_values = X_test[X_test.index == index]["expected_results"].values
            ground_truth = g_values[0] if len(g_values) else None

            # if ground_truth is None, skip the sample and continue to the next sample
            if ground_truth is None:
                results["total"] -= 1
                continue

            expected_results = sample.expected_results
            actual_results = sample.actual_results

            original = sample.original_context + "\n" + sample.original_question
            perturbed = sample.perturbed_context + "\n" + sample.perturbed_question

            from ..utils.custom_types.helpers import is_pass_llm_eval
            from ..langtest import EVAL_MODEL

            o_result = is_pass_llm_eval(
                eval_model=EVAL_MODEL,
                dataset_name=sample.dataset_name,
                original_question=original,
                answer="\n".join(map(str, ground_truth)),
                perturbed_question=perturbed,
                prediction=expected_results,
            )

            p_result = is_pass_llm_eval(
                eval_model=EVAL_MODEL,
                dataset_name=sample.dataset_name,
                original_question=original,
                answer="\n".join(map(str, ground_truth)),
                perturbed_question=perturbed,
                prediction=actual_results,
            )

            results["original"].append(int(o_result) if o_result else 0)
            results["perturbed"].append(int(p_result) if p_result else 0)

        accuracy_score1 = sum(results["original"]) / max(results["total"], 1)
        accuracy_score2 = sum(results["perturbed"]) / max(results["total"], 1)

        return accuracy_score1, accuracy_score2

    @staticmethod
    def show_results():
        import pandas as pd
        import matplotlib.pyplot as plt

        data = DegradationAnalysis.result_data
        if not data:
            raise ValueError("No data found for degradation analysis.")

        for category, tests in data.items():
            df = pd.DataFrame(tests).T

            fig, ax = plt.subplots(figsize=(12, 6))

            y_labels = df.index
            y_pos = range(len(y_labels))

            for i, label in enumerate(y_labels):
                # Before robustness bar (back layer)
                ax.broken_barh(
                    [(0, df["before"][i])],
                    (i - 0.4, 0.8),
                    color="#d3d3d3",
                    zorder=0,
                    label="Before" if i == 0 else "",
                )
                # After robustness bar (front layer)
                ax.broken_barh(
                    [(0, df["after"][i])],
                    (i - 0.2, 0.4),
                    color="#1f77b4",
                    label="After" if i == 0 else "",
                    zorder=1,
                )

                # Adjust label position dynamically
                offset = 0.03 if abs(df["before"][i] - df["after"][i]) < 0.05 else 0.01

                # Text for "after" bar
                ax.text(
                    df["after"][i] + 0.01,
                    i - 0.1,
                    f"{df['after'][i]:.2f}",
                    va="center",
                    ha="left",
                    color="#1f77b4",
                    zorder=2,
                )
                # Text for "before" bar
                ax.text(
                    df["before"][i] + offset,
                    i + 0.1,
                    f"{df['before'][i]:.2f}",
                    va="center",
                    ha="left",
                    color="black",
                    zorder=2,
                )

            ax.set_xlim(0, 1)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(y_labels)
            ax.set_xlabel(f"Accuracy Score Over {category} Tests")
            ax.set_title(f"Comparison of Accuracy Before and After {category} Tests")
            ax.legend()

            plt.tight_layout()
            plt.show()
