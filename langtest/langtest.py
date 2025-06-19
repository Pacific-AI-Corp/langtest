import json
import logging
import os
import pickle
import importlib
import subprocess
from collections import defaultdict
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml
import random

from pkg_resources import resource_filename

from langtest.types import DatasetConfig, HarnessConfig, ModelConfig

from .tasks import TaskManager
from .augmentation import AugmentRobustness, TemplaticAugment
from .datahandler.datasource import DataFactory
from .modelhandler import LANGCHAIN_HUBS
from .transform import TestFactory
from .utils import report_utils as report, config_utils

from .transform.utils import RepresentationOperation
from langtest.utils.benchmark_utils import Leaderboard, Summary
from langtest.utils.lib_manager import try_import_lib
from langtest.utils.custom_types.helpers import TestResultManager
from langtest.utils.checkpoints import divide_into_batches, CheckpointManager
from langtest.prompts import PromptManager
from .errors import Warnings, Errors

EVAL_MODEL = None
GLOBAL_HUB = None
HARNESS_CONFIG = None
GLOBAL_DATASET_CONFIG = None


class Harness:
    """Harness is a testing class for NLP models.

    Harness class evaluates the performance of a given NLP model. Given test data is
    used to test the model. A report is generated with test results.
    """

    SUPPORTED_TASKS = [
        "ner",
        "text-classification",
        "question-answering",
        "summarization",
        "fill-mask",
        "text-generation",
        "translation",
    ]
    SUPPORTED_HUBS = [
        "spacy",
        "huggingface",
        "johnsnowlabs",
        "openai",
        "cohere",
        "custom",
        "ai21",
    ] + list(LANGCHAIN_HUBS.keys())
    DEFAULTS_DATASET = {
        ("ner", "dslim/bert-base-NER", "huggingface"): "conll/sample.conll",
        ("ner", "en_core_web_sm", "spacy"): "conll/sample.conll",
        ("ner", "ner.dl", "johnsnowlabs"): "conll/sample.conll",
        ("ner", "ner_dl_bert", "johnsnowlabs"): "conll/sample.conll",
        (
            "text-classification",
            "lvwerra/distilbert-imdb",
            "huggingface",
        ): "imdb/sample.csv",
        ("text-classification", "textcat_imdb", "spacy"): "imdb/sample.csv",
        (
            "text-classification",
            "en.sentiment.imdb.glove",
            "johnsnowlabs",
        ): "imdb/sample.csv",
    }
    SUPPORTED_HUBS_HF_DATASET_NER = ["johnsnowlabs", "huggingface", "spacy"]
    SUPPORTED_HUBS_HF_DATASET_CLASSIFICATION = ["johnsnowlabs", "huggingface", "spacy"]
    SUPPORTED_HUBS_HF_DATASET_LLM = [
        "openai",
        "cohere",
        "ai21",
        "huggingface-inference-api",
    ]

    DEFAULTS_CONFIG = config_utils.DEFAULTS_CONFIG

    def __init__(
        self,
        task: Union[str, dict],
        model: Optional[Union[List[ModelConfig], ModelConfig]] = None,
        data: Optional[Union[List[DatasetConfig], DatasetConfig]] = None,
        config: Optional[Union[HarnessConfig, str]] = None,
        benchmarking: dict = None,
    ):
        """Initialize the Harness object.

        Args:
            task (str, optional): Task for which the model is to be evaluated.
            model (ModelConfig, list | dict, optional): Specifies the model to be evaluated.
                If provided as a list, each element should be a dictionary with 'model' and 'hub' keys.
                If provided as a dictionary, it must contain 'model' and 'hub' keys when specifying a path.
            data (DatasetConfig, dict, optional): The data to be used for evaluation.
            config (str | HarnessConfig , optional): Configuration for the tests to be performed.

        Raises:
            ValueError: Invalid arguments.
        """
        super().__init__()

        self.is_default = False
        self.__data_dict = data
        self.__is_multi_model = False
        self.__model_info = model
        self.__benchmarking = benchmarking

        # check the list of strings in the data
        if isinstance(data, list) and all(isinstance(i, str) for i in data):
            temp_data = []
            for dataset in data:
                if isinstance(task, dict):
                    temp_task = task["category"]
                else:
                    temp_task = task
                temp_data.append(
                    config_utils.BenchmarkDatasets.get_dataset_dict(
                        dataset_name=dataset, task=temp_task
                    )
                )

            data = temp_data
            self.__data_dict = data

        # reset classes to default state
        self.__reset_defaults()

        # set dataset config as global
        global GLOBAL_DATASET_CONFIG
        GLOBAL_DATASET_CONFIG = data

        # loading model and hub
        if isinstance(model, list):
            self.__is_multi_model = True
            for item in model:
                if not isinstance(item, dict):
                    raise ValueError(Errors.E000())
                if "model" not in item or "hub" not in item:
                    raise ValueError(Errors.E001())
        elif isinstance(model, dict):
            if "model" not in model or "hub" not in model:
                raise ValueError(Errors.E002())

        else:
            raise ValueError(Errors.E003())

        if isinstance(model, dict):
            additional_info = (
                {k: v for k, v in model.items() if k not in {"hub", "model", "type"}}
                if model
                else {}
            )
            hub, model, model_type = model["hub"], model["model"], model.get("type")
            self.hub = hub
            self._actual_model = model
        else:
            additional_info = {}
            hub = None
            model_type = None

        # loading task

        self.task = TaskManager(task)

        # Loading default datasets
        if isinstance(data, list):
            self.data = self.__multi_datasets_loading(task, hub, model, data)
        else:
            self.data = self.__single_dataset_loading(task, hub, model, data)

        # config loading
        if config is not None:
            self._config = self.configure(config)
        elif self.task.category in self.DEFAULTS_CONFIG:
            category = self.task.category
            if isinstance(self.DEFAULTS_CONFIG[category], dict):
                if hub in self.DEFAULTS_CONFIG[category]:
                    self._config = self.configure(self.DEFAULTS_CONFIG[category][hub])
                else:
                    self._config = self.configure(
                        self.DEFAULTS_CONFIG[category]["default"]
                    )
            elif isinstance(self.DEFAULTS_CONFIG[category], str):
                self._config = self.configure(self.DEFAULTS_CONFIG[category])
        elif self.task in self.DEFAULTS_CONFIG:
            if isinstance(self.DEFAULTS_CONFIG[self.task], dict):
                if hub in self.DEFAULTS_CONFIG[self.task]:
                    self._config = self.configure(self.DEFAULTS_CONFIG[self.task][hub])
                else:
                    self._config = self.configure(
                        self.DEFAULTS_CONFIG[self.task]["default"]
                    )
            elif isinstance(self.DEFAULTS_CONFIG[self.task], str):
                self._config = self.configure(self.DEFAULTS_CONFIG[self.task])
        else:
            logging.info(Warnings.W001())
            self._config = self.configure(
                resource_filename("langtest", "data/config.yml")
            )

        # prompt config
        self.__prompt_config = self._config.get("prompt_config", None)
        if self.__prompt_config:
            self.prompt_manager = PromptManager.from_prompt_configs(self.__prompt_config)

        # model section
        if isinstance(model, list):
            model_dict = {}
            for i in model:
                model = i["model"]
                hub = i["hub"]

                # additional info for each model
                additional_info = (
                    {k: v for k, v in i.items() if k not in {"hub", "model", "type"}}
                    if i
                    else {}
                )

                # get config from model_parameters
                model_params = self._config.get("model_parameters", {})
                if isinstance(model_params, dict) and model in model_params:
                    # additional info for each model
                    additional_info = {**model_params[model], **additional_info}
                else:
                    # same additional info for all models
                    additional_info = {**additional_info, **model_params}

                model_dict[model] = self.task.model(
                    model, hub, model_type, **additional_info
                )

                self.model = model_dict

        else:
            additional_info = {
                **(additional_info or {}),
                **self._config.get("model_parameters", {}),
            }
            self.model = self.task.model(model, hub, model_type, **additional_info)
        # end model selection
        formatted_config = json.dumps(self._config, indent=1)
        print("Test Configuration : \n", formatted_config)

        global EVAL_MODEL, GLOBAL_HUB
        if not isinstance(model, list):
            EVAL_MODEL = self.model
            GLOBAL_HUB = hub

        self._testcases = None
        self.batches = None
        self._checkpoints = None
        self._generated_results = None
        self.accuracy_results = None
        self.min_pass_dict = None
        self.default_min_pass_dict = None
        self.df_report = None

    def __repr__(self) -> str:
        return ""

    def __str__(self) -> str:
        return object.__repr__(self)

    def configure(self, config: Union[HarnessConfig, dict, str]) -> HarnessConfig:
        """Configure the Harness with a given configuration.

        Args:
            config (str | dict): Configuration file path or dictionary
                for the tests to be performed.

        Returns:
            dict: Loaded configuration.
        """
        if type(config) == dict:
            self._config = config
        else:
            with open(config, "r", encoding="utf-8") as yml:
                self._config = yaml.safe_load(yml)
        self._config_copy = self._config

        global HARNESS_CONFIG
        HARNESS_CONFIG = self._config

        return self._config

    def generate(self, seed: int = None) -> "Harness":
        """Generate the testcases to be used when evaluating the model.

        The generated testcases are stored in `_testcases` attribute.
        """
        if seed:
            random.seed(seed)
        if self._config is None:
            raise RuntimeError(Errors.E005())
        if self._testcases is not None:
            raise RuntimeError(Errors.E006())

        self._testcases = []

        if isinstance(self.data, list) and not self.__is_multi_model:
            self._testcases = self.__single_dataset_generate(self.data)
        elif isinstance(self.data, list) and self.__is_multi_model:
            self._testcases = self.__multi_datasets_generate(self.data)
        elif isinstance(self.data, dict) and len(self.data) > 1:
            self._testcases = self.__multi_datasets_generate(self.data)
        else:
            self._testcases = self.__single_dataset_generate(self.data)

        return self

    def run(
        self,
        checkpoint: bool = False,
        batch_size=500,
        save_checkpoints_dir: str = "checkpoints",
    ) -> "Harness":
        """Run the tests on the model using the generated test cases.

        Args:
            checkpoint (bool): If True, enable checkpointing to save intermediate results.
            batch_size (int): Batch size for dividing test cases into batches.
            save_checkpoints_dir (str): Directory to save checkpoints and intermediate results.

        Returns:
            Harness: The updated Harness object with test results stored in `generated_results` attribute.

        Raises:
            RuntimeError: Raised if test cases are not provided (None).
        """
        if isinstance(self._testcases, list) and not self.__is_multi_model:
            self.is_multi_dataset = False
            self._generated_results = self.__single_dataset_run(
                self._testcases, self.data, checkpoint, save_checkpoints_dir, batch_size
            )
        elif isinstance(self.data, list) and self.__is_multi_model:
            self.is_multi_dataset = False
            self._generated_results = self.__single_dataset_run(
                self._testcases, self.data, checkpoint, save_checkpoints_dir, batch_size
            )
        else:
            self.is_multi_dataset = True
            self._generated_results = self.__multi_datasets_run(
                self._testcases, checkpoint, save_checkpoints_dir, batch_size
            )
        return self

    def model_response(self, category: str = None):
        """
        Retrieves the model response for a specific category.

        Args:
            category (str): The category for which the model response is requested. It should be one of the supported categories:
                "accuracy" or "fairness".

        Returns:
            pd.DataFrame: A DataFrame containing the model response data with columns including 'gender', 'original',
                'original_question', 'original_context', 'options', 'expected_results', and 'actual_results'.
                If the model response is empty or None, returns an empty DataFrame.

        Raises:
            ValueError: If the category is None or not one of the supported categories.
        """
        supported_category = ("accuracy", "fairness")

        if category is None:
            raise ValueError(Errors.E093())

        if category not in supported_category:
            raise ValueError(
                Errors.E094(category=category, supported_category=supported_category)
            )

        class_name = f"{category.title()}TestFactory"
        module = importlib.import_module("langtest.transform")
        testfactory_class = getattr(module, class_name)
        model_response = testfactory_class.model_result

        if model_response is None:
            logging.warning(Warnings.W021())

        else:
            data = [sample.model_copy() for sample in model_response]
            data_dict = [{key: value for key, value in x.__dict__.items()} for x in data]
            data_df = pd.DataFrame(data_dict)
            data_df = data_df.reset_index(drop=True)

            column_order = [
                "gender",
                "original",
                "original_question",
                "original_context",
                "options",
                "expected_results",
                "actual_results",
            ]

            # add the dataset_name column if the data is multi-dataset
            if self.is_multi_dataset:
                column_order.insert(0, "dataset_name")

            columns = [c for c in column_order if c in data_df.columns]
            data_df = data_df[columns]

            return data_df.fillna("-")

    @classmethod
    def load_checkpoints(cls, task, model, save_checkpoints_dir: str) -> "Harness":
        """Load checkpoints and other necessary data to recreate a Harness object.

        Args:
            task: The task for which the model was tested.
            model: The model or models used for testing.
            save_checkpoints_dir (str): Directory containing saved checkpoints and data.

        Returns:
            Harness: A Harness object reconstructed with loaded checkpoints and data.

        Raises:
            OSError: Raised if necessary files (config.yaml, data.pkl) are missing in the checkpoint directory.
        """
        if not os.path.isdir(save_checkpoints_dir):
            raise OSError(Errors.E092(directory=save_checkpoints_dir))

        for filename in ["config.yaml", "data.pkl"]:
            if not os.path.exists(os.path.join(save_checkpoints_dir, filename)):
                raise OSError(Errors.E017(filename=filename))

        with open(os.path.join(save_checkpoints_dir, "data.pkl"), "rb") as reader:
            data = pickle.load(reader)

        harness = Harness(
            task=task,
            model=model,
            data={"data_source": data},
            config=os.path.join(save_checkpoints_dir, "config.yaml"),
        )

        is_multi_dataset = isinstance(data, dict)

        if is_multi_dataset:
            harness._testcases = {}
            harness._checkpoints = {}
            harness.batches = {}
            for dataset_name, samples in data.items():
                dataset_checkpoint_dir = os.path.join(save_checkpoints_dir, dataset_name)
                checkpoint_manager = CheckpointManager(
                    checkpoint_folder=dataset_checkpoint_dir
                )
                harness._checkpoints[dataset_name] = checkpoint_manager.load_checkpoint()
                harness._testcases[dataset_name] = (
                    checkpoint_manager.load_remaining_batch()
                )
                harness.batches[dataset_name] = checkpoint_manager.load_batches()

        elif isinstance(model, dict):
            checkpoint_manager = CheckpointManager(checkpoint_folder=save_checkpoints_dir)
            harness._checkpoints = checkpoint_manager.load_checkpoint()
            harness._testcases = checkpoint_manager.load_remaining_batch()
            harness.batches = checkpoint_manager.load_batches()
        else:
            harness._testcases = {}
            harness._checkpoints = {}
            harness.batches = {}
            for model_name in model:
                model_checkpoint_dir = os.path.join(
                    save_checkpoints_dir, model_name["model"]
                )
                checkpoint_manager = CheckpointManager(
                    checkpoint_folder=model_checkpoint_dir
                )
                harness._checkpoints[model_name["model"]] = (
                    checkpoint_manager.load_checkpoint()
                )
                harness._testcases[model_name["model"]] = (
                    checkpoint_manager.load_remaining_batch()
                )
                harness.batches[model_name["model"]] = checkpoint_manager.load_batches()
        return harness

    def report(
        self,
        format: str = "dataframe",
        save_dir: str = None,
        mlflow_tracking: bool = False,
    ) -> pd.DataFrame:
        """Generate a report of the test results.

        Args:
            format (str): format in which to save the report
            save_dir (str): name of the directory to save the file
        Returns:
            pd.DataFrame:
                DataFrame containing the results of the tests.
        """

        # benchmarking true
        if self.__benchmarking:
            self.__tracking()

        if self._generated_results is None:
            raise RuntimeError(Errors.E011())

        # plot the degradation analysis results
        if (
            "accuracy" in self._config["tests"]
            and "degradation_analysis" in self._config["tests"]["accuracy"]
        ):
            from langtest.transform.accuracy import DegradationAnalysis

            DegradationAnalysis.show_results()

        if isinstance(self._config, dict):
            self.default_min_pass_dict = self._config["tests"]["defaults"].get(
                "min_pass_rate", 0.65
            )
            self.min_pass_dict = {
                j: k.get("min_pass_rate", self.default_min_pass_dict)
                for i, v in self._config["tests"].items()
                for j, k in v.items()
                if isinstance(k, dict) and "min_pass_rate" in k.keys()
            }

        summary = defaultdict(lambda: defaultdict(int))

        if self.task.category == "ideology":
            self.df_report = report.political_report(self._generated_results)
            return self.df_report

        elif self.is_multi_dataset and isinstance(self.model, dict):
            self.df_report = report.multi_dataset_multi_model_report(
                summary,
                self.min_pass_dict,
                self.default_min_pass_dict,
                self._generated_results,
                self.model,
            )

            report.save_format(format, save_dir, self.df_report)
            return self.df_report
        elif self.is_multi_dataset:
            self.df_report = report.multi_dataset_report(
                summary,
                self.min_pass_dict,
                self.default_min_pass_dict,
                self._generated_results,
                self._actual_model if isinstance(self._actual_model, str) else "",
            )

            report.save_format(format, save_dir, self.df_report)
            return self.df_report

        elif not isinstance(self._generated_results, dict):
            self.df_report = report.model_report(
                summary,
                self.min_pass_dict,
                self.default_min_pass_dict,
                self._generated_results,
            )

            if mlflow_tracking:
                experiment_name = (
                    self._actual_model
                    if isinstance(self._actual_model, str)
                    else self._actual_model.__class__.__module__
                )

                report.mlflow_report(experiment_name, self.task, self.df_report)

            report.save_format(format, save_dir, self.df_report)
            return self.df_report

        else:
            df_final_report = pd.DataFrame()
            for k in self.model.keys():
                summary = defaultdict(lambda: defaultdict(int))
                df_report = report.multi_model_report(
                    summary,
                    self.min_pass_dict,
                    self.default_min_pass_dict,
                    self._generated_results,
                    k,
                )
                if mlflow_tracking:
                    experiment_name = k
                    report.mlflow_report(
                        experiment_name, self.task, df_report, multi_model_comparison=True
                    )

                df_final_report = pd.concat([df_final_report, df_report])

            df_final_report["model_name"] = df_final_report["model_name"].astype(str)

            df_final_report["minimum_pass_rate"] = (
                df_final_report["minimum_pass_rate"].str.rstrip("%").astype("float")
                / 100.0
            )

            df_final_report["pass_rate"] = (
                df_final_report["pass_rate"].str.rstrip("%").astype("float") / 100.0
            )

            pivot_df = df_final_report.pivot_table(
                index="model_name",
                columns="test_type",
                values="pass_rate",
                aggfunc="mean",
            )

            styled_df = pivot_df.style.apply(
                report.color_cells, df_final_report=df_final_report
            )

            if format == "dataframe":
                return styled_df

            else:
                report.save_format(format, save_dir, styled_df)

    def generated_results(self) -> Optional[pd.DataFrame]:
        """Generates an overall report with every textcase and labelwise metrics.

        Returns:
            pd.DataFrame: Generated dataframe.
        """

        column_order = [
            "dataset_name",
            "model_name",
            "category",
            "test_type",
            "original_image",
            "original",
            "context",
            "prompt",
            "original_context",
            "original_question",
            "completion",
            "dialogue",
            "test_case",
            "perturbed_context",
            "perturbed_image",
            "perturbed_question",
            "sentence",
            "question",
            "patient_info_A",
            "patient_info_B",
            "case",
            "legal_claim",
            "legal_conclusion_A",
            "legal_conclusion_B",
            "correct_conlusion",
            "model_conclusion",
            "masked_text",
            "diagnosis",
            "treatment_plan_A",
            "treatment_plan_B",
            "mask1",
            "mask2",
            "mask1_score",
            "mask2_score",
            "sent_stereo",
            "sent_antistereo",
            "log_prob_stereo",
            "log_prob_antistereo",
            "diff_threshold",
            "options",
            "expected_result",
            "prompt_toxicity",
            "actual_result",
            "completion_toxicity",
            "hypothesis",
            "statements",
            "article_sentence",
            "correct_sentence",
            "incorrect_sentence",
            "ground_truth",
            "result",
            "swapped_result",
            "model_response",
            "feedback",
            "eval_score",
            "similarity_score",
            "original_result",
            "perturbed_result",
            "pass",
        ]

        if self._generated_results is None:
            logging.warning(Warnings.W000())
            return

        if isinstance(self._generated_results, dict) and isinstance(self.model, dict):
            generated_results_df = []
            for k, v in self._generated_results.items():
                if isinstance(v, dict):
                    model_generated_results_df = pd.DataFrame.from_dict(
                        [
                            {
                                "dataset_name": k,
                                "model_name": model_name,
                                **sample.to_dict(),
                            }
                            for model_name, samples in v.items()
                            if isinstance(samples, list)
                            for sample in samples
                        ]
                    )
                else:
                    model_generated_results_df = pd.DataFrame.from_dict(
                        [{"model_name": k, **x.to_dict()} for x in v]
                    )
                if (
                    "test_case" in model_generated_results_df.columns
                    and "original_question" in model_generated_results_df.columns
                ):
                    model_generated_results_df["original_question"].update(
                        model_generated_results_df.pop("test_case")
                    )
                # model_generated_results_df["model_name"] = k
                generated_results_df.append(model_generated_results_df)
            generated_results_df = pd.concat(generated_results_df).reset_index(drop=True)

        elif self.is_multi_dataset:
            generated_results_df = pd.DataFrame(
                [
                    {**x.to_dict(), "dataset_name": dataset_name}
                    for dataset_name, samples in self._generated_results.items()
                    for x in samples
                ]
            )
            generated_results_df = generated_results_df.reset_index(drop=True)
            if "prompt" in generated_results_df.columns:
                return generated_results_df.fillna("-")

            elif (
                "test_case" in generated_results_df.columns
                and "original_question" in generated_results_df.columns
            ):
                generated_results_df["original_question"].update(
                    generated_results_df.pop("test_case")
                )

            columns = [c for c in column_order if c in generated_results_df.columns]
            generated_results_df = generated_results_df[columns]

            return generated_results_df.fillna("-")
        else:
            generated_results_df = pd.DataFrame.from_dict(
                [x.to_dict() for x in self._generated_results]
            )
            if "dataset_name" in column_order:
                column_order.remove("dataset_name")

        columns = [c for c in column_order if c in generated_results_df.columns]
        generated_results_df = generated_results_df[columns]

        generated_results_df = generated_results_df[
            ~generated_results_df["test_type"].isin(["degradation_analysis", "amega"])
        ]

        return generated_results_df.fillna("-")

    def augment(
        self,
        training_data: dict,
        save_data_path: str,
        custom_proportions: Union[Dict, List] = None,
        export_mode: str = "add",
        templates: Optional[Union[str, List[str]]] = None,
        append_original: bool = False,
        generate_templates: bool = False,
        show_templates: bool = False,
    ) -> "Harness":
        """Augments the data in the input file located at `input_path` and saves the result to `output_path`.

        Args:
            training_data (dict): A dictionary containing the input data for augmentation.
            save_data_path (str): Path to save the augmented data.
            custom_proportions (Union[Dict, List]):
            export_mode (str, optional): Determines how the samples are modified or exported.
                                    - 'inplace': Modifies the list of samples in place.
                                    - 'add': Adds new samples to the input data.
                                    - 'transformed': Exports only the transformed data, excluding untransformed samples.
                                    Defaults to 'add'.
            templates (Optional[Union[str, List[str]]]):
            append_original (bool, optional): If set to True, appends the original data to the augmented data. Defaults to False.
            generate_templates (bool, optional): if set to True, generates sample templates from given ones.
            show_templates (bool, optional): if set to True, displays the used templates.

        Returns:
            Harness: The instance of the class calling this method.

        Raises:
            ValueError: If the `pass_rate` or `minimum_pass_rate` columns have an unexpected data type.

        Note:
            This method uses an instance of `AugmentRobustness` to perform the augmentation.

        """
        dtypes = list(
            map(
                str,
                self.df_report[["pass_rate", "minimum_pass_rate"]].dtypes.values.tolist(),
            )
        )
        if dtypes not in [["int64"] * 2, ["int32"] * 2]:
            self.df_report["pass_rate"] = (
                self.df_report["pass_rate"].str.replace("%", "").astype(int)
            )
            self.df_report["minimum_pass_rate"] = (
                self.df_report["minimum_pass_rate"].str.replace("%", "").astype(int)
            )

        # checking if the custom_proportions are valid
        if custom_proportions:
            vaild_test_types = set(
                custom_proportions.keys()
                if isinstance(custom_proportions, dict)
                else custom_proportions
            )
            report_test_types = set(self.df_report["test_type"].unique())
            vaild_test_types = set(
                custom_proportions.keys()
                if isinstance(custom_proportions, dict)
                else custom_proportions
            )
            report_test_types = set(self.df_report["test_type"].unique())

            if not (vaild_test_types.issubset(report_test_types)):
                raise ValueError(
                    Errors.E014(test_name=(vaild_test_types - report_test_types))
                )

        if templates:
            _ = TemplaticAugment(
                templates=templates,
                task=self.task,
                generate_templates=generate_templates,
                show_templates=show_templates,
            ).fix(
                training_data=training_data,
                output_path=save_data_path,
                append_original=append_original,
            )

        else:
            _ = AugmentRobustness(
                task=self.task,
                config=self._config,
                h_report=self.df_report,
                custom_proportions=custom_proportions,
            ).fix(
                training_data=training_data,
                output_path=save_data_path,
                export_mode=export_mode,
            )

        return self

    def testcases(self, additional_cols=False) -> pd.DataFrame:
        """Testcases after .generate() is called

        Returns:
            pd.DataFrame:
                testcases formatted into a pd.DataFrame
        """
        column_order = [
            "dataset_name",
            "model_name",
            "category",
            "test_type",
            "original_image",
            "original",
            "context",
            "original_context",
            "original_question",
            "test_case",
            "sentence",
            "patient_info_A",
            "patient_info_B",
            "dialogue",
            "mask1",
            "mask2",
            "sent_stereo",
            "sent_antistereo",
            "case",
            "legal_claim",
            "legal_conclusion_A",
            "legal_conclusion_B",
            "correct_conlusion",
            "masked_text",
            "diagnosis",
            "hypothesis",
            "statements",
            "article_sentence",
            "correct_sentence",
            "incorrect_sentence",
            "perturbed_context",
            "perturbed_image",
            "perturbed_question",
            "question",
            "ground_truth",
            "options",
            "expected_result",
        ]

        if additional_cols:
            column_order.extend(["transformations"])
        if isinstance(self._testcases, dict) and not self.is_multi_dataset:
            testcases_df = []
            for k, v in self._testcases.items():
                model_testcases_df = pd.DataFrame([x.to_dict() for x in v])
                if "prompt" in model_testcases_df.columns:
                    return model_testcases_df.fillna("-")

                elif (
                    "test_case" in model_testcases_df.columns
                    and "original_question" in model_testcases_df.columns
                ):
                    model_testcases_df["original_question"].update(
                        model_testcases_df.pop("test_case")
                    )

                model_testcases_df["model_name"] = k
                testcases_df.append(model_testcases_df)

            testcases_df = pd.concat(testcases_df).reset_index(drop=True)

        elif isinstance(self._testcases, dict) and isinstance(self.model, dict):
            testcases_df = []
            for k, v in self._testcases.items():
                if isinstance(v, dict):
                    model_testcases_df = pd.DataFrame.from_dict(
                        [
                            {
                                "dataset_name": k,
                                "model_name": model_name,
                                **sample.to_dict(),
                            }
                            for model_name, samples in v.items()
                            if isinstance(samples, list)
                            for sample in samples
                        ]
                    )
                else:
                    model_testcases_df = pd.DataFrame.from_dict(
                        [{"model_name": k, **x.to_dict()} for x in v]
                    )

                if "prompt" in model_testcases_df.columns:
                    return model_testcases_df.fillna("-")

                elif (
                    "test_case" in model_testcases_df.columns
                    and "original_question" in model_testcases_df.columns
                ) and self.task != "political":
                    model_testcases_df["original_question"].update(
                        model_testcases_df.pop("test_case")
                    )

                testcases_df.append(model_testcases_df)
            testcases_df = pd.concat(testcases_df).reset_index(drop=True)

            columns = [c for c in column_order if c in testcases_df.columns]
            testcases_df = testcases_df[columns]

            return testcases_df.fillna("-")

        elif self.is_multi_dataset and isinstance(self._testcases, dict):
            testcases_df = pd.DataFrame(
                [
                    {**x.to_dict(), "dataset_name": dataset_name}
                    for dataset_name, samples in self._testcases.items()
                    for x in samples
                ]
            )
            testcases_df = testcases_df.reset_index(drop=True)
            if "prompt" in testcases_df.columns:
                return testcases_df.fillna("-")

            elif (
                "test_case" in testcases_df.columns
                and "original_question" in testcases_df.columns
            ) and self.task != "political":
                testcases_df["original_question"].update(testcases_df.pop("test_case"))

            if hasattr(self, "is_multi_dataset") and self.is_multi_dataset:
                column_order.insert(2, "dataset_name")
            columns = [c for c in column_order if c in testcases_df.columns]
            testcases_df = testcases_df[columns]

            return testcases_df.fillna("-")

        else:
            testcases_df = pd.DataFrame([x.to_dict() for x in self._testcases])
            testcases_df = testcases_df.reset_index(drop=True)
            if "prompt" in testcases_df.columns:
                return testcases_df.fillna("-")

            elif (
                "test_case" in testcases_df.columns
                and "original_question" in testcases_df.columns
            ) and self.task != "political":
                testcases_df["original_question"].update(testcases_df.pop("test_case"))

            if "dataset_name" in column_order:
                column_order.remove("dataset_name")

        columns = [c for c in column_order if c in testcases_df.columns]
        testcases_df = testcases_df[columns]

        testcases_df = testcases_df[
            ~testcases_df["test_type"].isin(["degradation_analysis", "amega"])
        ]

        return testcases_df.fillna("-")

    def save(self, save_dir: str, include_generated_results: bool = False) -> None:
        """Save the configuration, generated testcases and the `DataFactory` to be reused later.

        Args:
            save_dir (str): path to folder to save the different files
        Returns:

        """
        if self._config is None:
            raise RuntimeError(Errors.E015())

        if self._testcases is None:
            raise RuntimeError(Errors.E016())

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        if include_generated_results and self._generated_results:
            with open(os.path.join(save_dir, "generated_results.pkl"), "wb") as writer:
                pickle.dump(self._generated_results, writer)

        with open(os.path.join(save_dir, "config.yaml"), "w", encoding="utf-8") as yml:
            yml.write(yaml.safe_dump(self._config_copy))

        with open(os.path.join(save_dir, "test_cases.pkl"), "wb") as writer:
            pickle.dump(self._testcases, writer)

        with open(os.path.join(save_dir, "data.pkl"), "wb") as writer:
            pickle.dump(self.data, writer)

    @classmethod
    def load(
        cls,
        save_dir: str,
        task: str,
        model: Optional[Union[list, dict]] = None,
        load_testcases: bool = False,
        load_model_response: bool = False,
    ) -> "Harness":
        """Loads a previously saved `Harness` from a given configuration and dataset

        Args:
            save_dir (str):
                path to folder containing all the needed files to load an saved `Harness`
            task (str):
                task for which the model is to be evaluated.
            model (Union[list, dict], optional): Specifies the model to be evaluated.
                If provided as a list, each element should be a dictionary with 'model' and 'hub' keys.
                If provided as a dictionary, it must contain 'model' and 'hub' keys when specifying a path.
            hub (str, optional):
                model hub to load from the path. Required if path is passed as 'model'.

        Returns:
            Harness:
                `Harness` loaded from from a previous configuration along with the new model to evaluate
        """
        if not os.path.isdir(save_dir):
            raise OSError(Errors.E092(directory=save_dir))

        for filename in ["config.yaml", "data.pkl"]:
            if not os.path.exists(os.path.join(save_dir, filename)):
                raise OSError(Errors.E017(filename=filename))

        with open(os.path.join(save_dir, "data.pkl"), "rb") as reader:
            data = pickle.load(reader)

        harness = Harness(
            task=task,
            model=model,
            data={"data_source": data},
            config=os.path.join(save_dir, "config.yaml"),
        )
        if load_testcases:
            if os.path.exists(os.path.join(save_dir, "test_cases.pkl")):
                with open(os.path.join(save_dir, "test_cases.pkl"), "rb") as reader:
                    testcases = pickle.load(reader)
                if harness.is_multi_dataset:
                    for _, samples in testcases.items():
                        for sample in samples:
                            if sample.category is not None and sample.category not in [
                                "accuracy",
                                "fairness",
                                "representation",
                            ]:
                                sample.expected_results = None
                else:
                    for sample in testcases:
                        sample.expected_results = None
                harness._testcases = testcases
            else:
                logging.warning(Warnings.W013(save_dir=save_dir))
                harness.generate()
        else:
            harness.generate()

        if load_model_response and os.path.exists(
            os.path.join(save_dir, "generated_results.pkl")
        ):
            with open(os.path.join(save_dir, "generated_results.pkl"), "rb") as reader:
                generated_results = pickle.load(reader)
            harness._generated_results = generated_results
        return harness

    def edit_testcases(
        self, output_path: str = "./edit_testcases.csv", return_dataframe=False, **kwargs
    ):
        """Testcases are exported to a csv file to be edited.

        The edited file can be imported back to the harness

        Args:
            output_path (str): path to save the testcases to
        """
        temp_df = self.testcases(additional_cols=True)
        temp_df = temp_df[temp_df["category"].isin(["robustness", "bias"])]
        if return_dataframe:
            return temp_df
        temp_df.to_csv(output_path, index=False)

    def import_edited_testcases(self, input_path: str, **kwargs):
        """Testcases are imported from a csv file

        Args:
            input_path (str): location of the file to load
        """
        if isinstance(self.data, dict) and not self.__is_multi_model:
            self._testcases = {}
        elif isinstance(self.data, list):
            self._testcases = []

        # check the category of the testcases and count the categories
        categories_count = list(self._config.get("tests", {}))

        # Remove the "robustness" and "bias" categories from categories_count
        categories_count = len(
            [
                category
                for category in categories_count
                if category not in ["robustness", "bias", "defaults"]
            ]
        )

        # multi dataset case is handled separately
        if isinstance(self._testcases, dict) and not self.__is_multi_model:
            temp_testcases = {
                k: [
                    sample
                    for sample in v
                    if sample.category not in ["robustness", "bias"]
                ]
                for k, v in self._testcases.items()
            }

            if categories_count > 0:
                testcases = self.__temp_generate()._testcases
                for k, v in testcases.items():
                    if k in temp_testcases:
                        temp_testcases[k].extend(v)
                    else:
                        temp_testcases[k] = v

            imported_testcases = DataFactory(
                {"data_source": input_path}, task=self.task, is_import=True
            ).load()

            # merge the testcases with the imported ones to the temp_testcases
            for name, list_samples in imported_testcases.items():
                # check the task and apply transformations
                if name not in temp_testcases:
                    temp_testcases[name] = list_samples
                temp_testcases[name].extend(list_samples)

            # update the testcases in the harness
            for k, v in temp_testcases.items():
                if k in self._testcases:
                    self._testcases[k].extend(v)
                else:
                    self._testcases[k] = v

        # single dataset case
        elif isinstance(self._testcases, list):
            temp_testcases = [
                sample
                for sample in self._testcases
                if sample.category not in ["robustness", "bias"]
            ]

            if categories_count > 0:
                testcases = self.__temp_generate()._testcases
                temp_testcases.extend(testcases)

            self._testcases = DataFactory(
                {"data_source": input_path}, task=self.task, is_import=True
            ).load()

            self._testcases.extend(temp_testcases)

        return self

    @staticmethod
    def available_tests(test_type: str = None) -> Dict[str, List[str]]:
        """Returns a dictionary of available tests categorized by test type.

        Args:
            test_type (str, optional): The specific test type to retrieve. Defaults to None.

        Returns:
            dict: Returns a dictionary containing available tests for the specified test type and defaults to all available tests.

        Raises:
            ValueError: If an invalid test type is provided.
        """
        test_scenarios = TestFactory.test_scenarios()
        available_tests = {
            test: list(scenarios.keys()) for test, scenarios in test_scenarios.items()
        }

        if test_type:
            if test_type not in available_tests.keys():
                raise ValueError(
                    Errors.E018(
                        test_type=test_type, available_tests=available_tests.keys()
                    )
                )
            return {test_type: available_tests[test_type]}

        return available_tests

    def pass_custom_data(
        self,
        file_path: str,
        test_name: str = None,
        task: str = None,
        append: bool = False,
    ) -> None:
        """Load custom data from a JSON file and store it in a class variable.

        Args:
            file_path (str): Path to the JSON file.
            test_name (str, optional): Name parameter. Defaults to None.
            task (str, optional): Task type. Either "bias" or "representation". Defaults to None.
            append (bool, optional): Whether to append the data or overwrite it. Defaults to False.
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        if task == "bias":
            if test_name not in (
                "Country-Economic-Bias",
                "Religion-Bias",
                "Ethnicity-Name-Bias",
                "Gender-Pronoun-Bias",
            ):
                raise ValueError(Errors.E019(test_name=test_name))

            TestFactory.call_add_custom_bias(data, test_name, append)
        elif task == "representation":
            if test_name not in (
                "Country-Economic-Representation",
                "Religion-Representation",
                "Ethnicity-Representation",
                "Label-Representation",
            ):
                raise ValueError(Errors.E020(test_name=test_name))

            RepresentationOperation.add_custom_representation(
                data, test_name, append, check=self.task
            )

        else:
            raise ValueError(Errors.E021(category=task))

    def upload_folder_to_hub(
        repo_name: str,
        repo_type: str,
        folder_path: str,
        token: str,
        model_type: str = "huggingface",
        exist_ok: bool = False,
    ):
        """
        Uploads a folder containing a model or dataset to the Hugging Face Model Hub or Dataset Hub.

        This function facilitates the process of uploading a local folder containing a model or dataset to the Hugging Face
        Model Hub or Dataset Hub. It requires proper authentication through a valid token.

        Args:
            repo_name (str): The name of the repository on the Hub.
            repo_type (str): The type of the repository, either "model" or "dataset".
            folder_path (str): The local path to the folder containing the model or dataset files to be uploaded.
            token (str): The authentication token for accessing the Hugging Face Hub services.
            model_type (str, optional): The type of the model, currently supports "huggingface" and "spacy".
                                    Defaults to "huggingface".
            exist_ok (bool, optional): If True, do not raise an error if repo already exists.

        Raises:
            ValueError: If a valid token is not provided for Hugging Face Hub authentication.
            ModuleNotFoundError: If required package is not installed. This package needs to be installed based on
                                model_type ("huggingface" or "spacy").
        """
        if token is None:
            raise ValueError(Errors.E022())
        subprocess.run(["huggingface-cli", "login", "--token", token], check=True)

        if (
            model_type == "huggingface" and repo_type == "model"
        ) or repo_type == "dataset":
            LIB_NAME = "huggingface_hub"

            if try_import_lib(LIB_NAME):
                huggingface_hub = importlib.import_module(LIB_NAME)
                HfApi = getattr(huggingface_hub, "HfApi")
            else:
                raise ModuleNotFoundError(Errors.E023(LIB_NAME=LIB_NAME))
            api = HfApi()

            repo_id = repo_name.split("/")[1]
            api.create_repo(repo_id, repo_type=repo_type, exist_ok=exist_ok)

            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_name,
                repo_type=repo_type,
            )
        elif model_type == "spacy" and repo_type == "model":
            LIB_NAME = "spacy_huggingface_hub"

            if try_import_lib(LIB_NAME):
                dataset_module = importlib.import_module(LIB_NAME)
                push = getattr(dataset_module, "push")
            else:
                raise ModuleNotFoundError(Errors.E023(LIB_NAME=LIB_NAME))

            meta_path = os.path.join(folder_path, "meta.json")
            with open(meta_path, "r") as meta_file:
                meta_data = json.load(meta_file)

            lang = meta_data["lang"]
            version = meta_data["version"]

            v = f"{lang}_pipeline-{version}"
            wheel_filename = f"{v}-py3-none-any.whl"

            output_dir_base = "output"
            output_dir = output_dir_base
            index = 1
            while os.path.exists(output_dir):
                output_dir = f"{output_dir_base}{index}"
                index += 1

            os.makedirs(output_dir, exist_ok=True)
            wheel_path = os.path.join(output_dir, v, "dist", wheel_filename)

            os.system(f"python -m spacy package {folder_path} {output_dir} --build wheel")

            push(wheel_path)

    def upload_file_to_hub(
        repo_name: str,
        repo_type: str,
        file_path: str,
        token: str,
        exist_ok: bool = False,
        split: str = "train",
    ):
        """Uploads a file or a Dataset to the Hugging Face Model Hub.

        Args:
            repo_name (str): The name of the repository in the format 'username/repository'.
            repo_type (str): The type of the repository, e.g: 'dataset' or 'model'.
            file_path (str): Path to the file to be uploaded.
            token (str): Hugging Face Hub authentication token.
            exist_ok (bool, optional): If True, do not raise an error if repo already exists.
            split (str, optional): The split of the dataset. Defaults to 'train'.

        Raises:
            ValueError: Raised if a valid token is not provided.
            ModuleNotFoundError: Raised if required packages are not installed.

        Returns:
            None
        """
        if token is None:
            raise ValueError(Errors.E022())
        subprocess.run(["huggingface-cli", "login", "--token", token], check=True)

        file_extension = file_path.split(".")[-1]
        path_in_repo = os.path.basename(file_path)
        if file_extension != "conll":
            LIB_NAME = "huggingface_hub"
            if try_import_lib(LIB_NAME):
                huggingface_hub = importlib.import_module(LIB_NAME)
                HfApi = getattr(huggingface_hub, "HfApi")
            else:
                raise ModuleNotFoundError(Errors.E023(LIB_NAME=LIB_NAME))

            api = HfApi()
            repo_id = repo_name.split("/")[1]
            api.create_repo(repo_id, repo_type=repo_type, exist_ok=exist_ok)

            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id=repo_name,
                repo_type=repo_type,
                token=token,
            )
        else:
            LIB_NAME = "datasets"
            if try_import_lib(LIB_NAME):
                dataset_module = importlib.import_module(LIB_NAME)
                DatasetDict = getattr(dataset_module, "DatasetDict")
                Dataset = getattr(dataset_module, "Dataset")

            else:
                raise ModuleNotFoundError(Errors.E023(LIB_NAME=LIB_NAME))

            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            data = []
            tokens = []
            ner_tags = []

            for line in lines:
                line = line.strip()
                if line:
                    if not line.startswith("-DOCSTART-"):
                        parts = line.split()
                        tokens.append(parts[0])
                        ner_tags.append(parts[-1])
                elif tokens:
                    data.append({"tokens": tokens, "ner_tags": ner_tags})
                    tokens = []
                    ner_tags = []

            df = pd.DataFrame(data)
            dataset = Dataset.from_pandas(df)
            ds = DatasetDict({split: dataset})

            ds.push_to_hub(
                repo_id=repo_name,
                token=token,
            )

    def __multi_datasets_loading(self, task, hub, model, data):
        """Loads the data from the given source."""
        loaded_data = {}
        for dataset in data:
            processed_data = self.__single_dataset_loading(task, hub, model, dataset)
            dataset_name = dataset.get("data_source")
            loaded_data[dataset_name] = processed_data
        self.is_multi_dataset = True
        return loaded_data

    def __single_dataset_loading(self, task, hub, model, data):
        """Loads the data from the given source."""
        # Loading default datasets
        o_data = []
        if data is None and (self.task, model, hub) in self.DEFAULTS_DATASET:
            data_path = os.path.join(
                "data", self.DEFAULTS_DATASET[(self.task, model, hub)]
            )
            data = {"data_source": resource_filename("langtest", data_path)}
            o_data = DataFactory(data, task=self.task).load()
            if model == "textcat_imdb":
                model = resource_filename("langtest", "data/textcat_imdb")
            self.is_default = True
            logging.info(Warnings.W002(info=(self.task, model, hub)))
        elif data is None and self.task.category == "ideology":
            o_data = []
        elif data is None and (task, model, hub) not in self.DEFAULTS_DATASET.keys():
            raise ValueError(Errors.E004())

        if isinstance(data, dict):
            if isinstance(data.get("data_source"), list):
                o_data = data.get("data_source")
            elif isinstance(data.get("data_source"), dict):
                o_data = data.get("data_source")
                self.is_multi_dataset = True
                return o_data
            else:
                o_data = DataFactory(data, task=self.task).load()

        self.is_multi_dataset = False
        return o_data

    # Generate testcases functions
    def __single_dataset_generate(self, dataset: list):
        testcases = None

        tests = self._config["tests"]
        m_data = [sample.model_copy() for sample in dataset]

        if self.task in ["text-classification", "ner"]:
            if not isinstance(self.model, dict):
                _ = [
                    setattr(sample, "expected_results", self.model(sample.original))
                    for sample in m_data
                ]
            else:
                testcases = {}
                for k, v in self.model.items():
                    _ = [
                        setattr(sample, "expected_results", v(sample.original))
                        for sample in m_data
                    ]
                    (testcases[k]) = TestFactory.transform(
                        self.task, dataset, tests, m_data=m_data
                    )

                return testcases

        elif str(self.task) in ("question-answering", "summarization"):
            if (
                "bias" in tests.keys()
                and isinstance(self.__data_dict, dict)
                and "bias" == self.__data_dict.get("split")
            ):
                if self.__data_dict["data_source"] in ("BoolQ", "XSum"):
                    tests_to_filter = tests["bias"].keys()
                    testcases = DataFactory.filter_curated_bias(tests_to_filter, dataset)
                    if len(tests.keys()) > 2:
                        tests = {k: v for k, v in tests.items() if k != "bias"}
                        (other_testcases) = TestFactory.transform(
                            self.task, dataset, tests, m_data=m_data
                        )
                        testcases.extend(other_testcases)
                    return testcases
                else:
                    raise ValueError(
                        Errors.E007(data_source=self.__data_dict["data_source"])
                    )
            else:
                testcases = TestFactory.transform(
                    self.task, dataset, tests, m_data=m_data
                )
                return testcases

        elif str(self.task) in ["sensitivity", "sycophancy"]:
            test_data_sources = {
                "add_toxic_words": ("wikiDataset"),
                "add_negation": ("NQ-open", "OpenBookQA"),
                "sycophancy_math": ("synthetic-math-data"),
                "sycophancy_nlp": ("synthetic-nlp-data"),
            }

            category = tests.get(str(self.task).split("-")[0], {})
            test_name = next(iter(category), None)
            if test_name in test_data_sources:
                selected_data_sources = test_data_sources[test_name]

                if self.__data_dict["data_source"] in selected_data_sources:
                    testcases = TestFactory.transform(
                        self.task, dataset, tests, m_data=m_data
                    )
                    return self
                else:
                    raise ValueError(
                        Errors.E008(
                            test_name=test_name,
                            data_source=self.__data_dict["data_source"],
                            selected_data_sources=selected_data_sources,
                        )
                    )

            else:
                raise ValueError(Errors.E009(test_name=test_name))

        testcases = TestFactory.transform(self.task, dataset, tests, m_data=m_data)
        return testcases

    def __multi_datasets_generate(self, dataset: Dict[str, list]):
        testcases = {}
        if not isinstance(self.model, dict):
            for dataset_name, samples in dataset.items():
                print(f"{'':=^80}\n{dataset_name:^80}\n{'':=^80}")
                testcases[dataset_name] = self.__single_dataset_generate(samples)
                print(f"{'':-^80}\n")
        elif isinstance(self.data, list) and self.__is_multi_model:
            temp_testcases = self.__single_dataset_generate(dataset)
            if isinstance(temp_testcases, dict) and set(temp_testcases.keys()) == set(
                self.model.keys()
            ):
                testcases = temp_testcases
            else:
                for model_name, _ in self.model.items():
                    testcases[model_name] = [
                        sample.model_copy() for sample in temp_testcases
                    ]

        else:
            for dataset_name, samples in dataset.items():
                print(f"{'':=^80}\n{dataset_name:^80}\n{'':=^80}")
                if dataset_name not in testcases:
                    testcases[dataset_name] = {}
                for model_name, _ in self.model.items():
                    if model_name not in testcases[dataset_name]:
                        testcases[dataset_name][model_name] = []
                    testcases[dataset_name][model_name] = self.__single_dataset_generate(
                        samples
                    )
                print(f"{'':-^80}\n")

        return testcases

    # Run testcases functions
    def __single_dataset_run(
        self,
        testcases: list,
        data,
        checkpoint: bool = False,
        save_checkpoints_dir: str = None,
        batch_size: int = 500,
        dataset_name: str = None,
    ):
        generated_results = None
        if testcases is None:
            raise RuntimeError(Errors.E010())

        if not isinstance(testcases, dict) and not isinstance(self.model, dict):
            if checkpoint:
                if self.batches is None:
                    if self.is_multi_dataset:
                        self.batches = defaultdict(dict)
                        for dataset, samples in self._testcases.items():
                            checkpoint_manager = CheckpointManager(
                                checkpoint_folder=f"{save_checkpoints_dir}/{dataset}"
                            )
                            self.batches[dataset] = divide_into_batches(
                                samples, batch_size
                            )
                            checkpoint_manager.save_all_batches(self.batches[dataset])
                            logging.warning(
                                Warnings.W022(
                                    name=dataset, total_batches=len(self.batches[dataset])
                                )
                            )
                    else:
                        checkpoint_manager = CheckpointManager(
                            checkpoint_folder=save_checkpoints_dir
                        )
                        self.batches = divide_into_batches(testcases, batch_size)
                        checkpoint_manager.save_all_batches(self.batches)
                        logging.warning(Warnings.W018(total_batches=len(self.batches)))

                    self.save(save_checkpoints_dir)

                if generated_results is None:
                    generated_results = []

                if self.is_multi_dataset:
                    # print(dataset_name)
                    batches = self.batches[dataset_name]
                    checkpoint_manager = CheckpointManager(
                        checkpoint_folder=f"{save_checkpoints_dir}/{dataset_name}"
                    )
                else:
                    batches = self.batches
                    checkpoint_manager = CheckpointManager(
                        checkpoint_folder=save_checkpoints_dir
                    )

                for i, batch in batches.items():
                    batch_results = TestFactory.run(
                        batch,
                        self.model,
                        is_default=self.is_default,
                        raw_data=data,
                        **self._config.get("model_parameters", {}),
                    )

                    checkpoint_manager.save_checkpoint(
                        check_point_extension=f"batch_{i}", results_so_far=batch_results
                    )
                    generated_results.extend(batch_results)
                    checkpoint_manager.update_status(batch_number=i)

            else:
                generated_results = TestFactory.run(
                    testcases,
                    self.model,
                    is_default=self.is_default,
                    raw_data=data,
                    **self._config.get("model_parameters", {}),
                )
            if self._checkpoints is not None:
                if self.is_multi_dataset and isinstance(self._checkpoints, dict):
                    if self._generated_results is None:
                        self._generated_results = defaultdict(list)
                    for k, v in self._checkpoints.items():
                        if k not in self._generated_results:
                            self._generated_results[k] = []
                        self._generated_results[k].extend(v)
                    self._checkpoints = None
                else:
                    generated_results.extend(self._checkpoints)
        else:
            # multi-model run
            generated_results = defaultdict(list)
            if checkpoint:
                if self.batches is None:
                    self.batches = {}
                    for k, v in self.model.items():
                        self.batches[k] = divide_into_batches(testcases[k], batch_size)
                        print(
                            Warnings.W019(model_name=k, total_batches=len(self.batches))
                        )

                    for k, v in self.batches.items():
                        k_checkpoint_dir = os.path.join(save_checkpoints_dir, k)
                        checkpoint_manager = CheckpointManager(
                            checkpoint_folder=k_checkpoint_dir
                        )
                        checkpoint_manager.save_all_batches(v)

                    self.save(save_checkpoints_dir)

                for k, v in self.model.items():
                    k_checkpoint_dir = os.path.join(save_checkpoints_dir, k)
                    checkpoint_manager = CheckpointManager(
                        checkpoint_folder=k_checkpoint_dir
                    )
                    generated_results[k] = []
                    for i, batch in self.batches[k].items():
                        batch_results = TestFactory.run(
                            batch,
                            v,
                            is_default=self.is_default,
                            raw_data=data,
                            **self._config.get("model_parameters", {}),
                        )

                        checkpoint_manager.save_checkpoint(
                            check_point_extension=f"batch_{i}",
                            results_so_far=batch_results,
                        )
                        generated_results[k].extend(batch_results)
                        checkpoint_manager.update_status(batch_number=i)

            else:
                for k, v in self.model.items():
                    generated_results[k] = TestFactory.run(
                        testcases[k],
                        v,
                        is_default=self.is_default,
                        raw_data=data,
                        **self._config.get("model_parameters", {}),
                    )
            if self._checkpoints is not None:
                for k, v in self.model.items():
                    generated_results[k].extend(self._checkpoints[k])

        # clear cache
        if isinstance(self.model, dict):
            for k, v in self.model.items():
                v.predict.cache_clear()
        else:
            self.model.predict.cache_clear()
        return generated_results

    def __multi_datasets_run(
        self,
        testcases: Dict[str, list],
        checkpoint: bool = False,
        save_checkpoints_dir: str = None,
        batch_size: int = 500,
    ):
        generated_results = {}

        # temp_store_prompt
        temp_store_prompt = self._config.get("model_parameters", {}).get(
            "user_prompt", None
        )

        # Run the testcases for each dataset
        for dataset_name, samples in testcases.items():
            # set prompt in prompt manager
            if hasattr(self, "prompt_manager") and self.prompt_manager is not None:
                self.prompt_manager.default_state = dataset_name
            # update user prompt for each dataset
            if temp_store_prompt and isinstance(temp_store_prompt, dict):
                self._config.get("model_parameters", {}).update(
                    {"user_prompt": temp_store_prompt.get(dataset_name)}
                )
            # Get the raw data for the dataset
            if isinstance(self.data, dict):
                raw_data = self.data.get(dataset_name)
            elif isinstance(self.data, list):
                raw_data = self.data

            print(f"{'':=^80}\n{dataset_name:^80}\n{'':=^80}")

            # Check if the dataset is empty
            if len(samples) == 0:
                print(Warnings.W023(name=dataset_name))
            else:
                generated_results[dataset_name] = self.__single_dataset_run(
                    samples,
                    raw_data,
                    checkpoint,
                    f"{save_checkpoints_dir}",
                    batch_size,
                    dataset_name,
                )

            print(f"{'':-^80}\n")

        # resore user prompt
        if temp_store_prompt:
            self._config.get("model_parameters", {}).update(
                {"user_prompt": temp_store_prompt}
            )

        if (
            self.is_multi_dataset
            and self._generated_results is None
            and self._checkpoints is not None
        ):
            self._generated_results = self._checkpoints
            self._checkpoints = None
            return self._generated_results
        return generated_results

    def __reset_defaults(self):
        """Reset the default values."""
        model_response = TestResultManager()
        model_response.clear_data()

        # Reset the PromptManager
        prompt_manager = PromptManager()
        prompt_manager.reset()

        # Prometheus eval
        from langtest.metrics.prometheus_eval import PrometheusEval

        PrometheusEval.reset_pipeline()

    def __tracking(self, *args, **kwargs):
        """Track the progress of the testcases."""
        if self.__benchmarking:
            df = self.generated_results()

            path = self.__benchmarking.get(
                os.path.expanduser("save_dir"),
                os.path.expanduser("~/.langtest/leaderboard/"),
            )
            summary = Summary(path)

            # temp dict
            temp_dict = {}
            if isinstance(self.__data_dict, dict):
                temp_dict[self.__data_dict.get("data_source")] = self.__data_dict
            else:
                for i in self.__data_dict:
                    temp_dict[i.get("data_source")] = i

            # add the dataset_name column if the data is single dataset
            if isinstance(self.__data_dict, dict) and (not self.is_multi_dataset):
                df["dataset_name"] = self.__data_dict.get("data_source", "-")

            df["split"] = df["dataset_name"].apply(
                lambda x: temp_dict[x].get("split", "-")
            )
            df["subset"] = df["dataset_name"].apply(
                lambda x: temp_dict[x].get("subset", "-")
            )

            df["hub"] = self.__model_info.get("hub", "-")
            if self.__model_info.get("hub", "-") == "lm-studio":
                import requests as req

                response = req.get(
                    "http://localhost:1234/v1/models",
                ).json()

                model_name = response["data"][0]["id"]
                df["model"] = model_name
            else:
                df["model"] = self.__model_info.get("model", "-")
            df["task"] = str(self.task)
            summary.add_report(df)

    def get_leaderboard(
        self,
        indices=[],
        columns=[],
        category=False,
        split_wise=False,
        test_wise=False,
        rank_by: Union[str, list] = "Avg",
        *args,
        **kwargs,
    ):
        """Get the rank of the model on the leaderboard."""

        if os.path.exists(os.path.expanduser(self.__benchmarking.get("save_dir"))):
            path = os.path.expanduser(self.__benchmarking.get("save_dir"))
        elif os.path.exists(os.path.expanduser("~/.langtest/leaderboard/")):
            path = os.path.expanduser("./.langtest/leaderboard/")
        else:
            raise FileNotFoundError(f"Summary.csv File is not exists in {path}")

        leaderboard = Leaderboard(path)

        # print(leaderboard.default().to_markdown())
        if indices or columns:
            return leaderboard.custom_wise(indices, columns)
        if category:
            return leaderboard.category_wise(rank_by=rank_by)

        if test_wise:
            return leaderboard.test_wise(rank_by=rank_by)

        if split_wise:
            return leaderboard.split_wise(rank_by=rank_by)

        return leaderboard.default(rank_by=rank_by)

    def __temp_generate(self, *args, **kwargs):
        """Temporary function to generate the testcases."""

        # temp config other than robustness and bias
        temp_config = {
            "tests": {
                k: v
                for k, v in self._config.get("tests", {}).items()
                if k not in ["robustness", "bias"]
            }
        }
        print(
            f"{'':=^80}\n{'Adding Test Cases Other than Robustness and Bias Categories ':^80}\n{'':=^80}"
        )
        temp_harness = self.__class__(
            task=str(self.task),
            model=self.__model_info,
            data=self.__data_dict,
            config=temp_config,
        )
        temp_harness.generate()
        print(f"{'':-^80}\n")

        return temp_harness
