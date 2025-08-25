from abc import ABC, abstractmethod
import asyncio
from collections import defaultdict
import logging
import os
import random
import re
from typing import List, Dict, Literal, TypedDict, Union

import importlib_resources
from langtest.errors import Errors, Warnings
from langtest.modelhandler.modelhandler import ModelAPI
from langtest.tasks.task import TaskManager
from langtest.transform.base import ITests, TestFactory
from langtest.transform.utils import (
    GENERIC2BRAND_TEMPLATE,
    CLINICALNOTE_SUMMARY_INSTRUCTIONS,
    filter_unique_samples,
)
from langtest.utils.custom_types.helpers import (
    HashableDict,
    build_qa_input,
    build_qa_prompt,
)
from langtest.utils.custom_types.sample import QASample, Sample


class ClinicalTestFactory(ITests):
    """Factory class for the clinical tests"""

    alias_name = "clinical"
    supported_tasks = [
        "clinical",
        "text-generation",
    ]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initializes the ClinicalTestFactory"""

        self.supported_tests = self.available_tests()
        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

        # check if configured tests are supported
        not_supported_tests = set(self.tests) - set(self.supported_tests)
        if len(not_supported_tests) > 0:
            raise ValueError(
                Errors.E049(
                    not_supported_tests=not_supported_tests,
                    supported_tests=list(self.supported_tests.keys()),
                )
            )

    def transform(self) -> List[Sample]:
        """Nothing to use transform for no longer to generating testcases.

        Returns:
            Empty list

        """
        all_samples = []
        no_transformation_applied_tests = {}
        tests_copy = self.tests.copy()
        for test_name, params in tests_copy.items():
            test_func = self.supported_tests[test_name].transform
            data_handler_copy = [sample.model_copy() for sample in self.data_handler]
            transformed_samples = test_func(data_handler_copy, **params)

            if test_name in (
                "demographic-bias",
                "amega",
                "clinical_note_summary",
                "mental_health",
            ):
                all_samples.extend(transformed_samples)
            else:
                new_transformed_samples, removed_samples_tests = filter_unique_samples(
                    TestFactory.task, transformed_samples, test_name
                )
                all_samples.extend(new_transformed_samples)

                no_transformation_applied_tests.update(removed_samples_tests)

        if no_transformation_applied_tests:
            warning_message = Warnings._W009
            for test, count in no_transformation_applied_tests.items():
                warning_message += Warnings._W010.format(
                    test=test, count=count, total_sample=len(self.data_handler)
                )

            logging.warning(warning_message)

        return all_samples

    @classmethod
    def available_tests(cls) -> Dict[str, Union["BaseClinical", "ClinicalTestFactory"]]:
        """Returns the empty dict, no clinical tests

        Returns:
            Dict[str, str]: Empty dict, no clinical tests
        """
        test_types = BaseClinical.available_tests()
        # test_types.update({"demographic-bias": cls})
        return test_types


class BaseClinical(ABC):
    """
    Baseclass for the clinical tests
    """

    test_types = defaultdict(lambda: BaseClinical)
    alias_name = None
    supported_tasks = [
        "question-answering",
    ]

    # TestConfig
    TestConfig = TypedDict(
        "TestConfig",
        min_pass_rate=float,
    )

    @staticmethod
    @abstractmethod
    def transform(*args, **kwargs):
        """Transform method for the clinical tests"""

        raise NotImplementedError

    @staticmethod
    @abstractmethod
    async def run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Run method for the clinical tests"""

        progress = kwargs.get("progress_bar", False)
        for sample in sample_list:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list

    @classmethod
    async def async_run(cls, *args, **kwargs):
        """Async run method for the clinical tests"""
        created_task = asyncio.create_task(cls.run(*args, **kwargs))
        return await created_task

    @classmethod
    def available_tests(cls) -> Dict[str, "BaseClinical"]:
        """Available tests for the clinical tests"""

        return cls.test_types

    def __init_subclass__(cls) -> None:
        """Initializes the subclass for the clinical tests"""
        alias = cls.alias_name if isinstance(cls.alias_name, list) else [cls.alias_name]
        for name in alias:
            BaseClinical.test_types[name] = cls


class DemographicBias(BaseClinical):
    """
    DemographicBias class for the clinical tests
    """

    alias_name = ["demographic-bias", "demographic_bias"]
    supported_tasks = ["question-answering", "text-generation"]

    @staticmethod
    def transform(sample_list: List[Sample], *args, **kwargs):
        """Transform method for the DemographicBias class"""
        for sample in sample_list:
            sample.test_type = "demographic-bias"
            sample.category = "clinical"
        return sample_list

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs):
        """Run method for the DemographicBias class"""

        progress_bar = kwargs.get("progress_bar", False)

        for sample in sample_list:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
            if progress_bar:
                progress_bar.update(1)
        return sample_list


class Generic2Brand(BaseClinical):
    """
    GenericBrand class for the clinical tests
    """

    alias_name = "drug_generic_to_brand"
    template = GENERIC2BRAND_TEMPLATE

    @staticmethod
    def transform(sample_list: List[Sample] = [], *args, **kwargs):
        """Transform method for the GenericBrand class"""

        # reset the template
        Generic2Brand.template = GENERIC2BRAND_TEMPLATE

        # update the template with the special tokens
        system_token = kwargs.get("system_token", "system")
        user_token = kwargs.get("user_token", "user")
        assistant_token = kwargs.get("assistant_token", "assistant\n")
        end_token = kwargs.get("end_token", "\nend")

        Generic2Brand.template = Generic2Brand.template.format(
            system=system_token,
            user=user_token,
            assistant=assistant_token,
            end=end_token,
            text="{text}",
        )

        if len(sample_list) <= 0 or kwargs.get("curated_dataset", False):
            import pandas as pd

            task = TestFactory.task
            count = kwargs.get("count", 50)

            # loading the dataset and creating the samples
            data = []
            if task == "ner":
                dataset_path = "ner_g2b.jsonl"
            elif task == "question-answering":
                dataset_path = "qa_generic_to_brand_v2.jsonl"
                file_path = (
                    importlib_resources.files("langtest")
                    / "data"
                    / "DrugSwap"
                    / dataset_path
                )
                df = pd.read_json(file_path, lines=True)
                for _, row in df.iterrows():
                    sample = QASample(
                        original_context="-",
                        original_question=row["original_question"],
                        perturbed_question=row["perturbed_question"],
                    )
                    sample.expected_results = row["answer_option"]
                    # sample.actual_results = row["actual_results"]
                    sample.test_type = "drug_generic_to_brand"
                    sample.category = "clinical"
                    data.append(sample)

            return random.choices(data, k=count)
        else:
            # loading the posology model for the drug swap
            posology = Posology(drug_swap_type="generic_to_brand", seed=25)
            for sample in sample_list:
                sample.test_type = "drug_generic_to_brand"
                sample.category = "clinical"

                if isinstance(sample, QASample):
                    query = sample.original_question
                    if len(sample.options) > 1:
                        query = f"{query}\nOptions:\n{sample.options}"
                        sample.original_question = query
                        sample.options = "-"

                    sample.perturbed_question = posology(query)

                    if len(sample.original_context) > 1:
                        sample.perturbed_context = posology(sample.original_context)
                    else:
                        sample.perturbed_context = "-"

                    if isinstance(sample.expected_results, list):
                        sample.expected_results = "\n".join(sample.expected_results)

                else:
                    sample.test_case = posology(sample.original)

            return sample_list

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Run method for the GenericBrand class"""

        progress_bar = kwargs.get("progress_bar", False)

        for sample in sample_list:
            # if hasattr(sample, "run"):
            #     sample.run(model, **kwargs)
            # else:
            if isinstance(sample, QASample):
                temp_temlate = "Context:\n {context}\nQuestion:\n {text}"
                query = {"text": sample.perturbed_question}
                if len(sample.original_context) > 1:
                    query["context"] = sample.perturbed_context
                else:
                    temp_temlate = "Question:\n {text}"

                sample.actual_results = model.predict(
                    text=HashableDict(
                        {
                            "text": temp_temlate.format(**query),
                        }
                    ),
                    prompt=HashableDict(
                        {
                            "template": Generic2Brand.template,
                            "input_variables": ["text"],
                        }
                    ),
                    server_prompt="Perform the task to the best of your ability:",
                )
            else:
                sample.actual_results = model.predict(sample.test_case)
            if progress_bar:
                progress_bar.update(1)
            sample.state = "done"
        return sample_list


class Brand2Generic(BaseClinical):
    """
    BrandGeneric class for the clinical tests
    """

    alias_name = "drug_brand_to_generic"

    @staticmethod
    def transform(sampe_list: List[Sample] = [], *args, **kwargs):
        """Transform method for the BrandGeneric class"""

        # reset the template
        Generic2Brand.template = GENERIC2BRAND_TEMPLATE

        # update the template with the special tokens
        system_token = kwargs.get("system_token", "system")
        user_token = kwargs.get("user_token", "user")
        assistant_token = kwargs.get("assistant_token", "assistant\n")
        end_token = kwargs.get("end_token", "\nend")

        Generic2Brand.template = Generic2Brand.template.format(
            system=system_token,
            user=user_token,
            assistant=assistant_token,
            end=end_token,
            text="{text}",
        )

        if len(sampe_list) <= 0 or kwargs.get("curated_dataset", False):
            import pandas as pd

            task = TestFactory.task
            count = kwargs.get("count", 50)

            data = []
            if task == "ner":
                dataset_path = "ner_b2g.jsonl"
            elif task == "question-answering":
                dataset_path = "qa_brand_to_generic.jsonl"
                file_path = (
                    importlib_resources.files("langtest")
                    / "data"
                    / "DrugSwap"
                    / dataset_path
                )
                df = pd.read_json(file_path, lines=True)
                for _, row in df.iterrows():
                    sample = QASample(
                        original_context="-",
                        original_question=row["original_question"],
                        perturbed_question=row["perturbed_question"],
                    )
                    sample.expected_results = row["answer_option"]
                    # sample.actual_results = row["actual_results"]
                    sample.test_type = "drug_generic_to_brand"
                    sample.category = "clinical"
                    data.append(sample)

            return random.choices(data, k=count)
        else:
            # loading the posology model for the drug swap
            posology = Posology(drug_swap_type="brand_to_generic", seed=25)
            for sample in sampe_list:
                sample.test_type = "drug_brand_to_generic"
                sample.category = "clinical"

                if isinstance(sample, QASample):
                    query = sample.original_question
                    if len(sample.options) > 1:
                        query = f"{query}\nOptions:\n{sample.options}"
                        sample.original_question = query
                        sample.options = "-"

                    sample.perturbed_question = posology(query)

                    if len(sample.original_context) > 1:
                        sample.perturbed_context = posology(sample.original_context)
                    else:
                        sample.perturbed_context = "-"

                    if isinstance(sample.expected_results, list):
                        sample.expected_results = "\n".join(sample.expected_results)
                else:
                    sample.test_case = posology(sample.original)

            return sampe_list

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs):
        """Run method for the BrandGeneric class"""

        progress_bar = kwargs.get("progress_bar", False)

        for sample in sample_list:
            if isinstance(sample, QASample):
                # build the template
                temp_temlate = "Context:\n {context}\nQuestion:\n {text}"

                # build the query
                query = {"text": sample.perturbed_question}
                if len(sample.original_context) > 1:
                    query["context"] = sample.perturbed_context
                else:
                    temp_temlate = "Question:\n {text}"

                sample.actual_results = model.predict(
                    text=HashableDict(
                        {
                            "text": temp_temlate.format(**query),
                        }
                    ),
                    prompt=HashableDict(
                        {"template": Generic2Brand.template, "input_variables": ["text"]}
                    ),
                    server_prompt="Perform the task to the best of your ability:",
                )
            else:
                sample.actual_results = model.predict(sample.test_case)
            if progress_bar:
                progress_bar.update(1)
            sample.state = "done"
        return sample_list


class Posology:
    """Posology class is replacing the generic to brand or brand to generic drug names in given text"""

    def __init__(self, drug_swap_type="generic_to_brand", seed=25) -> None:
        """
        Initialize the Posology class.

        Args:
            drug_swap_type (str, optional): The type of drug swap to perform. Defaults to "generic_to_brand".
            seed (int, optional): The seed value for random number generation. Defaults to 25.
        """
        from johnsnowlabs import nlp, medical

        # Set the seed
        self.drug_swap_type = drug_swap_type
        self.seed = seed

        # Initialize Spark NLP
        self.spark = nlp.start()

        # Build the pipeline
        document_assembler = (
            nlp.DocumentAssembler().setInputCol("text").setOutputCol("document")
        )

        sentence_detector = (
            nlp.SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
        )

        tokenizer = nlp.Tokenizer().setInputCols("sentence").setOutputCol("token")

        word_embeddings = (
            nlp.WordEmbeddingsModel.pretrained(
                "embeddings_clinical", "en", "clinical/models"
            )
            .setInputCols(["sentence", "token"])
            .setOutputCol("embeddings")
        )

        # NER model to detect drug in the text
        clinical_ner = (
            medical.NerModel.pretrained("ner_posology", "en", "clinical/models")
            .setInputCols(["sentence", "token", "embeddings"])
            .setOutputCol("ner")
            .setLabelCasing("upper")
        )

        ner_converter = (
            medical.NerConverterInternal()
            .setInputCols(["sentence", "token", "ner"])
            .setOutputCol("ner_chunk")
            .setWhiteList(["DRUG"])
        )

        if self.drug_swap_type == "generic_to_brand":
            mapper_dataset = str(
                importlib_resources.files("langtest")
                / "data"
                / "resources"
                / "chunk_mapper_g2b_dataset.json"
            )

            chunkerMapper = (
                medical.ChunkMapperApproach()
                .setInputCols(["ner_chunk"])
                .setOutputCol("mappings")
                .setDictionary(mapper_dataset)
                .setRels(["brand"])
            )  # or change generic to brand

        elif self.drug_swap_type == "brand_to_generic":
            mapper_dataset = str(
                importlib_resources.files("langtest")
                / "data"
                / "resources"
                / "chunk_mapper_b2g_dataset.json"
            )
            chunkerMapper = (
                medical.ChunkMapperApproach()
                .setInputCols(["ner_chunk"])
                .setOutputCol("mappings")
                .setDictionary(mapper_dataset)
                .setRels(["generic"])
            )  # or change brand to generic

        # Define the pipeline
        self.pipeline = nlp.Pipeline().setStages(
            [
                document_assembler,
                sentence_detector,
                tokenizer,
                word_embeddings,
                clinical_ner,
                ner_converter,
                chunkerMapper,
            ]
        )

        text = ["The patient was given 1 unit of metformin daily."]
        test_data = self.spark.createDataFrame([text]).toDF("text")
        self.model = self.pipeline.fit(test_data)
        self.res = self.model.transform(test_data)

        # Light pipeline
        self.light_pipeline = nlp.LightPipeline(self.model)

    def __call__(self, text: str) -> str:
        """
        Applies the clinical transformation to the input text.

        Args:
            text (str): The input text to be transformed.

        Returns:
            str: The transformed text.
        """
        result = self.light_pipeline.fullAnnotate(text)
        return self.__drug_swap(result, text)

    def __drug_swap(self, result: str, text: str) -> str:
        """
        Swaps drug names in the given text with random alternatives.

        Args:
            result (str): The result string containing the drug information.
            text (str): The original text to perform the drug name swapping.

        Returns:
            str: The modified text with drug names swapped.

        """
        import random

        if self.seed:
            random.seed(self.seed)

        for n, maps in zip(result[0]["ner_chunk"], result[0]["mappings"]):
            # skip if drug brand is not found or generic is not found
            if maps.result == "NONE":
                continue
            words = maps.metadata["all_k_resolutions"].split(":::")

            # remove the word if length is 0 from the words
            words = [word for word in words if len(word) > 1]

            if len(words) > 0:
                random_word: str = random.choice(words) if len(words) > 1 else words[0]
                if len(random_word.strip()) > 0:
                    text = text.replace(n.result, random_word)

        return text


class FCT(BaseClinical):
    """
    FCT class for the clinical tests
    False Confidence Test
    """

    alias_name = "fct"
    supported_tasks = ["question-answering", "text-generation"]

    @staticmethod
    def transform(sample_list: List[Sample], *args, **kwargs):
        """Transform method for the FCT class"""

        transformed_samples = []
        upper_bound = len(sample_list) - 3

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):

                continue

            sample.category = "clinical"
            selected = (
                random.randint(idx, upper_bound) if idx <= upper_bound else upper_bound
            )
            if idx == selected:
                selected = (selected + 1) % len(sample_list)
            selected_sample = sample_list[selected]

            if hasattr(sample, "options") and sample.options not in ["-", None]:
                if isinstance(selected_sample.options, list):
                    sample.options = selected_sample.options + ["F. None of the above"]
                elif isinstance(
                    selected_sample.options, str
                ) and not selected_sample.options.endswith("F. None of the above"):
                    sample.options = f"{selected_sample.options}\nF. None of the above"
            elif hasattr(sample, "original_context") and sample.original_context not in [
                "-",
                None,
            ]:
                sample.original_context = selected_sample.original_context

            sample.perturbed_context = ""
            sample.perturbed_question = ""
            sample.expected_results = "None of the above"
            transformed_samples.append(sample)

        return transformed_samples

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Run method for the FCT class"""

        progress_bar = kwargs.get("progress_bar", False)

        for sample in sample_list:
            if sample.state != "done":
                original_text_input = build_qa_input(
                    context=sample.original_context,
                    question=sample.original_question,
                    options=sample.options,
                )
                prompt = build_qa_prompt(
                    original_text_input, "default_question_answering_prompt", **kwargs
                )
                sample.actual_results = model(original_text_input, prompt=prompt)
                sample.state = "done"
            if progress_bar:
                progress_bar.update(1)
        return sample_list


class NOTA(BaseClinical):
    """
    NOTA class for the clinical tests
    """

    alias_name = "nota"
    supported_tasks = ["question-answering", "text-generation"]

    @staticmethod
    def transform(sample_list: List[Sample], *args, **kwargs):
        """Transform method for the NOTA class"""

        transformed_samples = []
        for sample in sample_list:
            if sample.expected_results is None:
                continue
            sample.category = "clinical"

            true_answer = "\n".join(map(str, sample.expected_results))
            options = sample.options

            if options is None:
                continue
            if (
                true_answer in options
                and isinstance(options, str)
                and isinstance(true_answer, str)
            ):
                # split by any letter. or number. or ) by re
                options = re.sub(rf"{true_answer[3:]}", "None of the above", options)

            elif (
                true_answer in options
                and isinstance(options, list)
                and isinstance(true_answer, str)
            ):
                options = [
                    re.sub(rf"{true_answer}", "None of the above", option)
                    for option in options
                ]
            sample.options = options

            # extract the [*]. from the true answer one character and .
            option_pos = re.search(r"\b(?:[A-Za-z]|[0-9]|[IVXLCDM]+)[\.\)]", true_answer)
            option_pos.end()
            if option_pos:
                option_pos = option_pos.end()
                expected_results = f"{true_answer[:option_pos]} None of the above"
            else:
                expected_results = "None of the above"

            sample.expected_results = expected_results
            sample.perturbed_context = ""
            sample.perturbed_question = ""
            transformed_samples.append(sample)

        return transformed_samples

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Run method for the NOTA class"""

        progress_bar = kwargs.get("progress_bar", False)

        for sample in sample_list:
            if sample.state != "done":
                original_text_input = build_qa_input(
                    context=sample.original_context,
                    question=sample.original_question,
                    options=sample.options,
                )
                prompt = build_qa_prompt(
                    original_text_input, "default_question_answering_prompt", **kwargs
                )
                sample.actual_results = model(original_text_input, prompt=prompt)
                sample.state = "done"
            if progress_bar:
                progress_bar.update(1)

        return sample_list


class FQT(BaseClinical):
    """
    FQT class for the clinical tests
    """

    alias_name = "fqt"
    supported_tasks = ["question-answering", "text-generation"]

    @staticmethod
    def transform(sample_list: List[Sample], *args, **kwargs):
        """Transform method for the FQT class"""

        transformed_samples = []

        questions = [q.original_question for q in sample_list if isinstance(q, QASample)]

        # randomly select a question and swap it with another question

        for sample in sample_list:
            sample.category = "clinical"
            sample.test_type = "fqt"
            if (
                sample.original_question is None
                or sample.original_context is None
                or len(sample.original_question) < 2
            ):
                continue
            if isinstance(sample, QASample):
                selected = random.choice(questions)
                if selected == sample.original_question:
                    selected = random.choice(questions)
                sample.original_question = selected
                sample.expected_results = kwargs.get(
                    "expected_results", "None of the above"
                )
                sample.perturbed_context = ""
                sample.perturbed_question = ""

            transformed_samples.append(sample)

        return transformed_samples

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Run method for the FQT class"""

        progress_bar = kwargs.get("progress_bar", False)

        for sample in sample_list:
            if sample.state != "done":
                original_text_input = build_qa_input(
                    context=sample.original_context,
                    question=sample.original_question,
                    # options=(sample.options if sample.options else ""),
                )
                prompt = build_qa_prompt(
                    original_text_input, "default_question_answering_prompt", **kwargs
                )
                sample.actual_results = model(original_text_input, prompt=prompt)
                sample.state = "done"
            if progress_bar:
                progress_bar.update(1)

        return sample_list


class AMEGA(BaseClinical):
    """
    AMEGA class for the clinical tests
    """

    alias_name = "amega"
    supported_tasks = ["question-answering", "text-generation"]

    @staticmethod
    def transform(sample_list: List[Sample], *args, **kwargs):
        """Transform method for the AMEGA class"""
        # Sample Class
        from langtest.utils.custom_types.sample import AMEGASample

        eval_model = kwargs.get("eval_model", "gpt-4o-mini")
        no_of_cases = kwargs.get("no_of_cases", 20)
        case_ids = kwargs.get("no_of_cases", [1, 2, 3, 4, 5])  # range from 1 to 20

        if isinstance(no_of_cases, int) and 0 < no_of_cases:
            # limit the number of cases to 20
            no_of_cases = min(no_of_cases, 20)

            # generate the case ids
            case_ids = list(range(1, no_of_cases + 1))

        sample = AMEGASample()
        sample.case_ids = case_ids
        sample.eval_model = eval_model
        sample.category = "clinical"
        sample.test_type = "amega"

        return [sample]

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Run method for the AMEGA class"""

        sample = sample_list[0]

        progress_bar = kwargs.get("progress_bar", False)

        results = AMEGA.generate_responses(model, sample, *args, **kwargs)

        sample.actual_results = results

        progress_bar.update(1)

        return [sample]

    @staticmethod
    def generate_responses(model: ModelAPI, sample, *args, **kwargs):
        from langtest.transform.utils import DataRetriever, ResponseGenerator
        from tqdm.auto import tqdm

        model_name = model.model.model if hasattr(model.model, "model") else model.model

        # parameters from sample_list
        tqdm_case_ids = tqdm(
            sample.case_ids,
            desc="AMEGA Benchmark",
            unit="cases",
            position=1,
        )

        data_retriever = DataRetriever()
        generator = ResponseGenerator(model.model)
        results = []
        for case_id in tqdm_case_ids:
            response_list, reask_list = generator.generate_all_responses(
                data_retriever, case_id
            )

            case_eval_df = AMEGA.evaluate_responses(
                case_id,
                data_retriever,
                response_list,
                reask_list,
                model_name,
            )

            results.append(case_eval_df)

        return results

    @staticmethod
    def evaluate_responses(
        case_id: int,
        data_retriever,
        response_list: List[str],
        reask_list: List[str],
        generator_model_name_or_path: str,
        eval_model: str = "gpt-4o-mini",
    ):

        from langtest.transform.utils import ResponseEvaluator

        evaluator = ResponseEvaluator(
            model=eval_model,
            case_id=case_id,
            generator_model_name_or_path=generator_model_name_or_path,
            generator_type="chat",
        )

        results = evaluator.evaluate_responses(
            data_retriever, responses=response_list, reask_responses=reask_list
        )

        return evaluator.aggregate_results(data_retriever, results)


class MedFuzz(BaseClinical):
    alias_name = "medfuzz"
    supported_tasks = ["question-answering", "text-generation"]

    @staticmethod
    def transform(sample_list: List[Sample], *args, **kwargs):
        # return super().transform(*args, **kwargs)
        from langtest.transform.utils import AttackerLLM, TargetLLM
        from langtest.utils.custom_types.sample import MedFuzzSample
        from tqdm.auto import tqdm

        try:
            attacker_model_info = kwargs.get("attacker_llm", None)
            if attacker_model_info is not None:
                task = TaskManager("question-answering")
                model = task.model(
                    model_path=attacker_model_info["model"],
                    model_hub=attacker_model_info["hub"],
                    model_type=attacker_model_info["type"],
                )
            else:
                from textwrap import dedent

                error_message = dedent(
                    """
                    Attack model information is not provided in Configuration. Please provide the attack model information.
                    {
                        "medfuzz": {
                            "attacker_llm": {
                                "model": "<model_name>",
                                "hub": "<model_hub>",
                                "type": "<chat | completion>"
                            }
                        }
                    }
                """
                ).strip()

                raise ValueError(error_message)

            # model = task.model(model=kwargs)

            samples = tqdm(
                sample_list,
                desc="Transforming the samples",
                unit="samples",
                position=1,
            )

            transformed_samples = []
            for sample in samples:
                # llms

                llm_attacker = AttackerLLM(model)
                llm_target = TargetLLM(model)

                # sample
                med_sample = MedFuzzSample(
                    **sample.model_dump(exclude_none=True, exclude_unset=True)
                )
                med_sample.test_type = "medfuzz"
                med_sample.category = "clinical"

                if med_sample.options not in [None, ""]:
                    med_sample.original_question = (
                        f"{med_sample.original_question}\n{med_sample.options}"
                    )
                    med_sample.options = None

                # ot = llm_target.process_user_text(f"{med_sample.original_question}\n{med_sample.options}")
                ot = llm_target.process_user_text(med_sample.original_question)

                # generate the attack plan
                llm_attacker.generate_attack_plan(
                    benchmark_item=med_sample.original_question,
                    correct_answer="".join(med_sample.expected_results),
                    reasoning=ot["reasoning"],
                    confidence=ot["confidence_scores"],
                )

                # med_sample.perturbed_context = llm_attacker.generate_modified_question(
                #     med_sample.original_question
                # )
                med_sample.perturbed_question = llm_attacker.generate_modified_question(
                    med_sample.original_question
                )

                med_sample.expected_results = "".join(
                    map(str, med_sample.expected_results)
                )[:1]

                transformed_samples.append(med_sample)

            return transformed_samples
        except Exception:
            import traceback

            traceback.print_exc()
            raise

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        # return super().run(*args, **kwargs)
        from langtest.transform.utils import TargetLLM

        progress_bar = kwargs.get("progress_bar", False)

        for sample in sample_list:
            if sample.state != "done":
                target_llm = TargetLLM(model)

                response = target_llm.process_user_text(sample.perturbed_question)

                sample.actual_results = response.get("final_answer", "")

                # del
                del target_llm

                sample.state = "done"

            if progress_bar:
                progress_bar.update(1)

        return sample_list

    @staticmethod
    def ollama_model_client(model, messages):
        from ollama import Client

        client = Client()

        res = client.chat(
            model=model,
            messages=messages,
            options={
                "temperature": 0.9,
            },
        )
        return res.message.content

    @staticmethod
    def openai_model_client(model, messages):
        import openai

        client = openai.Client()

        res = (
            client.chat.completions.create(model=model, messages=messages)
            .choices[0]
            .message.content
        )
        return res


class ClinicalNoteSummary(BaseClinical):
    """
    ClinicalSummary class for the clinical tests
    """

    alias_name = "clinical_note_summary"
    supported_tasks = ["summarization"]

    @staticmethod
    def transform(sample_list: List[Sample], *args, **kwargs):
        """Transform method for the ClinicalSummary class"""
        from langtest.utils.custom_types.sample import DialogueToSummarySample

        transformed_samples = []

        # load the dataset
        dataset_path = kwargs.get("dataset_path", None)
        num_samples = kwargs.get("num_samples", 0)
        dialogue_col = kwargs.get("dialogue_col", "dialogue")
        ground_truth_col = kwargs.get("ground_truth_col", "ground_truth")
        threshold = min(kwargs.get("threshold", 5), 10)

        if dataset_path is None:
            raise ValueError("Dataset path is not provided.")

        if dataset_path == "mts-dialog":
            dataset = ClinicalNoteSummary.mts_dialog()
        elif dataset_path in ("aci-dialog", "aci_bench"):
            dataset = ClinicalNoteSummary.aci_dialog()
        else:
            # based on file extension load the dataset using pandas
            import pandas as pd

            file_extension = os.path.splitext(dataset_path)[1].lstrip(".")

            dataset = getattr(pd, f"read_{file_extension}")(dataset_path).to_dict(
                orient="records"
            )

        if num_samples > 0:
            dataset = random.sample(dataset, num_samples)

        for each_row in dataset:
            sample = DialogueToSummarySample()
            sample.dialogue = each_row[dialogue_col]
            sample.dataset_name = dataset_path
            if ground_truth_col in each_row:
                sample.expected_results = each_row[ground_truth_col]

            sample.category = "clinical"
            sample.test_type = "clinical_note_summary"

            # set threshold for the sample
            sample.threshold = threshold

            # append to transformed_samples
            transformed_samples.append(sample)

        return transformed_samples

    @staticmethod
    def mts_dialog():
        """MTS Dialog class for the clinical tests"""
        import pandas as pd

        # read dataset from csv file
        dataset_path = (
            importlib_resources.files("langtest")
            / "data"
            / "MTSDialog"
            / "validation.csv"
        )
        df = pd.read_csv(dataset_path)
        df = df.dropna()
        df["ground_truth"] = df.apply(
            lambda x: x["section_header"] + " Section: \n" + x["section_text"], axis=1
        )

        return df[["dialogue", "ground_truth"]].to_dict(orient="records")

    @staticmethod
    def aci_dialog():
        """ACI Dialog class for the clinical tests"""
        import pandas as pd

        # read dataset from csv file
        dataset_path = (
            importlib_resources.files("langtest")
            / "data"
            / "ACIBench"
            / "aci_bench_test_1.json"
        )
        df = pd.read_json(dataset_path, orient="records")
        df = df.dropna()
        df.rename({"src": "dialogue", "tgt": "ground_truth"}, axis=1, inplace=True)

        return df[["dialogue", "ground_truth"]].to_dict(orient="records")

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Run method for the ClinicalSummary class"""

        progress_bar = kwargs.get("progress_bar", False)

        model_type: Literal["chat", "completion"] = model.kwargs.get("model_type", "chat")

        for sample in sample_list:
            if sample.state != "done":
                if model_type == "chat":
                    messages = [
                        {
                            "role": "system",
                            "content": ClinicalNoteSummary.get_instructions(sample),
                        },
                        {"role": "user", "content": sample.dialogue},
                    ]
                else:
                    messages = f"## Instructions:\n{ClinicalNoteSummary.get_instructions(sample)}\n## Dialogue\n{sample.dialogue}"

                sample.actual_results = model.model.invoke(messages).content
                # sample.actual_results = model(original_text_input, prompt=prompt)
                sample.state = "done"
            if progress_bar:
                progress_bar.update(1)

        return sample_list

    @classmethod
    def get_instructions(cls, sample: Sample) -> str:
        """Get MTS Dialog dataset"""
        if sample.dataset_name == "mts-dialog":
            # Extract the heading from md text '##'
            section_header = sample.expected_results.split("\n")[0]
            sections = f"{section_header}"
            return CLINICALNOTE_SUMMARY_INSTRUCTIONS.format(sections_info=sections)
        else:
            sections = (
                "four sections:\n\n"
                "1. HISTORY OF PRESENT ILLNESS\n"
                "2. PHYSICAL EXAM\n"
                "3. RESULTS\n"
                "4. ASSESSMENT AND PLANn\n"
            )
            return CLINICALNOTE_SUMMARY_INSTRUCTIONS.format(sections_info=sections)


class MentalHealth(BaseClinical):
    """
    MentalHealth class for the clinical tests
    """

    alias_name = "mental_health"
    supported_tasks = ["question-answering", "text-generation"]

    @staticmethod
    def transform(sampe_list: List[Sample] = [], *args, **kwargs):
        """Transform method for the MentalHealth class"""
        from datasets import load_dataset
        from langtest.utils.custom_types.sample import SimplePrompt

        sample_size = kwargs.get("sample_size", 50)
        include_ground_truth = kwargs.get("include_ground_truth", True)

        df = load_dataset("ShenLab/MentalChat16K", split="train").to_pandas()

        transformed_samples = []

        for each_row in df.iloc[:sample_size].to_dict(orient="records"):
            sample = SimplePrompt()
            sample.prompt = each_row["input"]
            if include_ground_truth:
                sample.expected_results = each_row["output"]

            transformed_samples.append(sample)
            sample.category = "clinical"
            sample.test_type = "mental_health"

        return transformed_samples

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Run method for the MentalHealth class"""

        progress_bar = kwargs.get("progress_bar", False)

        model_type: Literal["chat", "completion"] = model.kwargs.get("model_type", "chat")

        instructions = (
            "You are a helpful mental health counselling assistant, "
            "please answer the mental health questions based on the patient's description."
            "The assistant gives helpful, comprehensive, and appropriate answers to the user's questions."
        )

        for sample in sample_list:
            if sample.state != "done":
                if model_type == "chat":
                    messages = [
                        {
                            "role": "system",
                            "content": instructions,
                        },
                        {"role": "user", "content": sample.prompt},
                    ]
                else:
                    messages = f"## Instructions:\n{instructions}\n## Patient Query: \n{sample.prompt}"

                sample.actual_results = model.model.invoke(messages).content
                # sample.actual_results = model(original_text_input, prompt=prompt)
                sample.state = "done"
            if progress_bar:
                progress_bar.update(1)

        return sample_list
