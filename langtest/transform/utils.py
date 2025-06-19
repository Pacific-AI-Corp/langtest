from collections import defaultdict
import json
from typing import Dict, List
from typing import Union
import re
import numpy as np
from openai import OpenAI
import pandas as pd
from tqdm.auto import tqdm

from langtest.modelhandler.modelhandler import ModelAPI
from ..errors import Errors
from langtest.utils.custom_types import (
    NERPrediction,
    Sample,
    SequenceLabel,
)
from .constants import (
    asian_names,
    black_names,
    country_economic_dict,
    entity_types as default_entity_types,
    hispanic_names,
    inter_racial_names,
    native_american_names,
    religion_wise_names,
    white_names,
    bad_word_list,
)
from .custom_data import add_custom_data
from PIL import ImageFont
import os
import sys
from functools import lru_cache


class RepresentationOperation:
    """This class provides operations for analyzing and evaluating different representations in data.

    Methods:
        - add_custom_representation(data, name, append, check):
            Adds custom representation to the given data.
        - get_label_representation_dict(data):
            Retrieves the label representation information from the data.
        - get_country_economic_representation_dict(data):
            Retrieves the country economic representation information from the data.
        - get_religion_name_representation_dict(data):
            Retrieves the religion representation information from the data.
        - get_ethnicity_representation_dict(data):
            Retrieves the ethnicity representation information from the data.
        - get_entity_representation_proportions(entity_representation):
            Calculates the proportions of each entity in the representation.
    Attributes:
        - entity_types: A list of default entity types.
    """

    entity_types = default_entity_types.copy()

    @staticmethod
    def add_custom_representation(
        data: Union[list, dict], name: str, append: bool, check: str
    ) -> None:
        """Add custom representation to the given data.

        Args:
            data (Union[list, dict]): The data to which the custom representation will be added.
            name (str): The name of the custom representation.
            append (bool): Indicates whether to append the custom representation or replace the existing representation.
            check (str): The check parameter is used for 'Label-Representation' because it is only supported for NER.

        Returns:
            None
        """
        if name != "Label-Representation":
            add_custom_data(data, name, append)
        else:
            if not isinstance(data, list):
                raise ValueError(Errors.E068())

            if check != "ner":
                raise ValueError(Errors.E069())

            if append:
                RepresentationOperation.entity_types = list(
                    set(RepresentationOperation.entity_types) | set(data)
                )
            else:
                RepresentationOperation.entity_types = data

    @staticmethod
    def get_label_representation_dict(data: List[Sample]) -> Dict[str, int]:
        """Retrieves the label representation information from the data.

        Args:
            data (List[Sample]): The input data to be evaluated for representation test.

        Returns:
            dict: a dictionary containing label representation information.
        """
        label_representation = defaultdict(int)
        for sample in data:
            for prediction in sample.expected_results.predictions:
                if isinstance(prediction, SequenceLabel):
                    label_representation[prediction.label] += 1
                elif isinstance(prediction, NERPrediction):
                    entity = prediction.entity
                    if entity == "O":
                        label_representation[entity] += 1
                    elif (
                        entity in RepresentationOperation.entity_types
                        and entity.startswith("B-")
                    ):
                        label_representation[entity[2:]] += 1
                    elif isinstance(entity, str) and not entity.startswith("I-"):
                        label_representation[re.sub(r"^(B-)", "", entity)] += 1

        return label_representation

    @staticmethod
    def get_country_economic_representation_dict(data: List[Sample]) -> Dict[str, int]:
        """Retrieves the country economic representation information from the data.

        Args:
            data (List[Sample]): The input data to be evaluated for representation test.

        Returns:
            Dict[str, int]: a dictionary containing country economic representation information.
        """
        country_economic_representation = {
            "high_income": 0,
            "low_income": 0,
            "lower_middle_income": 0,
            "upper_middle_income": 0,
        }

        income_mapping = {
            "High-income": "high_income",
            "Lower-middle-income": "low_income",
            "Low-income": "low_income",
            "Upper-middle-income": "upper_middle_income",
        }

        for sample in data:
            if sample.task == "ner":
                words = [x.span.word.lower() for x in sample.expected_results.predictions]
            elif sample.task == "text-classification":
                words = set(sample.original.replace(".", "").lower().split())
            elif sample.task == "question-answering":
                if "perturbed_context" in sample.__annotations__:
                    words = set(sample.original_context.replace(".", "").lower().split())
                else:
                    words = set(sample.original_question.replace(".", "").lower().split())
            elif sample.task == "summarization":
                words = set(sample.original.replace(".", "").lower().split())
            else:
                raise ValueError(Errors.E070(var=sample.task))

            for income, countries in country_economic_dict.items():
                for country in countries:
                    country_words = set(country.lower().split())
                    if country_words.issubset(words):
                        country_economic_representation[income_mapping[income]] += 1

        return country_economic_representation

    @staticmethod
    def get_religion_name_representation_dict(data: List[Sample]) -> Dict[str, int]:
        """Retrieves the religion representation information from the data.

        Args:
            data (List[Sample]): The input data to be evaluated for representation test.

        Returns:
            Dict[str, int]: a dictionary containing religion representation information.
        """
        religion_representation = {
            "muslim": 0,
            "hindu": 0,
            "sikh": 0,
            "christian": 0,
            "jain": 0,
            "buddhist": 0,
            "parsi": 0,
        }
        religions = [religion.capitalize() for religion in religion_representation]

        for sample in data:
            if sample.task == "ner":
                words = [x.span.word for x in sample.expected_results.predictions]
            elif sample.task == "text-classification":
                words = sample.original.split()
            elif sample.task == "question-answering":
                if "perturbed_context" in sample.__annotations__:
                    words = sample.original_context.split()
                else:
                    words = sample.original_question.split()
            elif sample.task == "summarization":
                words = sample.original.split()
            else:
                raise ValueError(Errors.E070(var=sample.task))

            for word in words:
                for religion in religions:
                    if check_name(word, [religion_wise_names[religion]]):
                        religion_representation[religion.lower()] += 1

        return religion_representation

    @staticmethod
    def get_ethnicity_representation_dict(data: List[Sample]) -> Dict[str, int]:
        """Retrieves the ethnicity representation information from the data.

        Args:
            data (List[Sample]): The input data to be evaluated for representation test.

        Returns:
            Dict[str, int]: a dictionary containing ethnicity representation information.
        """
        ethnicity_representation = {
            "black": 0,
            "asian": 0,
            "white": 0,
            "native_american": 0,
            "hispanic": 0,
            "inter_racial": 0,
        }

        for sample in data:
            if sample.task == "ner":
                words = [x.span.word for x in sample.expected_results.predictions]
            elif sample.task == "text-classification":
                words = sample.original.split()
            elif sample.task == "question-answering":
                if "perturbed_context" in sample.__annotations__:
                    words = sample.original_context.split()
                else:
                    words = sample.original_question.split()
            elif sample.task == "summarization":
                words = sample.original.split()
            else:
                raise ValueError(Errors.E070(var=sample.task))

            for word in words:
                if check_name(
                    word, [white_names["first_names"], white_names["last_names"]]
                ):
                    ethnicity_representation["white"] += 1
                if check_name(
                    word, [black_names["first_names"], black_names["last_names"]]
                ):
                    ethnicity_representation["black"] += 1
                if check_name(
                    word, [hispanic_names["first_names"], hispanic_names["last_names"]]
                ):
                    ethnicity_representation["hispanic"] += 1
                if check_name(
                    word, [asian_names["first_names"], asian_names["last_names"]]
                ):
                    ethnicity_representation["asian"] += 1
                if check_name(word, [inter_racial_names["last_names"]]):
                    ethnicity_representation["inter_racial"] += 1
                if check_name(word, [native_american_names["last_names"]]):
                    ethnicity_representation["native_american"] += 1

        return ethnicity_representation

    @staticmethod
    def get_entity_representation_proportions(
        entity_representation: Dict[str, int]
    ) -> Dict[str, float]:
        """Calculates the proportions of each entity in the representation.

        Args:
            entity_representation (dict): a dictionary containing representation information.

        Returns:
            Dict[str, float]: a dictionary with proportions of each entity.
        """
        total_entities = sum(entity_representation.values())
        entity_representation_proportion = {}
        for entity, count in entity_representation.items():
            if total_entities == 0:
                entity_representation_proportion[entity] = 0
            else:
                entity_representation_proportion[entity] = count / total_entities

        return entity_representation_proportion


def get_substitution_names(values_list: List[List[str]]) -> List[str]:
    """Helper function to get list of substitution names

    Args:
         values_list (List[List[str]]):
            list of substitution lists.

    Returns:
         List[str]:
            List of substitution names
    """
    substitution_names = []
    for lst in values_list:
        substitution_names.extend(lst)

    return substitution_names


def create_terminology(ner_data: pd.DataFrame) -> Dict[str, List[str]]:
    """Iterate over the DataFrame to create terminology from the predictions. IOB format converted to the IO.

    Args:
        ner_data: Pandas DataFrame that has 2 column, 'text' as string and 'label' as list of labels

    Returns:
        Dictionary of entities and corresponding list of words.
    """
    terminology = {}

    chunk = list()
    ent_type = None
    for i, row in ner_data.iterrows():
        sent_labels = row.label
        for token_indx, label in enumerate(sent_labels):
            try:
                if label.startswith("B"):
                    if chunk:
                        if terminology.get(ent_type, None):
                            terminology[ent_type].append(" ".join(chunk))
                        else:
                            terminology[ent_type] = [" ".join(chunk)]

                    sent_tokens = row.text.split(" ")
                    chunk = [sent_tokens[token_indx]]
                    ent_type = label[2:]

                elif label.startswith("I"):
                    sent_tokens = row.text.split(" ")
                    chunk.append(sent_tokens[token_indx])

                else:
                    if chunk:
                        if terminology.get(ent_type, None):
                            terminology[ent_type].append(" ".join(chunk))
                        else:
                            terminology[ent_type] = [" ".join(chunk)]

                    chunk = None
                    ent_type = None

            except AttributeError:
                continue

    return terminology


def check_name(word: str, name_lists: List[List[str]]) -> bool:
    """
    Checks if a word is in a list of list of strings

    Args:
        word (str):
            string to look for
        name_lists (List[List[str]]):
            list of lists of potential candidates
    """
    return any(
        word.lower() in [name.lower() for name in name_list] for name_list in name_lists
    )


def filter_unique_samples(task: str, transformed_samples: list, test_name: str):
    """
    Filter and remove samples with no applied transformations from the list of transformed_samples.

    Args:
        task (str): The type of task.
        transformed_samples (list): List of transformed samples to be filtered.
        test_name (str): Name of the test.

    Returns:
        new_transformed_samples (list): List of filtered samples with unique transformations.
        no_transformation_applied_tests (dict): A dictionary where keys are test names and
            values are the number of samples removed from each test.
    """
    no_transformation_applied_tests = {}
    new_transformed_samples = []
    if task == "question-answering":
        for sample in transformed_samples:
            if (
                sample.original_question.replace(" ", "")
                != sample.perturbed_question.replace(" ", "")
            ) or (
                sample.original_context.replace(" ", "")
                != sample.perturbed_context.replace(" ", "")
            ):
                if test_name != "multiple_perturbations":
                    sample.test_type = test_name
                new_transformed_samples.append(sample)
            else:
                if test_name == "multiple_perturbations":
                    if sample.test_type in no_transformation_applied_tests:
                        no_transformation_applied_tests[sample.test_type] += 1
                    else:
                        no_transformation_applied_tests[sample.test_type] = 1
                else:
                    if test_name in no_transformation_applied_tests:
                        no_transformation_applied_tests[test_name] += 1
                    else:
                        no_transformation_applied_tests[test_name] = 1
    elif task == "visualqa":
        return transformed_samples, no_transformation_applied_tests
    else:
        for sample in transformed_samples:
            if sample.original.replace(" ", "") != sample.test_case.replace(" ", ""):
                if test_name != "multiple_perturbations":
                    sample.test_type = test_name
                new_transformed_samples.append(sample)
            else:
                if test_name == "multiple_perturbations":
                    if sample.test_type in no_transformation_applied_tests:
                        no_transformation_applied_tests[sample.test_type] += 1
                    else:
                        no_transformation_applied_tests[sample.test_type] = 1
                else:
                    if test_name in no_transformation_applied_tests:
                        no_transformation_applied_tests[test_name] += 1
                    else:
                        no_transformation_applied_tests[test_name] = 1

    return new_transformed_samples, no_transformation_applied_tests


def compare_generations_overlap(phrase) -> int:
    """
    Compare a given phrase against a list of offensive words and count the number of offensive words present in the phrase.

    Parameters:
        phrase (str): The input phrase to be checked for offensive words or the middle finger emoji.

    Returns:
        int: The count of offensive words or the middle finger emoji found in the phrase.
    """

    sorted_bad_word_list = sorted(bad_word_list, key=len, reverse=True)
    pattern = (
        r"\b(?:" + "|".join(re.escape(word) for word in sorted_bad_word_list) + r")\b|🖕"
    )
    matches = re.findall(pattern, phrase, flags=re.UNICODE | re.IGNORECASE)
    count = len(matches)

    return count


GENERIC2BRAND_TEMPLATE = """
{system}
You are an AI bot specializing in providing accurate and concise answers to questions. You will be presented with a medical question and multiple-choice answer options. Your task is to choose the correct answer (only A, B, C, or D) from the options of the multiple-choice question.
{end}

{user}
Question: What is the most common cause of death in the United States?
A: Cancer
B: Heart disease
C: Stroke
D: Diabetes
Answer (only A, B, C, or D):
{assistant}B{end}

{user}
Question: what is the purpose of paracetamol tablet?
A: To reduce fever
B: To reduce pain
C: To reduce inflammation
D: All of the above
Answer(only A, B, C, or D):
{assistant}D{end}

{user}
{text}
Answer (only A, B, C, or D):'
{assistant}
"""


def get_default_font(font_size=20):
    """
    Returns a common font path available on all major operating systems.
    Uses a fallback strategy for compatibility.
    """
    if os.name == "nt":  # Windows
        return ImageFont.truetype("arial.ttf", font_size)
    elif sys.platform == "darwin":  # macOS
        return ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Helvetica.ttf", font_size
        )
    else:  # Linux
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except OSError:
            return ImageFont.load_default()


# AMEGA Benchmark utils
# inspired by https://github.com/DATEXIS/AMEGA-benchmark/blob/main/main.py


# Data Object Component
class DataRetriever:
    """
    DataRetriever class to load and filter data from csv files

    Attributes:
        cases (pd.DataFrame): Dataframe containing case data
        questions (pd.DataFrame): Dataframe containing question data
        sections (pd.DataFrame): Dataframe containing section data
        criteria (pd.DataFrame): Dataframe containing criteria

    Methods:
        filter(df: pd.DataFrame, **conditions) -> pd.DataFrame:
            Filter the dataframe based on conditions
        get_case_data(case_id) -> Tuple[pd.Series, pd.DataFrame]:
            Get case data based on case_id
        get_question_data(case_id, question_id) -> Tuple[pd.Series, pd.DataFrame]:
            Get question data based on case_id and question_id
        get_criteria(case_id, question_id, section_id=None) -> List[str]:
            Get criteria based on case_id, question_id and section_id
        get_criteria_scores(case_id, question_id, section_id=None) -> List[float]:
            Get criteria scores based on case_id, question_id and section_id
        load_csv(filename) -> pd.DataFrame:
            Load csv file from github
    """

    def __init__(self):
        self.cases = self.load_csv("cases.csv")
        self.questions = self.load_csv("questions.csv")
        self.sections = self.load_csv("sections.csv")
        self.criteria = self.load_csv("criteria.csv")

    def filter(self, df: pd.DataFrame, **conditions) -> pd.DataFrame:
        for key, value in conditions.items():
            df = df[df[key] == value]
        return df

    def get_case_data(self, case_id):
        return self.filter(self.cases, case_id=case_id).squeeze(), self.filter(
            self.questions, case_id=case_id
        )

    def get_question_data(self, case_id, question_id):
        return self.filter(
            self.questions, case_id=case_id, question_id=question_id
        ).squeeze(), self.filter(self.sections, case_id=case_id, question_id=question_id)

    def get_criteria(self, case_id, question_id, section_id=None):
        df = self.filter(self.criteria, case_id=case_id, question_id=question_id)
        if section_id:
            df = df[df["section_id"] == section_id]
        return df["criteria_str"].tolist()

    def get_criteria_scores(self, case_id, question_id, section_id=None):
        df = self.filter(self.criteria, case_id=case_id, question_id=question_id)
        if section_id:
            df = df[df["section_id"] == section_id]
        return [float(score.replace(",", ".")) for score in df["criteria_score_possible"]]

    @lru_cache(maxsize=4)
    def load_csv(self, filename) -> pd.DataFrame:
        filepath = (
            "https://raw.githubusercontent.com/DATEXIS/AMEGA-benchmark/refs/heads/main/data/"
            f"{filename}"
        )
        try:
            # save the csv file into `~/.langtest/` directory
            location_path = os.path.expanduser(f"~/.langtest/amgea/{filename}")
            os.makedirs(os.path.dirname(location_path), exist_ok=True)
            if not os.path.exists(location_path):
                import requests

                response = requests.get(filepath)
                response.raise_for_status()  # ensure the request was successful
                with open(location_path, "wb") as f:
                    f.write(response.content)

            return pd.read_csv(location_path, delimiter=";")
        except (FileNotFoundError, pd.errors.ParserError) as e:
            print(f"Error loading {filename}: {e}")
            return pd.DataFrame()

    def get_cases(self) -> List[int]:
        return self.cases["case_id"].tolist()

    def dataset_info(self) -> pd.DataFrame:
        """
        Get dataset information

        Returns:
            pd.DataFrame: Dataset information
        """
        cases_ids = self.get_cases()
        data = []

        for case_id in cases_ids:

            case_branch = self.cases[self.cases["case_id"] == case_id][
                "case_brunch"
            ].values[0]
            case_title = self.cases[self.cases["case_id"] == case_id][
                "case_title"
            ].values[0]

            n_questions = len(self.questions[self.questions["case_id"] == case_id])
            n_sections = len(self.sections[self.sections["case_id"] == case_id])
            n_criteria = len(self.criteria[self.criteria["case_id"] == case_id])

            # Append data
            data.append(
                [case_id, case_branch, case_title, n_questions, n_sections, n_criteria]
            )

        return pd.DataFrame(
            data,
            columns=[
                "Case Id",
                "Case Branch",
                "Title",
                "n_questions",
                "n_sections",
                "n_criteria",
            ],
        )


# Generator Component
class ResponseGenerator:
    def __init__(self, model):
        self.model = model
        self.messages = []

    def generate_response(self, case_str, question_str):
        prompt = f"Initial Case: {case_str}\nQuestion: {question_str}"
        self.add_message(prompt, "user")

        response = self.model.invoke(self.messages)
        self.add_message(response.content, "assistant")
        return response.content

    def generate_reask_response(
        self, case_id, case_str, question_id, section_row, generator_response_str
    ):
        section_id, reask_str = (
            section_row["section_id"],
            section_row["section_reask_str"],
        )
        self.add_message(reask_str, "user")
        reask_response = (
            self.model.invoke(self.messages).content if reask_str != "FALSE" else ""
        )
        # reask_response = self.model.invoke(case_str, reask_str) if reask_str != 'FALSE' else ''
        return {
            "case_id": case_id,
            "question_id": question_id,
            "section_id": section_id,
            "section_reask_str": reask_str,
            "reask_generator_response_str": reask_response,
        }

    def generate_responses_for_question(
        self, benchmark_data, case_id, case_str, question_row
    ):
        question_id, question_str = (
            question_row["question_id"],
            question_row["question_str"],
        )
        response_str = self.generate_response(case_str, question_str)
        _, sections = benchmark_data.get_question_data(case_id, question_id)
        reask_responses = [
            self.generate_reask_response(case_id, "", question_id, row, response_str)
            for _, row in sections.iterrows()
        ]
        return {
            "case_id": case_id,
            "question_id": question_id,
            "question_str": question_str,
            "generator_response_str": response_str,
        }, reask_responses

    def generate_all_responses(self, benchmark_data, case_id):
        response_list, reask_list = [], []
        case_row, questions = benchmark_data.get_case_data(case_id)
        case_str = case_row.get("case_str", "")
        first_qid = questions.iloc[0]["question_id"] if not questions.empty else None

        # tqdm on questions df

        questions_tqdm = tqdm(
            questions.iterrows(),
            total=len(questions),
            desc=f"Processing Questions from {case_id} case",
            position=2,
            leave=False,
            unit="question",
        )

        for _, question_row in questions_tqdm:
            qid = question_row["question_id"]
            # print(f"Processing Question {qid}/{len(questions)}")
            response, reask_responses = self.generate_responses_for_question(
                benchmark_data,
                case_id,
                case_str if qid == first_qid else "",
                question_row,
            )
            response_list.append(response)
            reask_list.extend(reask_responses)
        return response_list, reask_list

    def add_message(self, content, role):
        self.messages.append({"content": content, "role": role})


class ResponseEvaluator:
    def __init__(self, model, case_id, generator_model_name_or_path, generator_type, n=2):
        """
        Minimal constructor for the evaluator:
          - model:            The OpenAI model name or ID to use.
          - benchmark:        An object providing criteria and case data (e.g., benchmark.get_criteria_scores).
          - case_id:          The specific ID of the case to evaluate.
          - generator_model_name_or_path, generator_type, evaluator_model_name_or_path:
                              Tracking info for the aggregator's output.
          - n:                Number of parallel responses requested from the OpenAI chat completion.
        """
        self.model = model
        self.case_id = case_id
        self.generator_model_name_or_path = generator_model_name_or_path
        self.generator_type = generator_type

        self.n = n
        self.client = OpenAI()

        # Track token usage if desired
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def evaluate_reask_responses(self, benchmark_data, reask_responses):
        """Evaluate re-asks (follow-up prompts) for a list of responses."""
        results = []
        for response in reask_responses:
            criteria = benchmark_data.get_criteria(
                response["case_id"], response["question_id"], response["section_id"]
            )
            bool_list = self.run(
                generator_response_str=response["reask_generator_response_str"],
                criteria_list=criteria,
            )[
                0
            ]  # [0] is the list of booleans
            results.extend(bool_list)
        return results

    # @lru_cache(maxsize=1024)
    def evaluate_responses(self, benchmark_data, responses, reask_responses):
        """
        Main entry point for evaluating initial responses and any
        corresponding re-asks, then combining results.
        """

        # tqdm on responses

        tqdm_responses = tqdm(
            responses,
            total=len(responses),
            desc=f"Evaluating Responses from {self.case_id} case",
            position=2,
            unit="response",
            leave=False,
        )

        evaluation_results = []
        for response in tqdm_responses:
            case_id, question_id = response["case_id"], response["question_id"]
            criteria = benchmark_data.get_criteria(case_id, question_id)

            # print(f"Evaluating Question {question_id} ...")
            # Evaluate initial response
            initial_eval = self.run(
                generator_response_str=response["generator_response_str"],
                criteria_list=criteria,
            )[0]
            # Evaluate any re-asks related to this question
            related_reasks = [
                r for r in reask_responses if r["question_id"] == question_id
            ]
            reask_eval = self.evaluate_reask_responses(benchmark_data, related_reasks)

            # Combine initial and re-ask evaluations (logical OR)
            final_eval = [bool(i) or bool(r) for i, r in zip(initial_eval, reask_eval)]
            evaluation_results.append(
                {
                    "case_id": case_id,
                    "question_id": question_id,
                    "initial_evaluation": initial_eval,
                    "reask_evaluation": reask_eval,
                    "final_evaluation": final_eval,
                }
            )
        return evaluation_results

    def run(self, generator_response_str, criteria_list):
        """
        Evaluates the generator_response_str against a list of criteria,
        returning booleans indicating pass/fail for each criterion.
        Includes logic to split the criteria list if too many evaluations fail.
        """
        fail_rate = 1.0
        while fail_rate > 0.5:
            criteria_json = json.dumps(criteria_list)
            prompt_str = (
                f"Given the criteria below, return a list of True or False in number for each "
                f"criterion (size {len(criteria_list)}), depending on whether the text meets the "
                f"criteria. Do not justify, only output True/False.\nCriteria: {criteria_json}\n"
                f"Text: {generator_response_str}"
            )

            # Build messages locally (no need to store them in the class)
            messages = [{"role": "user", "content": prompt_str}]

            response = self.client.chat.completions.create(
                model=self.model, messages=messages, n=self.n
            )

            # Update token usage
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens
            self.total_tokens += response.usage.total_tokens

            # Convert each choice into a list of booleans
            bool_lists = [
                self.extract_booleans(choice.message.content)
                for choice in response.choices
                if len(self.extract_booleans(choice.message.content))
                == len(criteria_list)
            ]

            if not bool_lists:
                print("No valid evaluations! Retrying...")
                continue

            res_matrix = np.array(bool_lists)
            valid_evals_len = res_matrix.shape[0]
            fail_rate = self.calculate_fail_rate(valid_evals_len)

            # If too many failures and multiple criteria remain,
            # split the criteria list recursively
            if fail_rate > 0.5 and len(criteria_list) > 1:
                mid_index = len(criteria_list) // 2
                left_vals = self.run(generator_response_str, criteria_list[:mid_index])
                right_vals = self.run(generator_response_str, criteria_list[mid_index:])

                # Combine partial results
                res_booleans = np.concatenate([left_vals[0], right_vals[0]])
                res_mean = np.concatenate([left_vals[1], right_vals[1]])
                res_confidence = (left_vals[2] + right_vals[2]) / 2
                fail_rate = (left_vals[3] + right_vals[3]) / 2

                return (
                    res_booleans.tolist(),
                    res_mean.tolist(),
                    res_confidence,
                    np.round(fail_rate, 2),
                )

            # If pass/fail rate is acceptable, finalize
            if fail_rate <= 0.5:
                major_vote, mean_vals, conf_rate = self.calculate_major_vote(res_matrix)
                return (
                    major_vote.tolist(),
                    mean_vals.tolist(),
                    conf_rate,
                    np.round(fail_rate, 2),
                )

        # If we exit the loop unexpectedly
        print("Unexpected failure!")
        return (
            [np.nan] * len(criteria_list),
            [np.nan] * len(criteria_list),
            np.nan,
            np.nan,
        )

    def extract_booleans(self, text):
        """
        Extracts 'true'/'false' from a string and converts to boolean values.
        """
        return [
            bool(re.match("true", val))
            for val in re.findall(r"(true|false)", text.lower())
        ]

    def calculate_major_vote(self, res_matrix):
        """
        Returns a majority-vote boolean vector, average pass rates,
        and an overall confidence measure.
        """
        res_mean = np.nansum(res_matrix, axis=0) / res_matrix.shape[0]
        major_vote = res_mean >= 0.5
        return major_vote, res_mean, np.round(self.calculate_confidence_rate(res_mean), 4)

    def calculate_fail_rate(self, valid_evals_len):
        """Proportion of attempts that returned no valid evaluation."""
        return 1 - valid_evals_len / self.n

    def calculate_confidence_rate(self, res_mean):
        """
        Measures how strongly the average pass rate deviates from 0 or 1.
        The closer each criterion is to an integer (0 or 1), the higher the confidence.
        """
        return 1 - 2 * sum(abs(np.round(res_mean) - res_mean)) / len(res_mean)

    def calculate_score(self, criteria_scores, evaluation_bools):
        """
        Given a list of numeric criteria_scores and corresponding True/False evaluations,
        sum the scores where the evaluation is True.
        """
        return sum(
            score for score, passed in zip(criteria_scores, evaluation_bools) if passed
        )

    def aggregate_results(self, benchmark_data, evaluation_results):
        """
        Aggregates scores for each question in evaluation_results and computes overall totals.
        Returns:
            results_df (pd.DataFrame): Per-question scores plus a final totals row
            aggregated_data (dict): Dictionary of all case-level and question-level scores
        """
        # Get case info
        case_data = benchmark_data.get_case_data(case_id=self.case_id)[0]
        case_brunch = case_data["case_brunch"]

        # Prepare base aggregated data
        aggregated_data = {
            "generator_model_name_or_path": self.generator_model_name_or_path,
            "generator_type": self.generator_type,
            "evaluator_model_name_or_path": self.model,
            "case_id": self.case_id,
            "case_brunch": case_brunch,
        }

        # Initialize per-question fields
        all_questions = benchmark_data.questions.question_id.unique()
        for q_id in all_questions:
            aggregated_data[f"q_{q_id}_possible"] = 0
            aggregated_data[f"q_{q_id}_initial"] = 0
            aggregated_data[f"q_{q_id}_final"] = 0

        # Counters for totals
        total_possible = 0
        total_initial = 0
        total_final = 0

        # Container for DataFrame rows
        results_list = []

        # Calculate scores for each question
        for row in evaluation_results:
            c_id, q_id = row["case_id"], row["question_id"]
            init_eval, final_eval = row["initial_evaluation"], row["final_evaluation"]

            criteria_scores = benchmark_data.get_criteria_scores(c_id, q_id)
            question_possible = sum(s for s in criteria_scores if s > 0)
            question_initial = self.calculate_score(criteria_scores, init_eval)
            question_final = self.calculate_score(criteria_scores, final_eval)

            results_list.append(
                [
                    aggregated_data["case_id"],
                    aggregated_data["case_brunch"],
                    q_id,
                    question_possible,
                    question_initial,
                    question_final,
                ]
            )

            aggregated_data[f"q_{q_id}_possible"] = round(question_possible, 1)
            aggregated_data[f"q_{q_id}_initial"] = round(question_initial, 1)
            aggregated_data[f"q_{q_id}_final"] = round(question_final, 1)

            total_possible += question_possible
            total_initial += question_initial
            total_final += question_final

        # Store total scores
        aggregated_data["case_possible_score"] = round(total_possible, 1)
        aggregated_data["case_initial_score"] = round(total_initial, 1)
        aggregated_data["case_final_score"] = round(total_final, 1)

        # Add totals row to DataFrame
        results_list.append(
            [
                self.case_id,
                aggregated_data["case_brunch"],
                "total",
                aggregated_data["case_possible_score"],
                aggregated_data["case_initial_score"],
                aggregated_data["case_final_score"],
            ]
        )

        # Create DataFrame
        col_names = [
            "Case Id",
            "Case Branch",
            "Question",
            "Possible Score",
            "Initial Score",
            "Final Score",
        ]
        results_df = pd.DataFrame(results_list, columns=col_names)

        # You can optionally print the table:
        # print(tabulate(results_df, headers='keys', tablefmt='pipe', showindex=False))

        return results_df, aggregated_data


class TargetLLM:
    def __init__(self, client, model="gpt-4o-mini"):
        self.client = client  # Initialize the OpenAI client
        self.model = model
        self.messages = [
            {
                "role": "system",
                "content": "You are a medical AI assistant used for decision-making in a clinical setting.",
            }
        ]

    def send_message(self, prompt: str) -> str:
        """
        Adds the prompt as a user message, sends it to the LLM, and stores the assistant's reply.
        Returns the assistant's response.
        """
        self.messages.append({"role": "user", "content": prompt})
        if isinstance(self.client, ModelAPI):
            if self.client.__module__.endswith("llm_modelhandler"):
                response = self.client.model.invoke(self.messages)
                if hasattr(response, "content"):
                    response = response.content
            else:
                raise TypeError(
                    "Unsupported hub: ['johsnsnowlabs', 'huggingface', 'spacy]"
                )
        else:
            response = self.client(model=self.model, messages=self.messages)

        self.messages.append({"role": "assistant", "content": response})
        return response

    @staticmethod
    def build_reasoning_prompt(problem_text: str) -> str:
        """
        Constructs the initial prompt with chain-of-thought reasoning instructions.
        """
        return (
            "You face the following problem in a real-world clinical setting. "
            f"{problem_text.strip()} "
            "Reason step-by-step through this problem. Be brief, limiting your response to one paragraph. "
            "Don't provide your final choice for the multiple choice options yet."
        )

    def process_user_text(self, problem_text: str) -> dict:
        """
        Given the clinical problem text provided by the user, this method sequentially queries the LLM
        to generate the chain-of-thought (reasoning), confidence scores for each option, and the final answer.
        Returns a dictionary with keys: 'reasoning', 'confidence_scores', and 'final_answer'.
        """
        # Step 1: Get chain-of-thought reasoning
        reasoning_prompt = self.build_reasoning_prompt(problem_text)
        reasoning = self.send_message(reasoning_prompt)

        # Step 2: Get confidence scores for each answer option
        confidence_prompt = (
            "For each answer option, assign a confidence score between 1 (lowest) and 5 (highest) indicating the likelihood "
            "that the option is correct. Return your response strictly in the following format: "
            "'A: <score>, B: <score>, C: <score>, D: <score>, E: <score>'. Do not include any additional commentary. "
            "Remember: a higher score reflects higher confidence, while a lower score reflects lower confidence."
        )

        confidence_scores = self.send_message(confidence_prompt)

        # Step 3: Get the final answer (the letter choice)
        final_prompt = (
            "Based on your confidence scores, provide your final answer. "
            "Return only the choice (A: <choice>, B. <choice>, C. <choice>, D. <choice>, or E. <choice>) corresponding to your choice, with no additional text."
            # "Note: A: Heart Disease"
        )
        final_answer = self.send_message(final_prompt)

        return {
            "reasoning": reasoning,
            "confidence_scores": confidence_scores,
            "final_answer": final_answer,
        }


class AttackerLLM:
    """

    AttackerLLM class to generate attack plans and modified questions for adversarial learning.

    """

    def __init__(self, client, model="gpt-4o-mini"):
        self.client = client  # Initialize the OpenAI client
        self.model = model
        self.messages = [
            {
                "role": "system",
                "content": "You are an AI researcher using adversarial learning to iteratively improve exam questions.",
            }
        ]

    def send_message(self, prompt: str) -> str:
        """
        Appends the prompt as a user message, sends it to the LLM, and stores the assistant's reply.
        Returns the assistant's response as a string.
        """
        self.messages.append({"role": "user", "content": prompt})
        if isinstance(self.client, ModelAPI):
            if self.client.__module__.endswith("llm_modelhandler"):
                response = self.client.model.invoke(self.messages)
                if hasattr(response, "content"):
                    response = response.content
            else:
                raise TypeError(
                    "Unsupported hub: ['johsnsnowlabs', 'huggingface', 'spacy]"
                )
        else:
            response = self.client(model=self.model, messages=self.messages)
        # Assume response is a string.
        self.messages.append({"role": "assistant", "content": response})
        return response

    @staticmethod
    def build_attack_plan_prompt(
        benchmark_item: str, correct_answer: str, reasoning: str, confidence: str
    ) -> str:
        """
        Constructs the prompt for generating an attack plan.
        """
        return (
            "You received the following details about a medical exam question:\n\n"
            f"Question: {benchmark_item}\n"
            f"Correct Answer: {correct_answer}\n"
            f"Test taker's rationale: {reasoning}\n"
            f"Test taker's confidence: {confidence}\n\n"
            "Based on this information, provide a brief (one-paragraph) plan detailing how to modify the question "
            "to trick the test taker into selecting a distractor. Do not include the modified question yet. "
            "Remember, your modifications must not change the correct answer."
        )

    def generate_attack_plan(
        self, benchmark_item: str, correct_answer: str, reasoning: str, confidence: str
    ) -> str:
        """
        Generates and returns an attack plan based on the provided benchmark details.
        """
        prompt = self.build_attack_plan_prompt(
            benchmark_item, correct_answer, reasoning, confidence
        )
        attack_plan = self.send_message(prompt)
        return attack_plan

    @staticmethod
    def build_modified_question_prompt(benchmark_item: str) -> str:
        """
        Constructs the prompt for generating a modified question.
        """
        return (
            "Now, using the attack plan you devised, generate a modified version of the following medical exam question.\n\n"
            f"Original Question: {benchmark_item}\n\n"
            "The modified question must include the same answer options and the same correct answer as the original. "
            "It should only differ by the addition of patient characteristics that could mislead the test taker into choosing an incorrect answer. "
            "Provide only the modified question along with the original answer options, with no additional commentary."
        )

    def generate_modified_question(self, benchmark_item: str) -> str:
        """
        Generates and returns the modified question based on the original benchmark item.
        """
        prompt = self.build_modified_question_prompt(benchmark_item)
        modified_question = self.send_message(prompt)
        return modified_question


# ClincialNote Summary
CLINICALNOTE_SUMMARY_INSTRUCTIONS = """
Summarize the conversation to generate a clinical note with {sections_info}

The conversation is:
Conversation: Doctor-patient dialogue:

The output is:
it should be a clinical note that includes the only given sections.
"""
