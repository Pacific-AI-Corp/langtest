import asyncio
from collections import defaultdict
import random
from langtest.logger import logger as logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, TypedDict

from langtest.modelhandler.modelhandler import ModelAPI
from langtest.transform.base import ITests, TestFactory
from langtest.transform.utils import filter_unique_samples, get_substitution_names
from .constants import (
    asian_names,
    black_names,
    country_economic_dict,
    female_pronouns,
    hispanic_names,
    inter_racial_names,
    male_pronouns,
    native_american_names,
    neutral_pronouns,
    religion_wise_names,
    white_names,
)
from ..utils.custom_types import Sample, Span, Transformation
from langtest.errors import Errors, Warnings


class BiasTestFactory(ITests):
    """
    A class for performing bias tests on a given dataset.
    """

    alias_name = "bias"

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

        if "replace_to_male_pronouns" in self.tests:
            self.tests["replace_to_male_pronouns"]["parameters"] = {}
            self.tests["replace_to_male_pronouns"]["parameters"][
                "pronouns_to_substitute"
            ] = [
                item for sublist in list(female_pronouns.values()) for item in sublist
            ] + [
                item for sublist in list(neutral_pronouns.values()) for item in sublist
            ]
            self.tests["replace_to_male_pronouns"]["parameters"]["pronoun_type"] = "male"

        if "replace_to_female_pronouns" in self.tests:
            self.tests["replace_to_female_pronouns"]["parameters"] = {}
            self.tests["replace_to_female_pronouns"]["parameters"][
                "pronouns_to_substitute"
            ] = [item for sublist in list(male_pronouns.values()) for item in sublist] + [
                item for sublist in list(neutral_pronouns.values()) for item in sublist
            ]
            self.tests["replace_to_female_pronouns"]["parameters"][
                "pronoun_type"
            ] = "female"

        if "replace_to_neutral_pronouns" in self.tests:
            self.tests["replace_to_neutral_pronouns"]["parameters"] = {}
            self.tests["replace_to_neutral_pronouns"]["parameters"][
                "pronouns_to_substitute"
            ] = [
                item for sublist in list(female_pronouns.values()) for item in sublist
            ] + [
                item for sublist in list(male_pronouns.values()) for item in sublist
            ]
            self.tests["replace_to_neutral_pronouns"]["parameters"][
                "pronoun_type"
            ] = "neutral"

        for income_level in [
            "Low-income",
            "Lower-middle-income",
            "Upper-middle-income",
            "High-income",
        ]:
            economic_level = income_level.replace("-", "_").lower()
            if f"replace_to_{economic_level}_country" in self.tests:
                countries_to_exclude = [
                    v for k, v in country_economic_dict.items() if k != income_level
                ]
                self.tests[f"replace_to_{economic_level}_country"]["parameters"] = {}
                self.tests[f"replace_to_{economic_level}_country"]["parameters"][
                    "country_names_to_substitute"
                ] = get_substitution_names(countries_to_exclude)
                self.tests[f"replace_to_{economic_level}_country"]["parameters"][
                    "chosen_country_names"
                ] = country_economic_dict[income_level]

        for religion in religion_wise_names.keys():
            if f"replace_to_{religion.lower()}_names" in self.tests:
                religion_to_exclude = [
                    v for k, v in religion_wise_names.items() if k != religion
                ]
                self.tests[f"replace_to_{religion.lower()}_names"]["parameters"] = {}
                self.tests[f"replace_to_{religion.lower()}_names"]["parameters"][
                    "names_to_substitute"
                ] = get_substitution_names(religion_to_exclude)
                self.tests[f"replace_to_{religion.lower()}_names"]["parameters"][
                    "chosen_names"
                ] = religion_wise_names[religion]

        ethnicity_first_names = {
            "white": white_names["first_names"],
            "black": black_names["first_names"],
            "hispanic": hispanic_names["first_names"],
            "asian": asian_names["first_names"],
        }
        for ethnicity in ["white", "black", "hispanic", "asian"]:
            test_key = f"replace_to_{ethnicity}_firstnames"
            if test_key in self.tests:
                self.tests[test_key]["parameters"] = {}
                self.tests[test_key]["parameters"] = {
                    "names_to_substitute": sum(
                        [
                            ethnicity_first_names[e]
                            for e in ethnicity_first_names
                            if e != ethnicity
                        ],
                        [],
                    ),
                    "chosen_ethnicity_names": ethnicity_first_names[ethnicity],
                }

        ethnicity_last_names = {
            "white": white_names["last_names"],
            "black": black_names["last_names"],
            "hispanic": hispanic_names["last_names"],
            "asian": asian_names["last_names"],
            "native_american": native_american_names["last_names"],
            "inter_racial": inter_racial_names["last_names"],
        }
        for ethnicity in [
            "white",
            "black",
            "hispanic",
            "asian",
            "native_american",
            "inter_racial",
        ]:
            test_key = f"replace_to_{ethnicity}_lastnames"
            if test_key in self.tests:
                self.tests[test_key]["parameters"] = {}
                self.tests[test_key]["parameters"] = {
                    "names_to_substitute": sum(
                        [
                            ethnicity_last_names[e]
                            for e in ethnicity_last_names
                            if e != ethnicity
                        ],
                        [],
                    ),
                    "chosen_ethnicity_names": ethnicity_last_names[ethnicity],
                }

    def transform(self) -> List[Sample]:
        """
        Runs the bias test and returns the resulting `Sample` objects.

        Returns:
            List[Sample]
                A list of `Sample` objects representing the resulting dataset after running the bias test.
        """
        all_samples = []
        no_transformation_applied_tests = {}
        for test_name, params in self.tests.items():
            data_handler_copy = [x.model_copy() for x in self._data_handler]

            test_func = self.supported_tests[test_name].transform
            
            if (
                TestFactory.task in ("question-answering", "summarization")
            ):
                _ = [
                    (sample.transform(
                        test_func,
                        params.get("parameters", {}),
                        prob=params.pop("prob", 1.0)
                    )
                    if hasattr(sample, "transform")
                    else sample
                    )
                    for sample in data_handler_copy
                ]
                transformed_samples = data_handler_copy
            
            else:

                transformed_samples = test_func(
                    data_handler_copy, **params.get("parameters", {})
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
    def available_tests() -> Dict[str, type["BaseBias"]]:
        """
        Get a dictionary of all available tests, with their names as keys and their corresponding classes as values.

        Returns:
            Dict: A dictionary of test names and classes.

        """

        # tests = {
        #     j: i
        #     for i in BaseBias.__subclasses__()
        #     for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        # }
        return BaseBias.test_types


class BaseBias(ABC):
    """Abstract base class for implementing bias measures.

    Attributes:
        alias_name (str): A name or list of names that identify the bias measure.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented bias measure.
    """

    test_types = defaultdict(lambda: BaseBias)

    alias_name = None
    supported_tasks = [
        "ner",
        "text-classification",
        "question-answering",
        "summarization",
    ]

    # Config Hint for the bias tests
    TestConfig = TypedDict("TestConfig", min_pass_rate=float)

    @abstractmethod
    def transform(self, sample_list: List[Sample], *args, **kwargs) -> List[Sample]:
        """Abstract method that implements the bias measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.

        Returns:
            List[Sample]: The transformed data based on the implemented bias measure.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs) -> List[Sample]:
        """Abstract method that implements the bias measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for the bias measure.

        Returns:
            List[Sample]: The transformed data based on the implemented bias measure.

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
        """Abstract method that implements the creation of an asyncio task for the bias measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for the bias measure.

        Returns:
            asyncio.Task: The asyncio task for the bias measure.
        """
        created_task = asyncio.create_task(cls.run(sample_list, model, **kwargs))
        return created_task

    def __init_subclass__(cls, *args, **kwargs) -> None:
        """Register the bias measure class in the test_types dictionary."""
        aliases = cls.alias_name if isinstance(cls.alias_name, list) else [cls.alias_name]
        for alias in aliases:
            BaseBias.test_types[alias] = cls


class GenderPronounBias(BaseBias):
    """Class for gender biases"""

    alias_name = [
        "replace_to_male_pronouns",
        "replace_to_female_pronouns",
        "replace_to_neutral_pronouns",
    ]

    @staticmethod
    def transform(
        sample_list: List[Sample], pronouns_to_substitute: List[str], pronoun_type: str, *args, **kwargs
    ) -> List[Sample]:
        """Replace pronouns to check the gender bias

        Args:
            sample_list (List[Sample]): List of sentences to apply perturbation.
            pronouns_to_substitute (List[str]): list of pronouns that need to be substituted.
            pronoun_type (str): replacing pronoun type string ('male', 'female' or 'neutral')

        Returns:
            List[Sample]: List of sentences with replaced pronouns
        """

        def gender_pronoun_bias(string, pronouns_to_substitute, pronoun_type):
            transformations = []
            replaced_string = string
            pattern = (
                r"\b(?:"
                + "|".join(re.escape(name) for name in pronouns_to_substitute)
                + r")(?!\w)"
            )
            tokens_to_substitute = re.findall(pattern, string, flags=re.IGNORECASE)

            for replace_token in tokens_to_substitute:
                if pronoun_type == "female":
                    combined_dict = {
                        k: male_pronouns[k] + neutral_pronouns[k]
                        for k in male_pronouns.keys()
                    }
                    chosen_dict = female_pronouns
                elif pronoun_type == "male":
                    combined_dict = {
                        k: female_pronouns[k] + neutral_pronouns[k]
                        for k in female_pronouns.keys()
                    }
                    chosen_dict = male_pronouns
                elif pronoun_type == "neutral":
                    combined_dict = {
                        k: female_pronouns[k] + male_pronouns[k]
                        for k in female_pronouns.keys()
                    }
                    chosen_dict = neutral_pronouns

                for key, value in combined_dict.items():
                    if replace_token.lower() in value:
                        type_of_pronoun = str(key)
                        break

                chosen_token = random.choice(chosen_dict[type_of_pronoun])
                if replace_token.endswith("."):
                    replace_token = replace_token.strip(replace_token[-1])
                regex = r"\b{}\b".format(replace_token)
                diff_len = len(chosen_token) - len(replace_token)
                nb_occurrences = len(re.findall(regex, replaced_string))
                for c in range(nb_occurrences):
                    span = re.search(regex, replaced_string)
                    replaced_string = re.sub(
                        regex, chosen_token, replaced_string, count=1
                    )
                    if not isinstance(sample, str) and sample.task in ("ner", "text-classification"):
                        transformations.append(
                            Transformation(
                                original_span=Span(
                                    start=span.start(), end=span.end(), word=replace_token
                                ),
                                new_span=Span(
                                    start=span.start(),
                                    end=span.end() + diff_len,
                                    word=chosen_token,
                                ),
                                ignore=False,
                            )
                        )

            return replaced_string, transformations

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx], _ = gender_pronoun_bias(
                    sample, pronouns_to_substitute, pronoun_type
                )
            else:
                sample.test_case, transformations = gender_pronoun_bias(
                    sample.original, pronouns_to_substitute, pronoun_type
                )
                if sample.task in ("ner", "text-classification"):
                    sample.transformations = transformations
                sample.category = "bias"
        return sample_list


class CountryEconomicBias(BaseBias):
    """Class for economical biases on countries"""

    alias_name = [
        "replace_to_high_income_country",
        "replace_to_low_income_country",
        "replace_to_upper_middle_income_country",
        "replace_to_lower_middle_income_country",
    ]

    @staticmethod
    def transform(
        sample_list: List[Sample],
        country_names_to_substitute: List[str],
        chosen_country_names: List[str],
        *args, **kwargs
    ) -> List[Sample]:
        """Replace country names to check the ethnicity bias

        Args:
            sample_list (List[Sample]): List of sentences to apply perturbation.
            country_names_to_substitute (List[str]): list of country names that need to be substituted.
            chosen_country_names (List[str]): list of country names to replace with.

        Returns:
            List[Sample]: List of sentences with replaced names
        """

        def country_economic_bias(
            string, country_names_to_substitute, chosen_country_names
        ):
            transformations = []
            replaced_string = string
            pattern = (
                r"\b(?:"
                + "|".join(re.escape(name) for name in country_names_to_substitute)
                + r")(?!\w)"
            )
            tokens_to_substitute = re.findall(pattern, string, flags=re.IGNORECASE)

            for replace_token in tokens_to_substitute:
                chosen_token = random.choice(chosen_country_names)
                if replace_token.endswith("."):
                    replace_token = replace_token.strip(replace_token[-1])
                regex = r"\b{}\b".format(replace_token)
                diff_len = len(chosen_token) - len(replace_token)
                nb_occurrences = len(re.findall(regex, replaced_string))
                for c in range(nb_occurrences):
                    span = re.search(regex, replaced_string)
                    replaced_string = re.sub(
                        regex, chosen_token, replaced_string, count=1
                    )
                    if not isinstance(sample, str) and sample.task in ("ner", "text-classification"):
                        transformations.append(
                            Transformation(
                                original_span=Span(
                                    start=span.start(), end=span.end(), word=replace_token
                                ),
                                new_span=Span(
                                    start=span.start(),
                                    end=span.end() + diff_len,
                                    word=chosen_token,
                                ),
                                ignore=False,
                            )
                        )

            return replaced_string, transformations

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx], _ = country_economic_bias(
                    sample, country_names_to_substitute, chosen_country_names
                )
            else:
                sample.test_case, transformations = country_economic_bias(
                    sample.original, country_names_to_substitute, chosen_country_names
                )
                if sample.task in ("ner", "text-classification"):
                    sample.transformations = transformations
                sample.category = "bias"

        return sample_list


class EthnicityNameBias(BaseBias):
    """Class for ethnicity biases"""

    alias_name = [
        "replace_to_white_firstnames",
        "replace_to_black_firstnames",
        "replace_to_hispanic_firstnames",
        "replace_to_asian_firstnames",
        "replace_to_white_lastnames",
        "replace_to_black_lastnames",
        "replace_to_hispanic_lastnames",
        "replace_to_asian_lastnames",
        "replace_to_native_american_lastnames",
        "replace_to_inter_racial_lastnames",
    ]

    @staticmethod
    def transform(
        sample_list: List[Sample],
        names_to_substitute: List[str],
        chosen_ethnicity_names: List[str],
        *args, **kwargs
    ) -> List[Sample]:
        """Replace names to check the ethnicity bias

        Ethnicity Dataset Curated from the United States Census Bureau surveys

        Args:
            sample_list (List[Sample]): List of sentences to apply perturbation.
            names_to_substitute (List[str]): list of ethnicity names that need to be substituted.
            chosen_ethnicity_names (List[str]): list of ethnicity names to replace with.

        Returns:
            List[Sample]: List of sentences with replaced names
        """

        def ethnicity_names_bias(string, names_to_substitutes, chosen_names):
            transformations = []
            replaced_string = string
            pattern = (
                r"\b(?:"
                + "|".join(re.escape(name) for name in names_to_substitutes)
                + r")(?!\w)"
            )
            tokens_to_substitute = re.findall(pattern, string, flags=re.IGNORECASE)

            for replace_token in tokens_to_substitute:
                chosen_token = random.choice(chosen_names)
                if replace_token.endswith("."):
                    replace_token = replace_token.strip(replace_token[-1])
                regex = r"\b{}\b".format(replace_token)
                diff_len = len(chosen_token) - len(replace_token)
                nb_occurrences = len(re.findall(regex, replaced_string))
                for c in range(nb_occurrences):
                    span = re.search(regex, replaced_string)
                    replaced_string = re.sub(
                        regex, chosen_token, replaced_string, count=1
                    )
                    transformations.append(
                        Transformation(
                            original_span=Span(
                                start=span.start(), end=span.end(), word=replace_token
                            ),
                            new_span=Span(
                                start=span.start(),
                                end=span.end() + diff_len,
                                word=chosen_token,
                            ),
                            ignore=False,
                        )
                    )

            return replaced_string, transformations

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx], _ = ethnicity_names_bias(
                    sample, names_to_substitute, chosen_ethnicity_names
                )
            else:
                sample.test_case, transformations = ethnicity_names_bias(
                    sample.original, names_to_substitute, chosen_ethnicity_names
                )
                if sample.task in ("ner", "text-classification"):
                    sample.transformations = transformations
                sample.category = "bias"

        return sample_list


class ReligionBias(BaseBias):
    """Class for religious biases"""

    alias_name = [
        "replace_to_muslim_names",
        "replace_to_hindu_names",
        "replace_to_christian_names",
        "replace_to_sikh_names",
        "replace_to_jain_names",
        "replace_to_parsi_names",
        "replace_to_buddhist_names",
    ]

    @staticmethod
    def transform(
        sample_list: List[Sample], names_to_substitute: List[str], chosen_names: List[str], *args, **kwargs
    ) -> List[Sample]:
        """Replace  names to check the religion bias

        Args:
            sample_list (List[Sample]): List of sentences to apply perturbation.
            names_to_substitute (List[str]): list of names that need to be substituted.
            chosen_names (List[str]): list of names to replace with.

        Returns:
            List[Sample]: List of sentences with replaced names
        """

        def religion_bias(string, names_to_substitutes, chosen_names):
            transformations = []
            replaced_string = string
            pattern = (
                r"\b(?:"
                + "|".join(re.escape(name) for name in names_to_substitutes)
                + r")(?!\w)"
            )
            tokens_to_substitute = re.findall(pattern, string, flags=re.IGNORECASE)

            for replace_token in tokens_to_substitute:
                chosen_token = random.choice(chosen_names)
                if replace_token.endswith("."):
                    replace_token = replace_token.strip(replace_token[-1])
                regex = r"\b{}\b".format(replace_token)
                diff_len = len(chosen_token) - len(replace_token)
                nb_occurrences = len(re.findall(regex, replaced_string))
                for c in range(nb_occurrences):
                    span = re.search(regex, replaced_string)
                    replaced_string = re.sub(
                        regex, chosen_token, replaced_string, count=1
                    )
                    transformations.append(
                        Transformation(
                            original_span=Span(
                                start=span.start(), end=span.end(), word=replace_token
                            ),
                            new_span=Span(
                                start=span.start(),
                                end=span.end() + diff_len,
                                word=chosen_token,
                            ),
                            ignore=False,
                        )
                    )

            return replaced_string, transformations

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx], _ = religion_bias(
                    sample, names_to_substitute, chosen_names
                )
            else:
                sample.test_case, transformations = religion_bias(
                    sample.original, names_to_substitute, chosen_names
                )
                if sample.task in ("ner", "text-classification"):
                    sample.transformations = transformations
                sample.category = "bias"
        return sample_list
