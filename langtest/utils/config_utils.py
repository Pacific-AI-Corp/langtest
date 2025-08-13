from typing import Any, Dict, List

# from pkg_resources import resource_filename
from importlib import resources


LLM_DEFAULTS_CONFIG = {
    "azure-openai": str(
        resources.files("langtest").joinpath(
            "data/config/QA_summarization_azure_config.yml"
        )
    ),
    "huggingface": str(
        resources.files("langtest").joinpath(
            "data/config/QA_summarization_huggingface_config.yml"
        )
    ),
    "default": str(
        resources.files("langtest").joinpath("data/config/QA_summarization_config.yml")
    ),
}

DEFAULTS_CONFIG: Dict[str, Any] = {
    "question-answering": LLM_DEFAULTS_CONFIG,
    "summarization": LLM_DEFAULTS_CONFIG,
    "ideology": str(
        resources.files("langtest").joinpath("data/config/political_config.yml")
    ),
    "toxicity": str(
        resources.files("langtest").joinpath("data/config/toxicity_config.yml")
    ),
    "clinical": str(
        resources.files("langtest").joinpath("data/config/clinical_config.yml")
    ),
    "legal": str(resources.files("langtest").joinpath("data/config/legal_config.yml")),
    "crows-pairs": str(
        resources.files("langtest").joinpath("data/config/crows_pairs_config.yml")
    ),
    "stereoset": str(
        resources.files("langtest").joinpath("data/config/stereoset_config.yml")
    ),
    "security": str(
        resources.files("langtest").joinpath("data/config/security_config.yml")
    ),
    "disinformation": str(
        resources.files("langtest").joinpath("data/config/disinformation_config.yml")
    ),
    "factuality": str(
        resources.files("langtest").joinpath("data/config/factuality_config.yml")
    ),
    "sycophancy": str(
        resources.files("langtest").joinpath("data/config/sycophancy_config.yml")
    ),
    "sensitivity": {
        "huggingface": str(
            resources.files("langtest").joinpath(
                "data/config/sensitivity_huggingface_config.yml"
            )
        ),
        "default": str(
            resources.files("langtest").joinpath("data/config/sensitivity_config.yml")
        ),
    },
    "translation": {
        "default": str(
            resources.files("langtest").joinpath(
                "data/config/translation_transformers_config.yml"
            )
        ),
        "johnsnowlabs": str(
            resources.files("langtest").joinpath(
                "data/config/translation_johnsnowlabs_config.yml"
            )
        ),
    },
    "wino-bias": {
        "huggingface": str(
            resources.files("langtest").joinpath(
                "data/config/wino_huggingface_config.yml"
            )
        ),
        "default": str(
            resources.files("langtest").joinpath("data/config/wino_llm_config.yml")
        ),
    },
}

BENCHMARK_DATASETS_DICT = {
    "ASDiv": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "ASDiv", "split": "test-tiny"},
    },
    "BBQ": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "BBQ", "split": "test-tiny"},
    },
    "Bigbench": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {
            "data_source": "Bigbench",
            "subset": "Abstract-narrative-understanding",
            "split": "test-tiny",
        },
    },
    "BoolQ": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "BoolQ", "split": "test-tiny"},
    },
    "CommonsenseQA": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "CommonsenseQA", "split": "validation-tiny"},
    },
    "FIQA": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "Fiqa", "split": "test-tiny"},
    },
    "HellaSwag": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "HellaSwag", "split": "test-tiny"},
    },
    "Consumer-Contracts": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "Consumer-Contracts", "split": "test"},
    },
    "Contracts": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "Contracts", "split": "test"},
    },
    "Privacy-Policy": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "Privacy-Policy", "split": "test"},
    },
    "LogiQA": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "LogiQA", "split": "test-tiny"},
    },
    "MMLU": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "MMLU", "split": "test-tiny"},
    },
    "NarrativeQA": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "NarrativeQA", "split": "test-tiny"},
    },
    "NQ-open": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "NQ-open", "split": "test-tiny"},
    },
    "OpenBookQA": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "OpenBookQA", "split": "test-tiny"},
    },
    "PIQA": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "PIQA", "split": "test-tiny"},
    },
    "Quac": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "Quac", "split": "test-tiny"},
    },
    "SIQA": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "SIQA", "split": "test-tiny"},
    },
    "TruthfulQA": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "TruthfulQA", "split": "test-tiny"},
    },
    "XSum": {
        "task": "summarization",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "XSum", "split": "test-tiny"},
    },
    "MultiLexSum": {
        "task": "summarization",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {"data_source": "MultiLexSum", "split": "test-tiny"},
    },
    "MedMCQA": {
        "task": "question-answering",
        "model": {"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
        "data": {
            "data_source": "MedMCQA",
            "subset": "MedMCQA-Test",
            "split": "Radiology",
        },
    },
    "MedQA": {
        "task": "question-answering",
        "model": {"model": "mistralai/Mistral-7B-Instruct-v0.1", "hub": "huggingface"},
        "data": {"data_source": "MedQA", "split": "test"},
        "config": {
            "evaluation": {
                "metric": "string_distance",
                "distance": "jaro",
                "threshold": 0.1,
            },
            "tests": {
                "defaults": {"min_pass_rate": 0.65},
                "robustness": {
                    "add_ocr_typo": {"min_pass_rate": 0.66},
                    "dyslexia_word_swap": {"min_pass_rate": 0.60},
                },
            },
        },
    },
    "PubMedQA": {
        "task": "question-answering",
        "model": {"model": "j2-jumbo-instruct", "hub": "ai21"},
        "data": {"data_source": "PubMedQA", "split": "pqaa"},
        "config": {
            "evaluation": {
                "metric": "string_distance",
                "distance": "jaro",
                "threshold": 0.1,
            },
            "tests": {
                "defaults": {"min_pass_rate": 0.65},
                "robustness": {
                    "add_ocr_typo": {"min_pass_rate": 0.66},
                    "dyslexia_word_swap": {"min_pass_rate": 0.60},
                },
            },
        },
    },
}


class BenchmarkDatasets:
    def __init__(self, task: str, dataset_name: str):
        self.__dataset_name = dataset_name
        self.__task = task

    @classmethod
    def get_dataset_dict(cls, dataset_name: str, task: str) -> dict:
        """Get the benchmark dataset configuration for the given dataset name."""
        dataset_config = BENCHMARK_DATASETS_DICT.get(dataset_name, None)

        if dataset_config is None:
            raise ValueError(
                f"Dataset {dataset_name} not found in the benchmark datasets list - {cls.get_datasets()}."
            )

        return dataset_config.get("data", None)

    @classmethod
    def get_datasets(cls) -> List[str]:
        """Get the list of benchmark datasets available in the library."""
        return list(BENCHMARK_DATASETS_DICT.keys())

    @property
    def dataset_name(self) -> str:
        return self.__dataset_name

    @property
    def task(self) -> str:
        return self.__task
