import logging
from typing import Any, List, Optional
import importlib.util
from ..errors import Errors, Warnings
from langtest.utils.lib_manager import try_import_lib
import os
import random
from typing import Tuple, Dict


class GatedRepoAccessError(Exception):
    """
    Exception raised when there is an attempt to access a gated Hugging Face repository without proper authorization.
    """

    pass


def get_model_n_tokenizer(model_name):
    """
    Load a pre-trained model and tokenizer from Hugging Face Model Hub.

    Args:
        model_name (str): The name or identifier of the pre-trained model.

    Returns:
        Tuple: A tuple containing the loaded model and tokenizer.
            - model: The loaded pre-trained model.
            - tokenizer: The tokenizer associated with the model.

    Raises:
        GatedRepoAccessError: If there is an attempt to access a gated repository without proper authorization.
    """
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )
    except ImportError:
        raise ValueError(Errors.E085())

    login_with_token()

    try:
        # Try loading the model as AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name)

    except ValueError:
        # Try loading the model as AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    except OSError:
        raise GatedRepoAccessError(Errors.E071(model_name=model_name))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def login_with_token():
    """
    Log in to the Hugging Face Hub using the provided API token.

    This function checks if the 'HUGGINGFACEHUB_API_TOKEN' environment variable is set.
    If the token is found and the 'huggingface_hub' module is installed, it imports the 'login' function
    and logs in using the provided API token.

    """
    if "HUGGINGFACEHUB_API_TOKEN" in os.environ:
        try:
            from huggingface_hub import login

            login(os.environ["HUGGINGFACEHUB_API_TOKEN"])
        except ImportError:
            pass


def clean_input(example: str) -> str:
    """
    Clean and format input example.

    Args:
        example (str): The input example to be cleaned.

    Returns:
        str: The cleaned and formatted input example.
    """
    example = example.replace('"', "")
    example = example.replace("\n", "")
    example = example.strip()
    example = f'"{example}"'
    return example


def build_dataset(
    dataset_name: str,
    dataset_subset: str,
    label_name: str,
    text_fields: List[str],
    natural_language_labels: List[str],
) -> Tuple[Dict[str, str], Dict[str, str], bool]:
    """
    Uses inputted dataset details to build dictionaries of train/test values.

    Args:
        dataset_name (str): The name of the dataset.
        dataset_subset (str): The name of the dataset subset.
        label_name (str): The name of the label.
        text_fields (List[str]): The list of text fields.
        natural_language_labels (List[str]): The list of natural language labels.

    Returns:
        Tuple[Dict[str, str], Dict[str, str], bool]: A tuple containing train and test dictionaries and a boolean indicating the presence of validation data.
    """

    LIB_NAME = "datasets"
    if try_import_lib(LIB_NAME):
        dataset_module = importlib.import_module(LIB_NAME)
        load_dataset = getattr(dataset_module, "load_dataset")
    else:
        raise ModuleNotFoundError(Errors.E023(LIB_NAME=LIB_NAME))

    """Uses inputted dataset details to build dictionary of train/test values."""
    if not dataset_subset:
        dataset_dict = load_dataset(dataset_name)
    else:
        dataset_dict = load_dataset(dataset_name, name=dataset_subset)

    train_dict, test_dict = {}, {}
    has_validation = False

    if "validation" in dataset_dict:
        train_data, test_data = dataset_dict["train"], dataset_dict["validation"]
        has_validation = True
    elif "test" in dataset_dict:
        train_data, test_data = dataset_dict["train"], dataset_dict["test"]
    else:
        temp = list(dataset_dict["train"])
        random.shuffle(temp)
        num_test_examples = int(0.2 * len(temp))
        train_data, test_data = temp[:-num_test_examples], temp[-num_test_examples:]

    for data, dict_ in zip([train_data, test_data], [train_dict, test_dict]):
        for i in range(len(data)):
            text = " and ".join([clean_input(data[i][x]) for x in text_fields])
            label = data[i][label_name]
            dict_[text] = natural_language_labels[label]

    return train_dict, test_dict, has_validation


class HuggingFacePipeline:
    """HuggingFace Pipeline API.

    To use, you should have the ``transformers`` python package installed.

    Only supports `text-generation`, `text2text-generation` and `summarization` for now.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        task: Optional[str] = None,
        device: Optional[int] = -1,
        pipeline: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Construct the pipeline object from model_id and task."""
        from transformers import Pipeline

        login_with_token()

        self.model_id = model_id
        self.pipeline: Pipeline = None
        self.model_type = kwargs.get("model_type", None)
        self.chat_template = kwargs.get("chat_template", None)
        if pipeline:
            self.pipeline = pipeline
        else:
            self.pipeline = self._initialize_pipeline(model_id, task, device, **kwargs)

    def _initialize_pipeline(
        self, model_id: str, task: str, device: Optional[int], **kwargs: Any
    ) -> Any:
        """
        Construct the pipeline object from model_id and task.

        Parameters:
        - model_id (str): The Hugging Face model identifier or path.
        - task (str): The specific task for which the pipeline is intended.
        - device (int, optional): The device on which the model should run (-1 for CPU, 0 or greater for GPU).
        - **kwargs (Any): Additional keyword arguments passed to the pipeline initialization.

        Returns:
        - Any: The initialized Hugging Face pipeline.
        """
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ValueError(Errors.E085())

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # remove the unnecessary kwargs
        kwargs.pop("model_type", None)

        # Set the pad_token_id for the tokenizer
        tokenizer.pad_token_id = tokenizer.eos_token_id

        try:
            if task == "text-generation":
                model = AutoModelForCausalLM.from_pretrained(model_id)
            elif task in ("text2text-generation", "summarization"):
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            else:
                raise ValueError(Errors.E086(task=task))
        except ImportError as e:
            raise ValueError(Errors.E087(task=task)) from e

        if (
            getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_loaded_in_8bit", False)
        ) and device is not None:
            logging.warning(Warnings.W015(device=device))
            device = None

        if device is not None and importlib.util.find_spec("torch") is not None:
            import torch

            cuda_device_count = torch.cuda.device_count()
            if device < -1 or (device >= cuda_device_count):
                raise ValueError(
                    Errors.E088(device=device, cuda_device_count=cuda_device_count)
                )
            if device < 0 and cuda_device_count > 0:
                logging.warning(Warnings.W016(cuda_device_count=cuda_device_count))

        # renove the unnecessary kwargs
        kwargs.pop("chat_template", None)

        return hf_pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            device=device,
            **kwargs,
        )

    def _generate(self, prompts: List[str]) -> List[str]:
        """
        Generate text based on the provided prompts using the initialized pipeline.

        Parameters:
        - prompts (List[str]): A list of text prompts for text generation.

        Returns:
        - List[str]: A list of generated texts corresponding to the input prompts.
        """
        text_generations: List[str] = []

        for prompt in prompts:
            if (
                self.pipeline.tokenizer.chat_template is None
                and self.model_type == "chat"
            ):
                self.pipeline.tokenizer.chat_template = self.chat_template

            # return_full_text = False is available in transformers>=4.11.0
            if self.pipeline.task == "text-generation":
                response = self.pipeline(prompt, return_full_text=False)
            else:
                response = self.pipeline(prompt)

            if isinstance(response, list):
                response = response[0]

            output_key = (
                f"{self.pipeline.return_name}_text"
                if hasattr(self.pipeline, "return_name")
                else "generated_text"
            )
            text = response[output_key]

            text_generations.append(text)

        return text_generations


class ParaphraseModel:
    _instance = None
    _device = "cpu"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        if not self._initialized:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "humarin/chatgpt_paraphraser_on_T5_base"
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                "humarin/chatgpt_paraphraser_on_T5_base"
            ).to(self._device)
            self._initialized = True


def paraphrase(
    text: str,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=1,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128,
    device="cpu",
):
    # Use ParaphraseModel singleton if model and tokenizer not provided
    # if model is None or tokenizer is None:
    paraphrase_model = ParaphraseModel()
    model = paraphrase_model.model
    tokenizer = paraphrase_model.tokenizer
    device = paraphrase_model._device

    input_ids = tokenizer(
        f"paraphrase: {text}",
        return_tensors="pt",
        padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)

    outputs = model.generate(
        input_ids,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        max_length=max_length,
        diversity_penalty=diversity_penalty,
        trust_remote_code=True,
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res
