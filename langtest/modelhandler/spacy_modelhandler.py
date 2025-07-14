from typing import List

import spacy
from spacy.tokens import Doc
from functools import lru_cache

from .modelhandler import ModelAPI
from ..utils.custom_types import NEROutput, NERPrediction, SequenceClassificationOutput

# from pkg_resources import resource_filename
from importlib_resources import files
from ..errors import Errors


class PretrainedModelForNER(ModelAPI):
    """SpaCy pretrained model for NER tasks

    Args:
        model: Pretrained SpaCy pipeline.
    """

    def __init__(self, model):
        """Constructor method

        Args:
            model: spacy model
        """
        annotation = getattr(model, "__call__").__annotations__
        assert annotation.get("return") and annotation["return"] is Doc, ValueError(
            Errors.E080(expected_type=Doc, returned_type=annotation.get("return", None))
        )

        self.model = model
        self.predict.cache_clear()

    @classmethod
    def load_model(cls, path: str):
        """Load and return SpaCy pipeline

        Args:
            path (str): name of path to model to load
        """
        try:
            return cls(spacy.load(path))
        except OSError:
            raise ValueError(Errors.E041(path=path))

    @lru_cache(maxsize=102400)
    def predict(self, text: str, *args, **kwargs) -> NEROutput:
        """Perform predictions on the input text.

        Args:
            text (str): Input text to perform NER on.
            kwargs: Additional keyword arguments.

        Keyword Args:
            group_entities (bool): Option to group entities.

        Returns:
            NEROutput: A list of named entities recognized in the input text.
        """
        doc = self.model(text)

        # if kwargs.get("group_entities"):
        return NEROutput(
            predictions=[
                NERPrediction.from_span(
                    entity=ent.label_,
                    word=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                )
                for ent in doc.ents
            ]
        )

    def __call__(self, text: str, *args, **kwargs) -> NEROutput:
        """Alias of the 'predict' method"""
        return self.predict(text=text)

    def predict_raw(self, text: str) -> List[str]:
        """Predict a list of labels in form of strings.

        Args:
            text (str): Input text to perform NER on.

        Returns:
            List[str]: A list of named entities recognized in the input text.
        """
        doc = self.model(text)
        return [
            f"{token.ent_iob_}-{token.ent_type_}" if token.ent_type_ else token.ent_iob_
            for token in doc
        ]


class PretrainedModelForTextClassification(ModelAPI):
    """SpaCy pretrained model for text classification tasks

    Args:
        model: Pretrained SpaCy pipeline.
    """

    def __init__(self, model):
        """Constructor method

        Args:
            model: pretrained SpaCy pipeline
        """
        annotation = getattr(model, "__call__").__annotations__
        assert annotation.get("return") and annotation["return"] is Doc, ValueError(
            Errors.E080(expected_type=Doc, returned_type=annotation.get("return", None))
        )

        self.model = model

    @property
    def labels(self):
        """Return classification labels of SpaCy model."""
        return self.model.get_pipe("textcat").labels

    @classmethod
    def load_model(cls, path: str):
        """Load and return SpaCy pipeline

        Args:
            path (str): name of path to model to load
        """
        try:
            if path == "textcat_imdb":
                path = files("langtest").joinpath("data/textcat_imdb")
            return cls(spacy.load(path))
        except OSError:
            raise ValueError(Errors.E041(path=path))

    @lru_cache(maxsize=102400)
    def predict(
        self, text: str, return_all_scores: bool = False, *args, **kwargs
    ) -> SequenceClassificationOutput:
        """Perform text classification predictions on the input text.

        Args:
            text (str): Input text to classify.
            return_all_scores (bool): Option to return score for all labels.

        Returns:
            SequenceClassificationOutput: Text classification predictions from the input text.
        """
        output = self.model(text).cats
        if not return_all_scores:
            label = max(output, key=output.get)
            output = [{"label": label, "score": output[label]}]
        else:
            output = [{"label": key, "score": value} for key, value in output.items()]

        return SequenceClassificationOutput(text=text, predictions=output)

    def predict_raw(self, text: str) -> List[str]:
        """Perform classification predictions on input text.

        Args:
            text (str): Input text to classify.

        Returns:
            List[str]: Predictions of the model.
        """
        output = self.model(text).cats
        return [max(output, key=output.get)]

    def __call__(
        self, text: str, return_all_scores: bool = False, *args, **kwargs
    ) -> SequenceClassificationOutput:
        """Alias of the 'predict' method"""
        return self.predict(text=text, return_all_scores=return_all_scores, **kwargs)


class PretrainedModelForTranslation(ModelAPI):
    """SpaCy pretrained model for translation tasks

    Args:
        model: Pretrained SpaCy pipeline.
    """

    def __init__(self, model):
        """Constructor method

        Args:
            model: Pretrained SpaCy pipeline.
        """
        annotation = getattr(model, "__call__").__annotations__
        assert annotation.get("return") and annotation["return"] is Doc, ValueError(
            Errors.E080(expected_type=Doc, returned_type=annotation.get("return", None))
        )

        self.model = model

    @classmethod
    def load_model(cls, path: str):
        """Load and return SpaCy pipeline

        Args:
            path (str): name of path to model to load
        """
        try:
            return cls(spacy.load(path))
        except OSError:
            raise ValueError(Errors.E041(path=path))

    @lru_cache(maxsize=102400)
    def predict(self, text: str, *args, **kwargs) -> str:
        """Perform translation predictions on the input text.

        Args:
            text (str): Input text to translate.

        Returns:
            str: Translated text.
        """
        return self.model(text).text

    def predict_raw(self, text: str) -> str:
        """Perform translation predictions on input text.

        Args:
            text (str): Input text to translate.

        Returns:
            str: Translated text.
        """
        return self.model(text).text

    def __call__(self, text: str, *args, **kwargs) -> str:
        """Alias of the 'predict' method"""
        return self.predict(text=text, **kwargs)
