from typing import List, Optional, TypeVar, Union
from pydantic import BaseModel, field_validator
from .helpers import Span
from .predictions import NERPrediction, SequenceLabel


class SequenceClassificationOutput(BaseModel):
    """Output model for text classification tasks."""

    predictions: List[SequenceLabel]
    multi_label: bool = False

    def to_str_list(self) -> str:
        """Convert the output into list of strings.

        Returns:
            List[str]: predictions in form of a list of strings.
        """
        return ", ".join([x.label for x in self.predictions])

    def __str__(self) -> str:
        """String representation"""
        if self.multi_label:
            return self.to_str_list()
        labels = {elt.label: elt.score for elt in self.predictions}
        return f"SequenceClassificationOutput(predictions={labels})"

    def __eq__(self, other: "SequenceClassificationOutput") -> bool:
        """Equality comparison method."""

        if self.multi_label:
            # get all labels
            self_labels = {elt.label for elt in self.predictions}
            other_labels = {elt.label for elt in other.predictions}
            return set(self_labels) == set(other_labels)
        elif len(self.predictions) == 0 and len(other.predictions) == 0:
            return True
        else:
            top_class = max(self.predictions, key=lambda x: x.score).label
            other_top_class = max(other.predictions, key=lambda x: x.score).label
            return top_class == other_top_class


class MinScoreOutput(BaseModel):
    """Output for accuracy/representation tests."""

    min_score: float

    def to_str_list(self) -> float:
        """Convert the output into list of strings.

        Returns:
            List[str]: predictions in form of a list of strings.
        """
        return round(self.min_score, 2)

    def __repr__(self) -> str:
        """Printable representation"""
        return f"{self.min_score:.3f}"

    def __str__(self) -> str:
        """String representation"""
        return f"{self.min_score:.3f}"


class MaxScoreOutput(BaseModel):
    """Output for accuracy/representation tests."""

    max_score: float

    def to_str_list(self) -> float:
        """Formatting helper"""
        return round(self.max_score, 2)

    def __repr__(self) -> str:
        """Printable representation"""
        return f"{self.max_score:.3f}"

    def __str__(self) -> str:
        """String representation"""
        return f"{self.max_score:.3f}"

    def __eq__(self, other: "MaxScoreOutput") -> bool:
        """Greater than comparison method."""
        return self.max_score >= other.max_score

    def __ge__(self, other: "MaxScoreOutput") -> bool:
        """Greater than comparison method."""
        return self.max_score >= other.max_score


class NEROutput(BaseModel):
    """Output model for NER tasks."""

    predictions: List[NERPrediction]

    @field_validator("predictions")
    @classmethod
    def sort_by_appearance(cls, v):
        """Sort spans by order of appearance in the text"""
        return sorted(v, key=lambda x: x.span.start)

    def __len__(self):
        """Number of detected entities"""
        return len(self.predictions)

    def __getitem__(
        self, item: Union[Span, int, str]
    ) -> Optional[Union[List[NERPrediction], NERPrediction]]:
        """Item getter"""
        if isinstance(item, int):
            return self.predictions[item]
        elif isinstance(item, str):
            for pred in self.predictions:
                if pred.span.word == item:
                    return pred
            return None
        elif isinstance(item, Span):
            for prediction in self.predictions:
                if prediction.span == item:
                    return prediction
            return None
        elif isinstance(item, slice):
            return [self.predictions[i] for i in range(item.start, item.stop)]

    def to_str_list(self) -> str:
        """Converts predictions into a list of strings.

        Returns:
            List[str]: predictions in form of a list of strings.
        """
        return ", ".join([str(x) for x in self.predictions if str(x)[-3:] != ": O"])

    def __repr__(self) -> str:
        """Printable representation"""
        return self.predictions.__repr__()

    def __str__(self) -> str:
        """String representation"""
        return [str(x) for x in self.predictions].__repr__()

    def __eq__(self, other: "NEROutput"):
        """Equality comparison method."""
        # NOTE: we need the list of transformations applied to the sample to be able
        # to align and compare two different NEROutput
        raise NotImplementedError()


class TranslationOutput(BaseModel):
    """Output model for translation tasks."""

    translation_text: str  # Changed from List[str] to str

    def to_str_list(self) -> List[str]:
        """Formatting helper

        Returns:
             List[str]: the translation_text as a list of strings.
        """
        return [self.translation_text]  # Wrap self.translation_text in a list

    def __str__(self):
        """String representation of TranslationOutput."""
        return self.translation_text  # Return translation_text directly

    def __eq__(self, other):
        """Equality comparison method."""
        if isinstance(other, TranslationOutput):
            return self.translation_text == other.translation_text
        if isinstance(other, list):
            return [self.translation_text] == other
        return False


Result = TypeVar(
    "Result",
    NEROutput,
    SequenceClassificationOutput,
    MinScoreOutput,
    TranslationOutput,
    MaxScoreOutput,
    List[str],
    str,
    None,
)
