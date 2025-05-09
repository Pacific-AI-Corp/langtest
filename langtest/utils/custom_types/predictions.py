from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .helpers import Span


class NERPrediction(BaseModel):
    """Single prediction obtained from a named entity recognition model"""

    entity: str = Field(None, alias="entity_group")
    span: Span
    score: Optional[float] = None
    doc_id: Optional[int] = None
    doc_name: Optional[str] = None
    pos_tag: Optional[str] = None
    chunk_tag: Optional[str] = None

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
    )

    @classmethod
    def from_span(
        cls,
        entity: str,
        word: str,
        start: int,
        end: int,
        score: float = None,
        doc_id: int = None,
        doc_name: str = None,
        pos_tag: str = None,
        chunk_tag: str = None,
    ) -> "NERPrediction":
        """"""
        return cls(
            entity=str(entity),
            span=Span(start=start, end=end, word=word),
            score=score,
            doc_id=doc_id,
            doc_name=doc_name,
            pos_tag=pos_tag,
            chunk_tag=chunk_tag,
        )

    def __hash__(self):
        """"""
        return hash(self.__repr__())

    def __eq__(self, other):
        """"""
        if isinstance(other, NERPrediction):
            return (
                self.entity == other.entity
                and self.span.start == other.span.start
                and self.span.end == other.span.end
            )
        return False

    def __str__(self) -> str:
        """"""
        return f"{self.span.word}: {self.entity}"

    def __repr__(self) -> str:
        """"""
        return f"<NERPrediction(entity='{self.entity}', span={self.span})>"


class SequenceLabel(BaseModel):
    """Single prediction obtained from text-classification models"""

    label: Union[str, int]
    score: float

    def __str__(self):
        """"""
        return f"{self.label}"
