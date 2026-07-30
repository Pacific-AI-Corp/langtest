import os
import json
from typing import TYPE_CHECKING, Callable, Dict, List

import pandas as pd

from langtest.datahandler.utils import ensure_download_and_unzip

if TYPE_CHECKING:
    from langtest.utils.custom_types.sample import Sample


PREDEFINED_DATASETS: Dict[str, Callable[..., List["Sample"]]] = {}


def register_predefined_dataset(name: str):
    """Decorator to register a predefined dataset."""

    def decorator(func: Callable[..., List["Sample"]]):
        PREDEFINED_DATASETS[name.lower()] = func
        return func

    return decorator


@register_predefined_dataset("medexqa")
def medexqa(subset="all", *args, **kwargs) -> List["Sample"]:
    """Load the MedExQA dataset."""
    from langtest.utils.custom_types import QASample

    # 1. Define the specific files and URL internally
    file_names = [
        "biomedical_engineer",
        "clinical_laboratory_scientist",
        "clinical_psychologist",
        "occupational_therapist",
        "speech_pathologist",
    ]
    base_url = "https://huggingface.co/datasets/bluesky333/MedExQA/resolve/main/test/"

    # 2. Filter the files based on the subset parameter
    if subset != "all":
        if subset not in file_names:
            raise ValueError(
                f"Subset '{subset}' is not valid. Choose from {file_names} or 'all'."
            )
        file_names = [subset]
    frames = []

    for file_name in file_names:
        file_path = f"{base_url}{file_name}_test.tsv"

        # 2. Read ONLY the required columns to save memory and parsing time
        df = pd.read_csv(
            file_path, delimiter="\t", header=None, usecols=[0, 1, 2, 3, 4, 7]
        )

        # 3. Assign clear column names immediately
        df.columns = ["question", "A", "B", "C", "D", "answer"]

        # 4. Create the 'options' dictionary column
        df["options"] = df[["A", "B", "C", "D"]].to_dict(orient="records")

        # 5. Append only the necessary final columns to our list
        frames.append(df[["question", "options", "answer"]])

    # 6. Concatenate all DataFrames at once
    raw_data = pd.concat(frames, ignore_index=True).iterrows()
    transformed_samples = []

    for sample in raw_data:
        sample = QASample(
            dataset_name="medexqa",
            original_context="-",
            original_question=sample[1]["question"],
            options="\n".join([f"{k}. {v}" for k, v in sample[1]["options"].items()]),
            expected_results=sample[1]["answer"],
        )

        transformed_samples.append(sample)
    return transformed_samples


@register_predefined_dataset("headqa")
def headqa(*args, **kwargs) -> List["Sample"]:
    """Load the HeadQA dataset."""
    from langtest.utils.custom_types import QASample

    headqa_dir = os.path.join(os.path.expanduser("~"), ".langtest", "datasets", "headqa")

    ensure_download_and_unzip(
        "https://huggingface.co/datasets/dvilares/head_qa/resolve/main/data/head-qa-es-en-pdfs.zip",
        extract_to=headqa_dir,
    )

    file_path = os.path.join(headqa_dir, "HEAD_EN", "test_HEAD_EN.json")

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:
        head_qa = json.load(f)

    def clean_answers(answers):
        return "\n".join(
            f"{chr(answer['aid'] + 64)}) {answer['atext'].strip()}" for answer in answers
        )

    df = (
        pd.DataFrame.from_dict(head_qa["exams"], orient="index")
        .reset_index(drop=True)
        .assign(
            exam_id=lambda x: x.index,
            name=lambda x: x["name"].str.strip(),
            year=lambda x: x["year"].str.strip(),
            category=lambda x: x["category"].str.strip(),
        )
        .pipe(
            lambda x: pd.json_normalize(
                x.to_dict("records"),
                record_path="data",
                meta=["exam_id", "name", "year", "category"],
            )
        )
        .assign(
            qid=lambda x: x["qid"].str.strip().astype(int),
            qtext=lambda x: x["qtext"].str.strip(),
            ra=lambda x: x["ra"].str.strip().astype(int),
            options=lambda x: x["answers"].apply(clean_answers),
        )
        .query("ra != 0")
        .assign(
            answer=lambda x: x["ra"].map(lambda value: chr(value + 64)),
        )[["qid", "qtext", "options", "answer"]]
    )

    transformed_samples = []

    for sample in df.iterrows():
        sample = QASample(
            dataset_name="headqa",
            original_context="-",
            original_question=sample[1]["qtext"],
            options=sample[1]["options"],
            expected_results=sample[1]["answer"],
        )

        transformed_samples.append(sample)
    return transformed_samples
