from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import psutil
import re
from textwrap import dedent

from langtest.modelhandler.modelhandler import ModelAPI
from langtest.tasks.task import TaskManager

# check the available of RAM or GPU memory is above 30 GB in current system


def check_memory():
    mem = psutil.virtual_memory()
    if mem.available > 30 * 1024 * 1024 * 1024:
        return True
    else:
        return False


class PrometheusEval:
    """Class for evaluating the Prometheus model."""

    pipeline = None

    def __init__(
        self,
        model_name: str = "prometheus-eval/prometheus-7b-v2.0",
        hub: str = "huggingface",
        eval_type: str = "absolute_grading",
        criteria_description: Dict[str, str] = None,
        model_kwargs: Dict[str, str] = None,
    ):
        """
        Initializes the PrometheusEval object.

        Args:
            model_name: The name of the model for evaluation.
        """
        self.model_name = model_name
        self.hub = hub
        self.input_variables = ["query", "result", "answer"]
        self.eval_type = eval_type
        self.criteria_description = criteria_description
        self.model_kwargs = model_kwargs

        try:
            if hub != "huggingface" and PrometheusEval.pipeline is None:
                task = TaskManager("question-answer")
                PrometheusEval.pipeline = task.model(model_path=model_name, model_hub=hub)
            else:
                if PrometheusEval.pipeline is None:
                    from transformers import pipeline

                    # Check if memory is available
                    assert check_memory(), "Memory is not available to run the model"

                    PrometheusEval.pipeline = pipeline(
                        model=model_name, task="text-generation"
                    )

        except MemoryError as e:
            raise MemoryError("Memory is not available to run the model", e)

    def _get_feedback(self, response_text: str) -> Optional[Tuple[str, int]]:
        """
        Get feedback from the text.

        Args:
            text: The text to extract feedback from.

        Returns:
            A tuple of feedback and result.
        """
        feedback = None
        result = None
        feedback_match = re.search(r"###Feedback:", response_text)
        result_match = re.search(r"\[RESULT\]", response_text)

        if feedback_match:
            feedback = response_text[feedback_match.end() : result_match.start()].strip()
        if result_match:
            result = response_text[result_match.end() :].strip()

        if feedback is None:
            feedback = response_text[: result_match.start()].strip()

        return feedback, result

    def evaluate_response(self, llm_response: Dict[str, str]) -> Tuple[str, int]:
        """
        Evaluate the model.

        Args:
            query: The query for the model.
            result: The result from the model.
            answer: The expected answer.

        Returns:
            A tuple of feedback and score.
        """
        query = llm_response.get("query", None)
        result = llm_response.get("result", None)
        answer = llm_response.get("answer", None)
        response_a = llm_response.get("response_a", None)
        response_b = llm_response.get("response_b", None)

        if any(v is None for v in [query, result, answer]):
            if any(v is None for v in [query, response_a, response_b]):
                raise ValueError("Input variables should be query, result, and answer.")

        if self.eval_type == "absolute_grading":
            build_prompt = AbsoluteGrading(
                instruction=query,
                response=result,
                reference_answer=answer,
                criteria_description=self.criteria_description,
            )
            prompt = build_prompt.get_prompt()

        elif self.eval_type == "relative_grading":
            build_prompt = RelativeGrading(
                instruction=query,
                response_a=response_a,
                response_b=response_b,
                reference_answer=answer,
                criteria_description=self.criteria_description,
            )
            prompt = build_prompt.get_prompt()

        # to support vllm endpoint from langchain
        if isinstance(self.pipeline, ModelAPI):
            response = self.pipeline.predict(
                text={
                    "text": prompt,
                },
                prompt={
                    "template": "{text}",
                    "input_variables": ["text"],
                },
            )
        else:
            if self.model_kwargs:
                response = self.pipeline(prompt, **self.model_kwargs)
            else:
                response = self.pipeline(
                    prompt, max_new_tokens=500, return_full_text=False
                )

        feedback, result = self._get_feedback(response[0]["generated_text"])
        return feedback, result

    def evaluate_batch(self, entries: List[Dict[str, str]]) -> List[Tuple[str, int]]:
        """
        Evaluate the model on a batch of queries.

        Args:
            queries: A list of queries for the model.
            results: A list of results from the model.
            answers: A list of expected answers.

        Returns:
            A list of tuples of feedback and score.
        """
        queries = [entry.get("query", None) for entry in entries]
        results = [entry.get("result", None) for entry in entries]
        answers = [entry.get("answer", None) for entry in entries]
        return [
            self.evaluate_response(query, result, answer)
            for query, result, answer in zip(queries, results, answers)
        ]

    def evaluate(
        self,
        inputs: List[Dict[str, str]],
        predictions: List[Dict[str, str]],
        question_key: str = "query",
        answer_key: str = "answer",
        prediction_key: str = "result",
    ) -> List[Tuple[str, int]]:
        """Evaluate question answering examples and predictions."""
        examples = [
            {
                "query": input_example.get(question_key, ""),
                "result": prediction_example.get(prediction_key, ""),
                "answer": input_example.get(answer_key, ""),
            }
            for input_example, prediction_example in zip(inputs, predictions)
        ]
        return self.evaluate_batch(examples)

    @staticmethod
    def reset_pipeline():
        PrometheusEval.pipeline = None


@dataclass
class AbsoluteGrading:
    """Class for absolute grading of the Prometheus model.

    Absolute Grading (Direct Assessment)
    Prometheus requires 4 components in the input: An instruction, a response to evaluate,
    a score rubric, and a reference answer. You could refer to the prompt format below.
    You should fill in the instruction, response, reference answer, criteria description,
    and score description for score in range of n number or True or False.
    """

    instruction: str
    response: str
    reference_answer: str
    criteria_description: Dict[str, str]

    def __post_init__(self):
        self.input_variables = ["instruction", "response", "reference_answer"]

    def get_prompt(self) -> str:
        """
        Get the prompt for the model.

        Returns:
            The prompt for the model.
        """
        s, f = self.get_score_rubric()
        prompt = dedent("""
        ###Task Description:
        An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets from {formatted_criteria_keys}, and a score rubric representing a evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is from {formatted_criteria_keys}. You should refer to the score rubric.
        3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (anyone of this {formatted_criteria_keys})\"
        4. Please do not generate any other opening, closing, and explanations.

        ###The instruction to evaluate:
        {instruction}

        ###Response to evaluate:
        {response}

        ###Reference Answer (Score in {formatted_criteria_keys}):
        {reference_answer}

        ###Score Rubrics:
        {score_rubric}

        ###Feedback:
        """)
        return prompt.format(
            instruction=self.instruction,
            response=self.response,
            reference_answer=self.reference_answer,
            score_rubric=s,
            formatted_criteria_keys=f,
        )

    def get_score_rubric(self) -> Dict[str, str]:
        """
        Get the score rubric for the model.

        Returns:
            The score rubric for the model.
        """
        # Format the criteria keys for the score rubric
        formatted_criteria_keys = ", ".join(
            f"'{i}'" for i in self.criteria_description.keys()
        )
        formatted_criteria_keys = f"[{formatted_criteria_keys}]"
        score_rubric = f"{formatted_criteria_keys}\n"

        # Add criteria and description to the score rubric
        for criteria, description in self.criteria_description.items():
            score_rubric += f"{criteria}: {description}\n"

        return score_rubric, formatted_criteria_keys


@dataclass
class RelativeGrading:
    """Class for relative grading of the Prometheus model.

    Relative Grading (Comparative Assessment)
    Prometheus requires 4 components in the input: An instruction, a response to evaluate,
    a reference answer, and a comparative answer. You could refer to the prompt format below.
    You should fill in the instruction, response, reference answer, and comparative answer.
    """

    instruction: str
    response_a: str
    response_b: str
    reference_answer: str
    criteria_description: Dict[str, str]

    def __post_init__(self):
        self.input_variables = ["instruction", "response", "reference_answer"]

    def get_prompt(self) -> str:
        """
        Get the prompt for the model.

        Returns:
            The prompt for the model.
        """
        s, f = self.get_score_rubric()
        prompt = dedent("""
        ###Task Description:
        An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, evaluate a both responses Response A and Response B. You should refer to the score rubric.
        3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (anyone from this {formatted_criteria_keys})"
        4. Please do not generate any other opening, closing, and explanations.

        ###Instruction:
        {instruction}

        ###Response A:
        {response_a}

        ###Response B:
        {response_b}

        ###Reference Answer:
        {reference_answer}

        ###Score Rubric:
        {score_rubric}

        ###Feedback:
        """)
        return prompt.format(
            instruction=self.instruction,
            response_a=self.response_a,
            response_b=self.response_b,
            reference_answer=self.reference_answer,
            score_rubric=s,
            formatted_criteria_keys=f,
        )

    def get_score_rubric(self) -> Dict[str, str]:
        """
        Get the score rubric for the model.

        Returns:
            The score rubric for the model.
        """
        # Format the criteria keys for the score rubric
        formatted_criteria_keys = ", ".join(
            f"'{i}'" for i in self.criteria_description.keys()
        )
        formatted_criteria_keys = f"[{formatted_criteria_keys}]"
        score_rubric = f"{formatted_criteria_keys}\n"

        # Add criteria and description to the score rubric
        for criteria, description in self.criteria_description.items():
            score_rubric += f"{criteria}: {description}\n"

        return score_rubric, formatted_criteria_keys
