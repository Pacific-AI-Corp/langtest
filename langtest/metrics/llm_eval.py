import re
import ast
import string
from textwrap import dedent
from typing import List, Mapping, Optional, Tuple
from ..utils.custom_types.helpers import HashableDict
from .eval_prompts import MENTAL_HEALTH_EVAL_PROMPT, MHCEvaluation

template = """You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: CORRECT or INCORRECT here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more or relevant information than the true answer, as long as it does not contain any conflicting statements. Begin!

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:"""
server_prompt = "Perform the task to the best of your ability."
input_variables = ["query", "result", "answer"]


class EvalTemplate:
    """
    The EvalTemplate class provides a method to build a prompt for evaluating student answers
    based on a given rubric. The prompt is designed for a teacher to grade a quiz by comparing
    the student's answer with the true answer and scoring it according to specified criteria.

    Methods
    -------
    build_prompt(rubic_score: Mapping[str, str] = {"CORRECT": None, "INCORRECT": None}) -> str
        Constructs and returns a grading prompt based on the provided rubric scores.

    """

    @staticmethod
    def build_prompt(
        rubic_score: Mapping[str, str] = {
            "CORRECT": None,
            "INCORRECT": None,
        }
    ):
        """ """
        grade_list = list(rubic_score.keys())
        grade_list = ", ".join(grade_list[:-1]) + f" or {grade_list[-1]}"

        eval_criteria = [
            f"{grade_name}: {criteria}\n"
            for grade_name, criteria in rubic_score.items()
            if criteria
        ]
        prompt = (
            "You are a teacher grading a quiz. You are given a question, the student's "
            "answer, and the true answer, and are asked to score the student answer as either "
            f"{grade_list}."
        )

        if eval_criteria:
            eval_criteria = "".join(eval_criteria)
            prompt += dedent(
                f"""\n\nScore the student answer based on the following criteria:\n{eval_criteria}"""
            )

        prompt += dedent(
            f"""
        Example Format:
        QUESTION: question here
        STUDENT ANSWER: student's answer here
        TRUE ANSWER: true answer here
        GRADE: {grade_list} here

        {
            ("Grade the student answers based ONLY on their factual accuracy. Ignore differences"
             " in punctuation and phrasing between the student answer and true answer. It is OK "
             "if the student answer contains more or relevant information than the true answer, as"
             " long as it does not contain any conflicting statements. Begin!")
        }

        QUESTION: {{query}}
        STUDENT ANSWER: {{result}}
        TRUE ANSWER: {{answer}}
        GRADE:"""
        )
        return prompt


class LlmEval:
    """llm_eval for evaluating question answering."""

    grade_list = None

    def __init__(
        self,
        llm,
        template=template,
        input_variables=input_variables,
        grade_list=None,
    ):
        """
        Initializes the LlmEval object.

        Args:
            llm: The language model for evaluation.
            template: Template for model prompts.
            input_variables: Variables expected in the input.
            server_prompt: Server prompt for model predictions.

        Raises:
            ValueError: If input variables do not match expected values.
        """
        self.llm = llm
        self.template = template
        self.input_variables = input_variables
        self.server_prompt = server_prompt
        LlmEval.grade_list = grade_list

        expected_input_vars = {"query", "answer", "result"}
        if expected_input_vars != set(self.input_variables):
            raise ValueError(
                f"Input variables should be {expected_input_vars}, "
                f"but got {self.input_variables}"
            )

    @staticmethod
    def _get_score(text: str) -> Optional[Tuple[str, int]]:
        if LlmEval.grade_list is None:
            default_grades = ["CORRECT", "INCORRECT"]
            grade_list_pattern = f"grade:\\s*({'|'.join(default_grades).lower()})"
        else:
            grade_list_pattern = f"(?:grade\\s*)?({'|'.join(LlmEval.grade_list).lower()})"

        match = re.search(grade_list_pattern, text.strip(), re.IGNORECASE)
        if match:
            grade = match.group(1).upper()
            if LlmEval.grade_list is None:
                if grade == "CORRECT":
                    return "CORRECT", 1
                elif grade == "INCORRECT":
                    return "INCORRECT", 0
            elif grade in LlmEval.grade_list:
                return grade, LlmEval.grade_list.index(grade)
        else:
            try:
                # Check for first word
                first_word = (
                    text.strip()
                    .split()[0]
                    .translate(str.maketrans("", "", string.punctuation))
                )
                if LlmEval.grade_list is None:
                    if first_word.upper() == "CORRECT":
                        return "CORRECT", 1
                    elif first_word.upper() == "INCORRECT":
                        return "INCORRECT", 0
                elif first_word.upper() in LlmEval.grade_list:
                    return first_word.upper(), LlmEval.grade_list.index(
                        first_word.upper()
                    )

                # Check for last word
                last_word = (
                    text.strip()
                    .split()[-1]
                    .translate(str.maketrans("", "", string.punctuation))
                )
                if LlmEval.grade_list is None:
                    if last_word.upper() == "CORRECT":
                        return "CORRECT", 1
                    elif last_word.upper() == "INCORRECT":
                        return "INCORRECT", 0
                elif last_word.upper() in LlmEval.grade_list:
                    return last_word.upper(), LlmEval.grade_list.index(last_word.upper())
            except IndexError:
                pass
        return None

    @staticmethod
    def _parse_string_eval_output(text: str) -> dict:
        """Parse the output text.

        Args:
            text (str): The output text to parse.

        Returns:
            Any: The parsed output.
        """
        reasoning = text.strip()
        parsed_scores = LlmEval._get_score(reasoning)
        if parsed_scores is None:
            value, score = None, None
        else:
            value, score = parsed_scores
        return {
            "reasoning": reasoning,
            "value": value,
            "score": score,
        }

    def evaluate_example(self, example: dict) -> dict:
        """
        Evaluates a single example using the language model.

        Args:
            example: Dictionary containing input details.

        Returns:
            dict: Evaluation results with reasoning, value, and score.
        """

        output = self.llm.predict(
            prompt=HashableDict(
                **{
                    "template": self.template,
                    "input_variables": self.input_variables,
                }
            ),
            text=HashableDict(**example),
            server_prompt=self.server_prompt,
        )

        parsed_result = self._parse_string_eval_output(output)

        return parsed_result

    def evaluate_batch(self, examples: List[dict]) -> List[dict]:
        """
        Evaluates a batch of examples using the language model.

        Args:
            examples: List of dictionaries containing input details.

        Returns:
            List[dict]: List of evaluation results for each example.
        """
        return [self.evaluate_example(example) for example in examples]

    def evaluate(
        self,
        inputs: List[dict],
        predictions: List[dict],
        question_key: str = "question",
        answer_key: str = "answer",
        prediction_key: str = "result",
    ) -> List[dict]:
        """Evaluate question answering examples and predictions."""
        examples = [
            {
                "query": input_example.get(question_key, ""),
                "answer": input_example.get(answer_key, ""),
                "result": prediction_example.get(prediction_key, ""),
            }
            for input_example, prediction_example in zip(inputs, predictions)
        ]

        return self.evaluate_batch(examples)


SUMMARY_EVAL_TEMPLATE = """
Evaluate the clinical summary generated from the doctor-patient dialogue below.

Instructions:
1. Factual Completeness (1-10): Provide an integer rating.
2. No Hallucinations (1-10): Provide an integer rating.
3. Clinical Tone & Structure (1-10): Provide an integer rating.
4. Overall Quality (1-10): Provide an integer rating.

Output a dict object with keys "Factual Completeness", "No Hallucinations", "Clinical Tone & Structure", and "Overall Quality" mapping to the integer ratings.

### Dialogue
{dialogue}
### Generated Summary
{summary}
"""


class SummaryEval:
    """SummaryEval for evaluating clinical summary generation from doctor-patient dialogues."""

    def __init__(
        self,
        llm,
        template: str = SUMMARY_EVAL_TEMPLATE,
        input_variables: List[str] = ["context", "summary"],
    ):
        self.llm = llm
        self.template = template
        self.input_variables = input_variables

    def evaluate(self, inputs: dict, predictions: dict) -> List[dict]:
        """Evaluate a list of dialogue-summary pairs."""

        dialogue = inputs.get("dialogue", "")
        summary = predictions.get("summary", "")

        content = self.llm.predict(
            prompt=HashableDict(
                **{
                    "template": self.template,
                    "input_variables": self.input_variables,
                }
            ),
            text=HashableDict(**{"dialogue": dialogue, "summary": summary}),
        )

        # Convert string output to dict (assuming the model returns a dictionary-like string)
        try:

            # Remove markdown code block formatting if present
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                dict_str = match.group(1)
            else:
                dict_str = content.strip()

            result_dict = ast.literal_eval(dict_str)
            # loaded_eval
            # result_dict = eval(evaluation.choices[0].message.content)
            return result_dict
        except Exception as e:
            return {
                "Factual Completeness": 0,
                "No Hallucinations": 0,
                "Clinical Tone & Structure": 0,
                "Overall Quality": 0,
                "error": str(e),
            }

    def evaluate_batch(self, inputs: List[dict], predictions: List[dict]) -> List[dict]:
        """Alias for evaluate - placeholder for future batch implementation."""
        results = []
        for input_example, prediction_example in zip(inputs, predictions):
            result = self.evaluate(input_example, prediction_example)
            results.append(result)
        return results


class RatingEval:
    """RatingEval for evaluating responses with customizable rating prompts."""

    def __init__(
        self,
        llm,
        eval_prompt: str = MENTAL_HEALTH_EVAL_PROMPT,
        input_variables: List[str] = ["prompt", "response"],
        include_groundtruth: bool = False,
    ):
        """
        Initialize RatingEval with custom evaluation prompt.

        Args:
            llm: The language model for evaluation
            eval_prompt: Custom prompt template for evaluation
            input_variables: Variables expected in the input
            include_groundtruth: Whether to include groundtruth in evaluation
        """
        self.llm = llm
        self.eval_prompt = eval_prompt
        self.input_variables = input_variables
        self.include_groundtruth = include_groundtruth

    def _parse_rating_output(self, text: str) -> dict:
        """
        Parse the rating output from the model response.

        Args:
            text: The model's response text

        Returns:
            dict: Parsed rating results
        """
        try:
            # Try to extract JSON/dict format
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                dict_str = match.group(1)
                result_dict = ast.literal_eval(dict_str)
                return result_dict

            # Try to parse as direct dict string
            if text.strip().startswith("{") and text.strip().endswith("}"):
                result_dict = ast.literal_eval(text.strip())
                return result_dict

            # Extract numeric scores if present
            score_pattern = r"(\d+(?:\.\d+)?)"
            scores = re.findall(score_pattern, text)

            if scores:
                return {"rating": float(scores[0]), "reasoning": text.strip()}

        except Exception as e:
            pass

        return {
            "rating": None,
            "reasoning": text.strip(),
            "error": "Could not parse rating",
        }

    def evaluate(self, inputs: dict, predictions: dict) -> List[dict]:
        """
        Evaluate a single prompt-response pair.

        Args:
            prompt: The input prompt
            response: The model's response
            groundtruth: Optional ground truth answer

        Returns:
            dict: Evaluation results
        """
        prompt = inputs.get("prompt", "")
        response = predictions.get("response", "")
        groundtruth = inputs.get("groundtruth") if self.include_groundtruth else None

        output = self.llm.predict(
            prompt=HashableDict(
                **{
                    "template": self.eval_prompt,
                    "input_variables": self.input_variables,
                    "partial_variables": {
                        "output_format": MHCEvaluation.model_json_schema(),
                    }
                }
            ),
            text=HashableDict(
                **{
                    "prompt": prompt,
                    "response": response,
                    "groundtruth": groundtruth,
                }
            ),
        )

        return self._parse_rating_output(output)

    def evaluate_batch(self, examples: List[dict]) -> List[dict]:
        """
        Evaluate a batch of examples.

        Args:
            examples: List of dicts with 'prompt', 'response', and optionally 'groundtruth'

        Returns:
            List[dict]: List of evaluation results
        """
        results = []
        for example in examples:
            prompt = example.get("prompt", "")
            response = example.get("response", "")
            groundtruth = example.get("groundtruth") if self.include_groundtruth else None

            result = self.evaluate(prompt, response, groundtruth)
            results.append(result)

        return results
