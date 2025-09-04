---
layout: docs
header: true
seotitle: PIQA Benchmark | LangTest | John Snow Labs
title: PIQA
key: benchmarks-piqa
permalink: /docs/pages/benchmarks/commonsense_scenario/piqa/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

<div class="h3-box" markdown="1">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/PIQA_dataset.ipynb)

**Source:** [PIQA: Reasoning about Physical Commonsense in Natural Language](https://arxiv.org/abs/1911.11641)

The PIQA dataset is a collection of multiple-choice questions that test the ability of language models to reason about physical commonsense in natural language. The questions are based on everyday scenarios that involve some physical knowledge, such as cooking, gardening, or cleaning. The test dataset contains 3084 questions, each with a goal, a solution, and two alternative solutions. The correct solution is the one that is most likely to achieve the goal, while the alternatives are either ineffective or harmful. The dataset is designed to challenge the models’ understanding of real-world interactions and causal effects.

You can see which subsets and splits are available below.

{:.table2}
| Split              | Details                                                                                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **test**      | Testing set from the PIQA dataset, containing 1500 questions. This dataset does not contain labels and accuracy & fairness tests cannot be run with it.  |
| **test-tiny** | Truncated version of PIQA test dataset which contains 50 questions. This dataset does not contain labels and accuracy & fairness tests cannot be run with it. |
| **validation**      | Validation set from the PIQA dataset, containing 1500 question and answer examples.  |
| **validation-tiny** | Truncated version of PIQA validation dataset which contains 50 question and answer examples. |

#### Example

In the evaluation process, we start by fetching *original_question* from the dataset. The model then generates an *expected_result* based on this input. To assess model robustness, we introduce perturbations to the *original_question*, resulting in *perturbed_question*. The model processes these perturbed inputs, producing an *actual_result*. The comparison between the *expected_result* and *actual_result* is conducted using the `llm_eval` approach (where llm is used to evaluate the model response). Alternatively, users can employ metrics like **String Distance** or **Embedding Distance** to evaluate the model's performance in the Question-Answering Task within the robustness category. For a more in-depth exploration of these approaches, you can refer to this [notebook](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Evaluation_Metrics.ipynb) discussing these three methods.

{:.table3}
| category   | test_type           | original_question  | perturbed_question  |   options                                                     | expected_result | actual_result    | pass  |
|------------|---------------------|--------------------|---------------------|---------------------------------------------------------------|-----------------|------------------|-------|
| robustness | 	dyslexia_word_swap | hands              | hands               | A. is used too put on shoe <br>B. is used too put on milk jug | A               | A                | True  |


> Generated Results for `gpt-3.5-turbo-instruct` model from `OpenAI`

</div>