---
layout: docs
header: true
seotitle: ASDiv Benchmark | LangTest | John Snow Labs
title: ASDiv
key: benchmarks-asdiv
permalink: /docs/pages/benchmarks/other_benchmarks/asdiv/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

<div class="h3-box" markdown="1">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/ASDiv_dataset.ipynb)

**Source:** [A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers](https://arxiv.org/abs/2106.15772)

The ASDiv benchmark is a dataset of math word problems (MWPs) designed to evaluate the capability of various MWP solvers. The dataset is diverse in terms of both language patterns and problem types. The dataset is intended to be a challenging benchmark for natural language processing models and to encourage the development of models that can perform complex reasoning and inference.

You can see which subsets and splits are available below.

{:.table2}
| Split           |Details                                                                                                |
| ----------------|------------------------------------------------------------------------------------------------------ |
| **test**       | Testing set from the ASDiv dataset, containing 2305 question and answers andexamples.                         |
| **test-tiny**  | Truncated version of the test set from the ASDiv dataset, containing 50 question and answers examples. |

#### Example

In the evaluation process, we start by fetching *original_context* and *original_question* from the dataset. The model then generates an *expected_result* based on this input. To assess model robustness, we introduce perturbations to the *original_context* and *original_question*, resulting in *perturbed_context* and *perturbed_question*. The model processes these perturbed inputs, producing an *actual_result*. The comparison between the *expected_result* and *actual_result* is conducted using the QAEvalChain approach from the LangChain library. Alternatively, users can employ metrics like **String Distance** or **Embedding Distance** to evaluate the model's performance in the Question-Answering Task within the robustness category. For a more in-depth exploration of these approaches, you can refer to this [notebook](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Evaluation_Metrics.ipynb) discussing these three methods.

{:.table3}
| category   | test_type    | original_context                                         | original_question                  | perturbed_context                                           | perturbed_question                     | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | add_ocr_typo | Seven red apples and two green apples are in the basket. | How many apples are in the basket? | zeien r^ed apples an^d tvO grcen apples are i^n t^e basket. | ho^w m^any apples are i^n t^he basket? | Nine apples are in the basket. | Four apples are in the basket. | False  |


> Generated Results for `gpt-3.5-turbo-instruct` model from `OpenAI`

</div>