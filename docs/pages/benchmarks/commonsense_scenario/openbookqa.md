---
layout: docs
header: true
seotitle: OpenBookQA Benchmark | LangTest | John Snow Labs
title: OpenBookQA
key: benchmarks-openbookqa
permalink: /docs/pages/benchmarks/commonsense_scenario/openbookqa/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

<div class="h3-box" markdown="1">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/OpenbookQA_dataset.ipynb)

**Source:** [Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering](https://arxiv.org/abs/1809.02789)

The OpenBookQA dataset is a collection of multiple-choice questions that require complex reasoning and inference based on general knowledge, similar to an “open-book” exam. The questions are designed to test the ability of natural language processing models to answer questions that go beyond memorizing facts and involve understanding concepts and their relations. The dataset contains 500 questions, each with four answer choices and one correct answer. The questions cover various topics in science, such as biology, chemistry, physics, and astronomy.

You can see which subsets and splits are available below.

{:.table2}
| Split                    | Details                                                                                                    |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| **test**      | Testing set from the OpenBookQA dataset, containing 500 multiple-choice elementary-level science questions |
| **test-tiny** | Truncated version of the test set from the OpenBookQA dataset, containing 50 multiple-choice examples.     |

#### Example

In the evaluation process, we start by fetching *original_question* from the dataset. The model then generates an *expected_result* based on this input. To assess model robustness, we introduce perturbations to the *original_question*, resulting in *perturbed_question*. The model processes these perturbed inputs, producing an *actual_result*. The comparison between the *expected_result* and *actual_result* is conducted using the `llm_eval` approach (where llm is used to evaluate the model response). Alternatively, users can employ metrics like **String Distance** or **Embedding Distance** to evaluate the model's performance in the Question-Answering Task within the robustness category. For a more in-depth exploration of these approaches, you can refer to this [notebook](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Evaluation_Metrics.ipynb) discussing these three methods.

{:.table3}
| category   | test_type        | original_question                            | perturbed_question                        |   options                                                  | expected_result                | actual_result                  | pass   |
|------------|------------------|----------------------------------------------|-------------------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | add_abbreviation | There is most likely going to be fog around: | There is most likely going 2 b fog around:| A. a marsh<br>B. a tundra<br>C. da plains<br>D. a desert   | A. a marsh | A. a marsh  | True |

> Generated Results for `gpt-3.5-turbo-instruct` model from `OpenAI`


</div>
