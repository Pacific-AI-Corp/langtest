---
layout: docs
header: true
seotitle: NQ-open Benchmark | LangTest | John Snow Labs
title: NQ-open
key: benchmarks-NQ-open
permalink: /docs/pages/benchmarks/other_benchmarks/nq-open/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

<div class="h3-box" markdown="1">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/NQ_open_dataset.ipynb)

**Source:** [Natural Questions: A Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026/)

The Natural Questions dataset is a large-scale collection of real user questions and answers from Wikipedia. It is designed to evaluate the performance of automatic question answering systems on open-domain questions. The dataset included in LangTest about 3,500 test examples. Each example consists of a query and one or more answers from the wikipedia page.

You can see which subsets and splits are available below.

{:.table2}
| Split                 | Details                                                                                            |
| --------------------- | -------------------------------------------------------------------------------------------------- |
| **combined**           | Training & development set from the NaturalQuestions dataset, containing 3,569 labeled examples    |
| **test**      | Development set from the NaturalQuestions dataset, containing 1,769 labeled examples               |
| **test-tiny** | Training, development & test set from the NaturalQuestions dataset, containing 50 labeled examples |

#### Example

In the evaluation process, we start by fetching *original_question* from the dataset. The model then generates an *expected_result* based on this input. To assess model robustness, we introduce perturbations to the *original_question*, resulting in *perturbed_question*. The model processes these perturbed inputs, producing an *actual_result*. The comparison between the *expected_result* and *actual_result* is conducted using the `llm_eval` approach (where llm is used to evaluate the model response). Alternatively, users can employ metrics like **String Distance** or **Embedding Distance** to evaluate the model's performance in the Question-Answering Task within the robustness category. For a more in-depth exploration of these approaches, you can refer to this [notebook](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Evaluation_Metrics.ipynb) discussing these three methods.


{:.table3}
| category   | test_type    | original_question                  |  perturbed_question                     | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | add_abbreviation | on the 6th day of christmas my true love sent to me | on da 6th day of christmas my true <3333 sent 2 me | Six geese a-laying. | On the sixth day of Christmas, my true love sent to me two turtle doves.	  | False |


> Generated Results for `gpt-3.5-turbo-instruct` model from `OpenAI`

</div>