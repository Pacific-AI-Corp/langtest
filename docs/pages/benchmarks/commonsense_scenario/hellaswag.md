---
layout: docs
header: true
seotitle: HellaSwag Benchmark | LangTest | John Snow Labs
title: HellaSwag
key: benchmarks-hellaswag
permalink: /docs/pages/benchmarks/commonsense_scenario/hellaswag/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

<div class="h3-box" markdown="1">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/HellaSwag_Question_Answering.ipynb)

**Source:** [HellaSwag: Can a Machine Really Finish Your Sentence?](https://aclanthology.org/P19-1472/)

HellaSWAG is a dataset for studying grounded commonsense inference. The samples start with one ore two sentence and the last sentence is left incomplete, there are some possible senseful completions in the dataset and model's completion is compared to them.

You can see which subsets and splits are available below.

{:.table2}
| Split                    | Details                                                                                                                                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **test**      | Dev set Training set from the hellaswag dataset with 3000 examples which is a benchmark for Commonsense NLI. It includes a context and some endings which complete the context. |
| **test-tiny** | Truncated version of the test set from the hellaswag dataset with 50 examples.                                                                                                  |

#### Example

In the evaluation process, we start by fetching *original_question* from the dataset. The model then generates an *expected_result* based on this input. To assess model robustness, we introduce perturbations to the *original_question*, resulting in *perturbed_question*. The model processes these perturbed inputs, producing an *actual_result*. The comparison between the *expected_result* and *actual_result* is conducted using the `llm_eval` approach (where llm is used to evaluate the model response). Alternatively, users can employ metrics like **String Distance** or **Embedding Distance** to evaluate the model's performance in the Question-Answering Task within the robustness category. For a more in-depth exploration of these approaches, you can refer to this [notebook](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Evaluation_Metrics.ipynb) discussing these three methods.


{:.table3}
| category   | test_type    | original_question                  | perturbed_question                     | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | 	add_slangs  | A huge crowd is in the stands in an arena. A man throws a javelin. Photographers take pictures in the background. several men | A humongous crowd is in the stands in an arena. A bloke throws a javelin. Photographers take pictures in the background. several men | and women cheer as the javelin sails through the air. <br><br>The javelin lands with a thud, and the crowd erupts in applause. | and women cheer as the javelin sails through the air. <br><br>The javelin lands with a thud in the center of the field, and the crowd erupts in applause.  | True |


> Generated Results for `gpt-3.5-turbo-instruct` model from `OpenAI`

</div>