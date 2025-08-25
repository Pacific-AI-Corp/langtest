---
layout: docs
header: true
seotitle: SocialIQA Benchmark | LangTest | John Snow Labs
title: SocialIQA
key: benchmarks-socialiqa
permalink: /docs/pages/benchmarks/commonsense_scenario/siqa/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

<div class="h3-box" markdown="1">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/SIQA_dataset.ipynb)

**Source:** [SocialIQA: Commonsense Reasoning about Social Interactions](https://arxiv.org/abs/1904.09728)

SocialIQA is a dataset for testing the social commonsense reasoning of language models. It consists of over 1900 multiple-choice questions about various social situations and their possible outcomes or implications. The questions are based on real-world prompts from online platforms, and the answer candidates are either human-curated or machine-generated and filtered. The dataset challenges the models to understand the emotions, intentions, and social norms of human interactions.

You can see which subsets and splits are available below.

{:.table2}
| Split              | Details                                                                                |
| ------------------ | -------------------------------------------------------------------------------------- |
| **test**      | Testing set from the SIQA dataset, containing 1954 question and answer examples.       |
| **test-tiny** | Truncated version of SIQA-test dataset which contains 50 question and answer examples. |

#### Example

In the evaluation process, we start by fetching *original_context* and *original_question* from the dataset. The model then generates an *expected_result* based on this input. To assess model robustness, we introduce perturbations to the *original_context* and *original_question*, resulting in *perturbed_context* and *perturbed_question*. The model processes these perturbed inputs, producing an *actual_result*. The comparison between the *expected_result* and *actual_result* is conducted using the `llm_eval` approach (where llm is used to evaluate the model response). Alternatively, users can employ metrics like **String Distance** or **Embedding Distance** to evaluate the model's performance in the Question-Answering Task within the robustness category.For a more in-depth exploration of these approaches, you can refer to this [notebook](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Evaluation_Metrics.ipynb) discussing these three methods.


{:.table3}
| category   | test_type          | original_context                                                | original_question                       | perturbed_context                                               | perturbed_question                       | options                                                                   | expected_result         | actual_result      | pass  |
|------------|--------------------|-----------------------------------------------------------------|-----------------------------------------|-----------------------------------------------------------------|------------------------------------------|---------------------------------------------------------------------------|-------------------------|--------------------|-------|
| robustness | dyslexia_word_swap | Tracy didn't go home that evening and resisted Riley's attacks. | What does Tracy need to do before this? | Tracy didn't go home that evening and resisted Riley's attacks. | What does Tracy need too do before this? |A. make a new plan<br>B. Go home and sea Riley<br>C. Find somewhere too go | C. Find somewhere to go | A. Make a new plan | False |


> Generated Results for `gpt-3.5-turbo-instruct` model from `OpenAI`

</div>