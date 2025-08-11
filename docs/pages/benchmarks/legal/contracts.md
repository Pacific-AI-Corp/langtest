---
layout: docs
header: true
seotitle: Contracts Benchmark | LangTest | John Snow Labs
title: Contracts
key: benchmarks-contracts
permalink: /docs/pages/benchmarks/legal/contracts/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

<div class="h3-box" markdown="1">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/LegalQA_Datasets.ipynb)

**Source:** [Answer yes/no questions about whether contractual clauses discuss particular issues.](https://github.com/HazyResearch/legalbench/tree/main/tasks/contract_qa)

**Contracts** is a binary classification dataset where the LLM must determine if language from a contract contains a particular type of content.

You can see which subsets and splits are available below.

{:.table2}
| Split                  | Details                                                                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **test**       | Test set from the Contracts dataset, containing 80 samples.                                                   |

#### Example

In the evaluation process, we start by fetching *original_context* and *original_question* from the dataset. The model then generates an *expected_result* based on this input. To assess model robustness, we introduce perturbations to the *original_context* and *original_question*, resulting in *perturbed_context* and *perturbed_question*. The model processes these perturbed inputs, producing an *actual_result*. The comparison between the *expected_result* and *actual_result* is conducted using the `llm_eval` approach (where llm is used to evaluate the model response). Alternatively, users can employ metrics like **String Distance** or **Embedding Distance** to evaluate the model's performance in the Question-Answering Task within the robustness category. For a more in-depth exploration of these approaches, you can refer to this [notebook](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Evaluation_Metrics.ipynb) discussing these three methods.


{:.table3}
| category   | test_type    | original_context                                         | original_question                  | perturbed_context                                           | perturbed_question                     | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | add_abbreviation | In the event that a user's credentials are compromised, the Company shall promptly notify the affected user and require them to reset their password. The Company shall also take reasonable steps to prevent unauthorized access to the user's account and to prevent future compromises of user credentials. | Does the clause discuss compromised user credentials? | In da event that a user's credentials r compromised, tdaCompany shall promptly notify thdaffected user and require them 2 reset their password. Thedampany shall also take reasonable steps t2prevent unauthorized access to2he user's account and to 2event future compromises of user credentials. |Does da clause discuss compromised user credentials? | True	 | True  | True |


> Generated Results for `gpt-3.5-turbo-instruct` model from `OpenAI`

</div>