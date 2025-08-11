---
layout: docs
header: true
seotitle: Legal Benchmark Datasets | LangTest | John Snow Labs
title: Legal Benchmark Datasets 
key: benchmarks-legal
permalink: /docs/pages/benchmarks/legal/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---


<div class="main-docs" markdown="1">
<div class="h3-box" markdown="1">

LangTest provides support for a variety of benchmark datasets in the legal domain which are listed below in the table, allowing you to assess the performance of your models on legal queries.

</div><div class="h3-box" markdown="1">


{:.table2}
| Dataset                                   | Task               | Category | Source                                                                                                                                                 | Colab                                                                                                                                                                                                                                      |
| ----------------------------------------- | ------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [**Contracts**](contracts)              | question-answering | `robustness`, `accuracy`, `fairness`        | [Answer yes/no questions about whether contractual clauses discuss particular issues.](https://github.com/HazyResearch/legalbench/tree/main/tasks/contract_qa)                                                                                                         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/LegalQA_Datasets.ipynb)               |
| [**Consumer-Contracts**](consumer-contracts)              | question-answering | `robustness`, `accuracy`, `fairness`        | [Answer yes/no questions on the rights and obligations created by clauses in terms of services agreements.](https://github.com/HazyResearch/legalbench/tree/main/tasks/consumer_contracts_qa)                                                                                                         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/LegalQA_Datasets.ipynb)               |
| [**Privacy-Policy**](privacy-policy)              | question-answering | `robustness`, `accuracy`, `fairness`        | [Given a question and a clause from a privacy policy, determine if the clause contains enough information to answer the question.](https://github.com/HazyResearch/legalbench/tree/main/tasks/privacy_policy_qa)                                                                                                         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/LegalQA_Datasets.ipynb)               |
| [**FIQA**](fiqa)                          | question-answering | `robustness`, `accuracy`, `fairness`        | [FIQA (Financial Opinion Mining and Question Answering)](https://huggingface.co/datasets/explodinggradients/fiqa)                                                    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/Fiqa_dataset.ipynb)                |
| [**MultiLexSum**](multilexsum)            | summarization      | `robustness`, `accuracy`, `fairness`        | [Multi-LexSum: Real-World Summaries of Civil Rights Lawsuits at Multiple Granularities](https://arxiv.org/abs/2206.10883)                              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/MultiLexSum_dataset.ipynb)            |

</div>