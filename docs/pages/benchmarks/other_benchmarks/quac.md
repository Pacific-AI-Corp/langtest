---
layout: docs
header: true
seotitle: QUAC Benchmark | LangTest | John Snow Labs
title: QUAC
key: benchmarks-quac
permalink: /docs/pages/benchmarks/other_benchmarks/quac/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

<div class="h3-box" markdown="1">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/quac_dataset.ipynb)

**Source:** [Quac: Question Answering in Context](https://aclanthology.org/D18-1241/)

The QUAC dataset is a collection of dialogues between a student and a teacher, where the student asks questions about a given passage or context and the teacher answers them. The dataset aims to evaluate the ability of question answering models to handle conversational contexts, where the questions may depend on previous turns or omit some information1. The dataset contains questions and answers covering various topics such as history, science, literature, and art.

You can see which subsets and splits are available below.

{:.table2}
| Split              | Details                                                                                                                            |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| **test**      | Testing set from the QuAC dataset with 1000 examples for modeling, understanding, and participating in information seeking dialog. |
| **test-tiny** | Truncated version of the val set from the QuAC dataset with 50 examples.                                                           |

</div>