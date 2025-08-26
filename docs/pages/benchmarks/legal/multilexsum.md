---
layout: docs
header: true
seotitle: MultiLexSum Benchmark | LangTest | John Snow Labs
title: MultiLexSum
key: benchmarks-multilexsum
permalink: /docs/pages/benchmarks/legal/multilexsum/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

<div class="h3-box" markdown="1">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/MultiLexSum_dataset.ipynb)

**Source:** [Multi-LexSum: Real-World Summaries of Civil Rights Lawsuits at Multiple Granularities](https://arxiv.org/abs/2206.10883)

The Multi-LexSum dataset is a collection of summaries of civil rights litigation lawsuits with summaries of three granularities. The dataset is designed to evaluate the performance of abstractive multi-document summarization systems. The dataset was created by multilexsum and is available on GitHub. The dataset is distinct from other datasets in its multiple target summaries, each at a different granularity (ranging from one-sentence “extreme” summaries to multi-paragraph narrations of over five hundred words).



You can see which subsets and splits are available below.

{:.table2}
| Split                     | Details                                                                                 |
| ------------------------- | --------------------------------------------------------------------------------------- |
| **test**      | Testing set from the MultiLexSum dataset, containing 868 document and summary examples. |
| **test-tiny** | Truncated version of XSum dataset which contains 50 document and summary examples.      |

</div>