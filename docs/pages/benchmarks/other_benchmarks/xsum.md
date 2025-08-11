---
layout: docs
header: true
seotitle: XSum Benchmark | LangTest | John Snow Labs
title: XSum
key: benchmarks-xsum
permalink: /docs/pages/benchmarks/other_benchmarks/xsum/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

<div class="h3-box" markdown="1">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/XSum_dataset.ipynb)

**Source:** [Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization](https://aclanthology.org/D18-1206/)

The XSum dataset is a collection of news articles and their corresponding one-sentence summaries. The dataset is designed to evaluate the performance of abstractive single-document summarization systems. The goal is to create a short, one-sentence summary answering the question "What is the article about?". The dataset is available in English and is monolingual. The dataset was created by EdinburghNLP and is available on Hugging Face.


You can see which subsets and splits are available below.

{:.table2}
| Split              | Details                                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| **test**      | Test set from the Xsum dataset, containing 1,000 labeled examples                                             |
| **test-tiny** | Truncated version of the test set from the Xsum dataset, containing 50 labeled examples                       |
| **bias**      | Manually annotated bias version of the Xsum dataset, containing 382 labeled examples                          |

</div>