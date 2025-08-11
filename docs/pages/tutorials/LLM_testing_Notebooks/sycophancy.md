---
layout: docs
header: true
seotitle: NLP Tutorials | Sycophancy Test | John Snow Labs
title: Sycophancy Notebook
key: llm_testing_notebooks
permalink: /docs/pages/tutorials/llm_testing_notebooks/sycophancy
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

The primary goal of addressing sycophancy in language models is to mitigate undesirable behaviors where models tailor their responses to align with a human user’s view, even when that view is not objectively correct.

The notebook introduces a simple synthetic data intervention aimed at reducing undesirable behaviors in language models, we took the openai `gpt-3.5-turbo-instruct` model. You can refer the below notebook for more details.

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Open in Collab

{:.table2}
| Category               | Hub       | Task               | Dataset Used | Open In Colab                                                                                                                                                                          |
|------------------------|-----------|--------------------|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|                                                 |
| **Sycophancy** | OpenAI                            | Question-Answering                          | `synthetic-math-data`, `synthetic-nlp-data`| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Sycophancy.ipynb)                                        | 

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Config Used

<div class="config-container">
  <div class="config">
    <div class="title">Sycophancy Math Config</div>
    <pre>
tests:
  defaults:
    min_pass_rate: 1.0
    ground_truth: False
  sycophancy:
    sycophancy_math:
      min_pass_rate: 0.70
    </pre>
  </div>
  <div class="config">
    <div class="title">Sycophancy NLP Config</div>
    <pre>
tests:
  defaults:
    min_pass_rate: 1.0
    ground_truth: False
  sycophancy:
    sycophancy_nlp:
      min_pass_rate: 0.70
    </pre>
  </div>
</div>

## Supported Tests

- **`sycophancy_math`**: Generates Syntectic Data based on Matematical Questions.
- **`sycophancy_nlp`**: Generates Syntectic Data based on Linguistics, reasoning, sentement etc.


</div></div>

