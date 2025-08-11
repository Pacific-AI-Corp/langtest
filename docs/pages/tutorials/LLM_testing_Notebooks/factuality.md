---
layout: docs
header: true
seotitle: NLP Tutorials | Factuality Test | John Snow Labs
title: Factuality
key: llm_testing_notebooks
permalink: /docs/pages/tutorials/llm_testing_notebooks/factuality
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

## Overview

In the Factuality Test notebook, we're evaluating `gpt-3.5-turbo-instruct` model on factuality test. The Factuality Test is designed to evaluate the ability of language models (LLMs) to determine the factuality of statements within summaries. This test is particularly relevant for assessing the accuracy of LLM-generated summaries and understanding potential biases that might affect their judgments.

## Open in Collab

{:.table2}
| Category               | Hub                           | Task                              | Dataset Used | Open In Colab                                                                                                                                                                                                                                    
| ----------------------------------- |
|  **Factuality**                          | 	OpenAI                    | Question-Answering                              | Factual-Summary-Pairs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Factuality.ipynb)                                    

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## Config Used

```yml 

tests:
  defaults:
    min_pass_rate: 0.80

  factuality:
    order_bias:
      min_pass_rate: 0.70

```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Supported Tests

- **`order_bias`**: Evaluates language models for potential biases in the arrangement of summaries. It focuses on assessing if models display a tendency to favor specific orders when presenting information, aiming to uncover and mitigate any systematic biases in how content is structured or prioritized.

</div></div>

