---
layout: docs
header: true
seotitle: NLP Tutorials | Ideology Test | John Snow Labs
title: Ideology Notebook
key: llm_testing_notebooks
permalink: /docs/pages/tutorials/llm_testing_notebooks/ideology
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In the Political Demo notebook, we're evaluating `gpt-3.5-turbo-instruct` model on Ideology test in capturing nuanced political beliefs beyond the traditional left-right spectrum.

## Open in Collab

{:.table2}
| Category               | Hub                           | Task                              | Open In Colab                                                                                                                                                                                                                                    
| ----------------------------------- |
|  **Ideology**                          | 	OpenAI                    | Question-Answering                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/test-specific-notebooks/Political_Demo.ipynb)                                    

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## Config Used

```yml 

model_parameters:
  temperature: 0.2
  max_tokens: 200

tests:
  defaults:
    min_pass_rate: 1.0

  ideology:
    political_compass:

```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Supported Tests

- **`political_compass`**: Political Compass aims for a nuanced understanding, avoiding oversimplification and capturing the full range of political opinions and beliefs.


</div></div>

