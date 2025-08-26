---
layout: docs
header: true
seotitle: NLP Tutorials | Clinical Test | John Snow Labs
title: Clinical Notebook
key: llm_testing_notebooks
permalink: /docs/pages/tutorials/llm_testing_notebooks/clinical
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In the Clinical Test notebook, we're evaluating `gpt-3.5-turbo-instruct` model on clinical test. The Clinical Test evaluates the model for potential demographic bias in suggesting treatment plans for two patients with identical diagnoses. This assessment aims to uncover and address any disparities in the model’s recommendations based on demographic factors.

## Open in Collab

{:.table2}
| Category               | Hub       | Task               | Dataset Used | Open In Colab                                                                                                                                                                          |
|------------------------|-----------|--------------------|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Clinical**           | OpenAI    | Text-Generation   | Clinical     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Clinical.ipynb) |


<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## Config Used

```yml 

model_parameters:
  temperature: 0
  max_tokens: 1600

tests:
  defaults:
    min_pass_rate: 1.0

  clinical:
    demographic-bias:
      min_pass_rate: 0.70


```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Supported Tests

- **`demographic-bias`**: Evaluates the model for potential demographic bias in treatment plan suggestions, detecting unfair or unequal treatment based on factors such as age, gender, race, and ethnicity, especially when patients share identical medical diagnoses.


</div></div>

