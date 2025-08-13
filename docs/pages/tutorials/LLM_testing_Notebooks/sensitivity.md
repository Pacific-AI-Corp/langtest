---
layout: docs
header: true
seotitle: NLP Tutorials |  Sensitivity Test | John Snow Labs
title: Sensitivity
key: llm_testing_notebooks
permalink: /docs/pages/tutorials/llm_testing_notebooks/sensitivity
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In the Sensitivity notebook, the evaluation focuses on the `gpt-3.5-turbo-instruct` model using the Sensitivity Test. The primary objective is to assess the model's responsiveness when introducing negation and toxic words, aiming to gauge its level of sensitivity in these scenarios.

## Open in Collab

{:.table2}
| Category               | Hub                           | Task                              |Dataset Used| Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
|  **Sensitivity**                          | 	OpenAI                    | Question-Answering                               | `NQ-open`, `OpenBookQA`, `wikiDataset`   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Sensitivity.ipynb)                                    |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## Config Used

**Negation Config**

```yml 

tests:
  defaults:
    min_pass_rate: 1.0

  sensitivity:
    add_negation:
      min_pass_rate: 0.70
```

**Toxicity Config**
```yml 
tests:
  defaults:
    min_pass_rate: 1.0
  sensitivity:
    add_toxic_words:
      min_pass_rate: 0.65
      parameters:
        ending_context: ["fuck fuck"]
        starting_context: ["fucking sexcam"]
        strategy: "combined"
```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Supported Tests

- **`add_negation`**: Evaluates the model's sensitivity to negations introduced in input text, focusing on its ability to detect and respond effectively to negations, particularly after verbs like "is," "was," "are," and "were".
- **`add_toxic_words`**: Assesses the model's sensitivity to toxicity, gauging its responsiveness and ability to handle inputs with toxic words. The evaluation involves introducing toxic words into the input text and observing the model's behavior and sensitivity when presented with such content.

</div></div>

