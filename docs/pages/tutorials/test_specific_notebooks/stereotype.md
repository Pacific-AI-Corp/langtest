---
layout: docs
header: true
seotitle: NLP Tutorials | Stereotype Test | John Snow Labs
title: Stereotype Notebooks
key: test_specific_notebooks
permalink: /docs/pages/tutorials/test_specific_notebooks/stereotype
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

The primary goal of stereotype tests is to evaluate how well models perform when confronted with common gender stereotypes, occupational stereotypes, or other prevailing biases. In these assessments, models are scrutinized for their propensity to perpetuate or challenge stereotypical associations, shedding light on their capacity to navigate and counteract biases in their predictions.

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## CrowS Pairs Notebook

In this notebook we are measuring the degree to which stereotypical biases are present in masked language models using Crows Pairs dataset.

### Open in Collab

{:.table2}
| Category              | Hub                           | Task                              | Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
| **CrowS Pairs** | Hugging Face                      | Fill-Mask                     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/task-specific-notebooks/Crows_Pairs_Notebook.ipynb)                         |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

### Config Used

```yml 
tests:
  defaults:
    min_pass_rate: 1.0

  stereotype:
    crows-pairs:
      min_pass_rate: 0.70
      diff_threshold: 0.10
      filter_threshold: 0.15
```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Wino-Bias HuggingFace Notebook

In this tutorial, we assess the model on gender occupational stereotype statements using Hugging Face fill mask models.

### Open in Collab

{:.table2}
| Category               | Hub                           | Task                              | Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
| **Wino-Bias**| Hugging Face                      | Fill-Mask                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/task-specific-notebooks/Wino_Bias.ipynb)                                    |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

### Config Used

```yml 
tests:
  defaults:
    min_pass_rate: 1.0

  stereotype:
    wino-bias:
      min_pass_rate: 0.70
      diff_threshold: 0.03

```

</div></div>