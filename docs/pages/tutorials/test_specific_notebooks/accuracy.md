---
layout: docs
header: true
seotitle: NLP Tutorials | Accuracy Test | John Snow Labs
title: Accuracy Notebook
key: test_specific_notebooks
permalink: /docs/pages/tutorials/test_specific_notebooks/accuracy
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

The Accuracy notebook is all about looking closely at how the ner.dl model does its thing. We're checking out if it's good at pinpointing stuff through precision, recall, and F1 score, especially for different labels. We throw it into a tough test using data it hasn't seen before, comparing what it thinks will happen with what actually goes down. And it's not just about the labels; we also dig into bigger metrics like micro F1, macro F1, and weighted F1 scores. This helps us see the bigger picture of how well the model deals with all kinds of data sorting challenges.

This deep dive into the ner.dl model's performance isn't just to show off its good and not-so-good points. It's a practical tool to make the model work better. The goal of the Accuracy notebook is to give users real, useful insights into how accurate the model is. It's all about helping them make smart choices on when and where to use the model in real-life situations.

## Open in Collab

{:.table2}
| Category               | Hub                           | Task                              | Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
| **Accuracy** | John Snow Labs                    | NER                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/test-specific-notebooks/Accuracy_Demo.ipynb)                                |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Config Used

```yml 
tests:     
  defaults:
    min_pass_rate: 0.65
  accuracy:
    min_f1_score:
      min_score: 0.60
    min_precision_score:
      O: 0.60
      PER: 0.60
      LOC: 0.60
```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Supported Tests

- **`min_precision_score`**: Determine if the actual precision score is less than the desired precision score.

- **`min_recall_score`**:  Determine if the actual recall score is less than the desired recall score.

- **`min_f1_score`**: Determine if the actual f1 score is less than the desired f1 score.

- **`min_micro_f1_score`**:  Determine if the actual micro-f1 score is less than the desired micro-f1 score.

- **`min_macro_f1_score`**:  Determine if the actual macro-f1 score is less than the desired macro-f1 score.

- **`min_weighted_f1_score`**:  Determine if the actual min-weighted-f1 score is less than the desired min-weighted-f1 score.


</div></div>