---
layout: docs
header: true
seotitle: NLP Tutorials |  Performance Test | John Snow Labs
title: Performance Notebook
key: test_specific_notebooks
permalink: /docs/pages/tutorials/test_specific_notebooks/performance
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In this performance notebook, we are evaluating `dslim/bert-base-NER` model on Performance test. The goal of performance testing is to determine the average time taken by the model to give the complete response.

## Open in Collab

{:.table2}
| Category               | Hub                           | Task                              | Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
| **Performance** | Huggingface                    | NER                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/PerformanceTest_Notebook.ipynb)                                |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Config Used

```yml 
tests:
  defaults:
    min_pass_rate: 0.65
  robustness:
    lowercase:
      min_pass_rate: 0.66
    uppercase:
      min_pass_rate: 0.66
  performance:
    speed:
      min_pass_rate: 100
      unit: 'token/sec'
```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

