---
layout: docs
header: true
seotitle: NLP Tutorials | Grammar Test | John Snow Labs
title: Grammar Notebook
key: test_specific_notebooks
permalink: /docs/pages/tutorials/test_specific_notebooks/grammar
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In the Grammar notebook, we're looking at how tough the `lvwerra/distilbert-imdb` model is. The Grammar Test assesses language models' proficiency in intelligently identifying and correcting intentional grammatical errors, ensuring refined language understanding and enhancing overall processing quality.

## Open in Collab

{:.table2}
| Category               | Hub                           | Task                              | Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
|  **Grammar**    | Huggingface                    | Text-Classification                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/test-specific-notebooks/Grammar_Demo.ipynb)                              |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Config Used

```yml 
tests:     
  defaults:
    min_pass_rate: 0.65
  grammar:
    paraphrase:
      min_score: 0.66  
```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Supported Tests

- **`paraphrase`**: Assesses a model's ability to identify and generate linguistic alternatives conveying the same meaning, evaluating its proficiency in understanding and expressing diverse language constructs within the Grammar Testing framework.

</div></div>