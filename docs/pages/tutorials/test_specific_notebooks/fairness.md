---
layout: docs
header: true
seotitle: NLP Tutorials | Fairness Test | John Snow Labs
title: Fairness Notebook
key: test_specific_notebooks
permalink: /docs/pages/tutorials/test_specific_notebooks/fairness
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In the Fairness notebook, we're digging into how the ner.dl model treats different groups. Fairness testing is a big deal because we want to make sure the model isn't favoring or working against specific groups of people. The idea is to run tests that check if the model gives fair results to everyone, regardless of things like gender. So, in this notebook, we're putting the ner.dl model through fairness tests to see how it handles and treats various groups. The goal is to make sure it's fair and square in its predictions, with no bias towards any particular bunch.

## Open in Collab

{:.table2}
| Category               | Hub                           | Task                              | Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
|  **Fairness**   | John Snow Labs                    | NER                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/test-specific-notebooks/Fairness_Demo.ipynb)                                |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Config Used

```yml 
tests:     
  defaults:
    min_pass_rate: 0.65
  fairness:
    min_gender_f1_score:
      min_score: 0.66  
    max_gender_f1_score:
      max_score:
        male: 0.99
        female: 0.95
```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## Supported Tests

- **`min_gender_f1_score`**: Determine if any gender(male, female or unknown) has less than the desired f1 score.

- **`max_gender_f1_score`**:  Determine if any gender(male, female or unknown) has more than the desired f1 score.

- **`max_gender_rouge1_score`**: This test evaluates the model for each gender seperately. The rouge1 score for each gender is calculated and test is passed if they are higher than config.

- **`max_gender_rouge2_score`**: This test evaluates the model for each gender seperately. The rouge2 score for each gender is calculated and test is passed if they are higher than config.

- **`max_gender_rougeL_score`**: This test evaluates the model for each gender seperately. The rougeL score for each gender is calculated and test is passed if they are higher than config.

- **`max_gender_rougeLsum_score`**: This test evaluates the model for each gender seperately. The rougeLsum score for each gender is calculated and test is passed if they are higher than config.

- **`min_gender_rouge1_score`**: This test evaluates the model for each gender seperately. The rouge1 score for each gender is calculated and test is passed if they are higher than config.

- **`min_gender_rouge2_score`**: This test evaluates the model for each gender seperately. The rouge2 score for each gender is calculated and test is passed if they are higher than config.

</div></div>