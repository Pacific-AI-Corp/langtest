---
layout: docs
header: true
seotitle: Tutorials | Model Comparison | LangTest | John Snow Labs
title: Comparing Models
key: miscellaneous_notebooks
permalink: /docs/pages/tutorials/miscellaneous_notebooks/comparing_models
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
show_edit_on_github: true
modify_date: "2023-11-15"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In the Comparing Models notebook, we're evaluating `en.sentiment.imdb.glove` model from JSL and `lvwerra/distilbert-imdb` from HF which are models trained for text classification task. The notebook showcases how we can get a view of how the models compare. The final report output is different than normal and includes both models's results.

## Open in Collab

{:.table2}
| Category                                                                                                | Hub                               | Task                    | Open In Colab                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------- |
| **Multiple Model Comparison**| Hugging Face/John Snow Labs/Spacy | NER/Text-Classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Comparing_Models_Notebook.ipynb) |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## Config Used

```yml 
tests:
  defaults:
    min_pass_rate: 1.0
  robustness:
    add_typo:
      min_pass_rate: 0.7
    american_to_british:
      min_pass_rate: 0.7
  accuracy:
    min_micro_f1_score:
      min_score: 0.7
  bias:
    replace_to_female_pronouns:
      min_pass_rate: 0.7
    replace_to_low_income_country:
      min_pass_rate: 0.7
  fairness:
    min_gender_f1_score:
      min_score: 0.6
  representation:
    min_label_representation_count:
      min_count: 50
```

