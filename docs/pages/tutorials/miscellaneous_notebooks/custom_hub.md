---
layout: docs
header: true
seotitle: Tutorials | Custom Hub | LangTest | John Snow Labs
title: Custom Hub
key: miscellaneous_notebooks
permalink: /docs/pages/tutorials/miscellaneous_notebooks/custom_hub
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

In the Custom Hub notebook, we're evaluating our very own trained model independent from hubs. The main focus is showing we can run the tests with any possible model/framework easily if we have a predict function in the similar format. The notebook showcases an implementation of an LSTM model for text classification task trained using pytorch. After creating the harness object with `hub`:`custom` parameter, we can continue to use it as always.

## Open in Collab

{:.table2}
| Category                                                                                 | Hub    | Task                | Open In Colab                                                                                                                                                                                        |
| ---------------------------------------------------------------------------------------- |
| **Custom Hub**| Custom | Text-Classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Custom_Hub_Notebook.ipynb) |

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
