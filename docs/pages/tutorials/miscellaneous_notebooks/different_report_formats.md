---
layout: docs
header: true
seotitle: Tutorials | Reports | LangTest | John Snow Labs
title: Different Report Formats
key: miscellaneous_notebooks
permalink: /docs/pages/tutorials/miscellaneous_notebooks/different_report_formats
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

In the Different Report Formats notebook, the LangTest's capability to show and export reports in numerous formats is showcased. The notebook uses `en_core_web_sm` from spaCy with some default config and data to run the tests. Then the report is exported in different formats like xml, csv, html etc.

## Open in Collab

{:.table2}
| Category                                                                                                    | Hub   | Task | Open In Colab                                                                                                                                                                                             |
| ----------------------------------------------------------------------------------------------------------- |
| **Report Exportation**| Spacy | NER  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Different_Report_formats.ipynb) |

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
