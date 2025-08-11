---
layout: docs
header: true
seotitle: NLP Tutorials | Bias Test | John Snow Labs
title: Bias Notebook
key: test_specific_notebooks
permalink: /docs/pages/tutorials/test_specific_notebooks/bias
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In this Bias notebook, our main goal is checking if the ner.dl model has any biases. Bias here means the model might lean in a certain direction consistently, like favoring one gender, ethnicity, religion, or country over others. This could lead to issues, like sticking to stereotypes or being unfair to certain groups.

We're changing things, like using different names or countries, and checking how it messes with the model's predictions. The idea is to comprehend if the model's leaning one way and how we can tweak it to make sure it's fair and works well for everyone.

## Open in Collab

{:.table2}
| Category               | Hub                           | Task                              | Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
|  **Bias**                          | John Snow Labs                    | NER                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/test-specific-notebooks/Bias_Demo.ipynb)                                    |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Config Used

```yml 
tests:     
  defaults:
    min_pass_rate: 0.65
  bias:
    replace_to_female_pronouns:
      min_pass_rate: 0.66
    replace_to_hindu_names:
      min_pass_rate: 0.60
```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## Supported Tests

- **`replace_to_male_pronouns`**: female/neutral pronouns of the test set are turned into male pronouns.

- **`replace_to_female_pronouns`**: male/neutral pronouns of the test set are turned into female pronouns.

- **`replace_to_neutral_pronouns`**: female/male pronouns of the test set are turned into neutral pronouns.

- **`replace_to_high_income_country`**: replace countries in test set to high income countries.

- **`replace_to_low_income_country`**: replace countries in test set to low income countries.
- **`replace_to_upper_middle_income_country`**: replace countries in test set to upper middle income countries.

- **`replace_to_lower_middle_income_country`**: replace countries in test set to lower middle income countries.

- **`replace_to_white_firstnames`**: replace other ethnicity first names to white firstnames.

- **`replace_to_black_firstnames`**: replace other ethnicity first names to black firstnames.

- **`replace_to_hispanic_firstnames`**: replace other ethnicity first names to hispanic firstnames.

- **`replace_to_asian_firstnames`**: replace other ethnicity first names to asian firstnames.

- **`replace_to_white_lastnames`**: replace other ethnicity last names to white lastnames.

- **`replace_to_black_lastnames`**: replace other ethnicity last names to black lastnames.

- **`replace_to_hispanic_lastnames`**: replace other ethnicity last names to hispanic lastnames.

- **`replace_to_asian_lastnames`**: replace other ethnicity last names to asian lastnames.

- **`replace_to_native_american_lastnames`**: replace other ethnicity last names to native-american lastnames.

- **`replace_to_inter_racial_lastnames`**: replace other ethnicity last names to inter-racial lastnames.

- **`replace_to_muslim_names`**: replace other religion people names to muslim names.

- **`replace_to_hindu_names`**:  replace other religion people names to hindu names.

- **`replace_to_christian_names`**:  replace other religion people names to christian names.

- **`replace_to_sikh_names`**:  replace other religion people names to sikh names.

- **`replace_to_jain_names`**:  replace other religion people names to jain names.

- **`replace_to_parsi_names`**:  replace other religion people names to parsi names.

- **`replace_to_buddhist_names`**:  replace other religion people names to buddhist names.

</div></div>