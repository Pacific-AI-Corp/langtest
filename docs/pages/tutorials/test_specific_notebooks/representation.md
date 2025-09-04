---
layout: docs
header: true
seotitle: NLP Tutorials | Representation Test | John Snow Labs
title: Representation Notebook
key: test_specific_notebooks
permalink: /docs/pages/tutorials/test_specific_notebooks/representation
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In this representation notebook, we are evaluating `ner.dl` model on representation tests. The goal of representation testing is to determine if a given dataset represents a specific population accurately or if it contains biases that could negatively impact the results of any analysis conducted on it.


## Open in Collab

{:.table2}
| Category               | Hub                           | Task                              | Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
| **Representation** | John Snow Labs                    | NER                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/test-specific-notebooks/Representation_Demo.ipynb)                                |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Config Used

```yml 
tests:     
  defaults:
    min_pass_rate: 0.55
  representation:
    min_religion_name_representation_count:
      min_count:
        christian: 10
        muslim: 5
        hindu: 15

    min_label_representation_proportion:
      min_proportion:
          O: 0.5
          LOC: 0.2
```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Supported Tests

- **`min_gender_representation_count`**: Determine if any gender(male, female or unknown) has less than the desired minimum representation count.

- **`min_gender_representation_proportion`**:  Determine if any gender(male, female or unknown) has less than the desired minimum representation proportion.

- **`min_ethnicity_name_representation_count`**: Determine if any ethnicity(black, asian, white, native_american, hispanic or inter_racial) has less than the desired minimum representation count.

- **`min_ethnicity_name_representation_proportion`**: Determine if any ethnicity(black, asian, white, native_american, hispanic or inter_racial) has less than the desired minimum representation proportion.

- **`min_label_representation_count`**: Determine if any label(O, LOC, PER, MISC or ORG) has less than the desired minimum representation count.

- **`min_label_representation_proportion`**: Determine if any label(O, LOC, PER, MISC or ORG) has less than the desired minimum representation proportion.

- **`min_religion_name_representation_count`**: Determine if any religion(muslim, hindu, sikh, christian, jain, buddhist or parsi) has less than the desired minimum representation count.

- **`min_religion_name_representation_proportion`**: Determine if any religion(muslim, hindu, sikh, christian, jain, buddhist or parsi) has less than the desired minimum representation proportion.

- **`min_country_economic_representation_count`**: Determine if any country(high_income, low_income, lower_middle_income or upper_middle_income) has less than the desired minimum representation count.

- **`min_country_economic_representation_proportion`**:Determine if any country(high_income, low_income, lower_middle_income or upper_middle_income) has less than the desired minimum representation proportion.

</div></div>