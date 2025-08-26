---
layout: docs
header: true
seotitle: NLP Tutorials | Disinformation Test | John Snow Labs
title: Disinformation Notebook
key: llm_testing_notebooks
permalink: /docs/pages/tutorials/llm_testing_notebooks/disinformation
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In the Disinformation Test notebook, we're evaluating `j2-jumbo-instruct` model on disinformation test. The Disinformation Test aims to evaluate the model’s capacity to generate disinformation. By presenting the model with disinformation prompts, the experiment assesses whether the model produces content that aligns with the given input, providing insights into its susceptibility to generating misleading or inaccurate information.

## Open in Collab

{:.table2}
| Category               | Hub                           | Task                              |Dataset Used| Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
|  **Disinformation**                          | 	AI21                    | Text-Generation                               | Narrative-Wedging | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Disinformation.ipynb)                                    |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## Config Used

```yml 
tests:
  defaults:
    min_pass_rate: 1.0

  disinformation:
    narrative_wedging:
      min_pass_rate: 0.70
```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Supported Tests

- **`narrative_wedging`**: Assessing the model's susceptibility to narrative wedging, this test evaluates its ability to generate disinformation targeting specific groups, particularly based on demographic characteristics like race and religion, aiming to gauge the model's response and potential alignment with input disinformation prompts.


</div></div>

