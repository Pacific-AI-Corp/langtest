---
layout: docs
header: true
seotitle: NLP Tutorials | Legal Test | John Snow Labs
title: Legal Notebook
key: llm_testing_notebooks
permalink: /docs/pages/tutorials/llm_testing_notebooks/legal_tests
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

The Legal Benchmark Test evaluates a model's legal reasoning and comprehension, specifically gauging its ability to assess support strength in case summaries. In this tutorial, using the LegalSupport dataset, we scrutinize the model's performance with text passages making legal claims and two corresponding case summaries. This concise exploration aims to reveal the model's proficiency in navigating legal nuances and discerning support strength, offering valuable insights into its legal reasoning capabilities.

## Open in Collab

{:.table2}
| Category               | Hub       | Task               | Dataset Used | Open In Colab                                                                                                                                                                          |
|------------------------|-----------|--------------------|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|                                                 |
| **Legal-Tests** | OpenAI                            | Question-Answering   | Legal-Support | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Legal_Support.ipynb)                                          |                                

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## Config Used

```yml 
tests:
  defaults:
    min_pass_rate: 1.0

  legal:
    legal-support:
      min_pass_rate: 0.70
```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Supported Tests

- **`legal-support`**: It tests a model's ability to reason regarding the strength of support a particular case summary provides.


</div></div>

