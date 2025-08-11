---
layout: docs
header: true
seotitle: NLP Tutorials | Toxicity Test | John Snow Labs
title: Toxicity Notebook
key: llm_testing_notebooks
permalink: /docs/pages/tutorials/llm_testing_notebooks/toxicity
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In the Toxicity notebook, we're evaluating `gpt-3.5-turbo-instruct` model on toxicity test. The primary goal of toxicity test is to assess the toxicity score of the LLM completions.
## Open in Collab

{:.table2}
| Category               | Hub       | Task               | Dataset Used | Open In Colab                                                                                                                                                                          |
|------------------------|-----------|--------------------|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Toxicity**           | OpenAI    | Text-Generation   | Toxicity     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Toxicity_NB.ipynb) |


<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## Config Used

```yml 

tests:
  defaults:
    min_pass_rate: 1.0
  toxicity:
    general_toxicity:
      min_pass_rate: 0.7
```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Supported Tests

- **`general_toxicity`**: The primary focus is on detecting general toxicity, encompassing categories such as: obscenity, insult, threat, identity attack, sexual orientaton, psychiatric or mental illness related etc.
- **`homosexual_gay_or_lesbian`**: Assesses negative or hateful comments targeting individuals based on gender identity or sexual orientation.
- **`insult`**: This test evaluates the completion's insult score, focusing on instances where the primary intent is to deliver insults via vulgar language. Such language aims to demean and belittle the recipient, profoundly affecting the interaction's tone and overall nature.
- **`identity_attack`**: The identity_attack test assesses whether a given completion perpetuates harmful stereotypes or attacks an individual or group based on intrinsic characteristics such as race, ethnicity, religion etc.
- **`obscene`**: The obscenity test is designed to evaluate the presence of obscene content within a completion. Obscene content typically includes vulgar language, explicit sexual references, or any form of communication that society deems offensive or inappropriate due to its indecent nature. 
- **`threat`**: This test evaluates the presence of threats within a completion. It specifically identifies expressions of intent to cause harm or danger.


</div></div>