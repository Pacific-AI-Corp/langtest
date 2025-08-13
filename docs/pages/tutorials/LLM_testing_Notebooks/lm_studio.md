---
layout: docs
header: true
seotitle: NLP Tutorials | LM Studio | John Snow Labs
title: LM Studio
key: llm_testing_notebooks
permalink: /docs/pages/tutorials/llm_testing_notebooks/lm_studio
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In the notebook, we are conducting robustness and accuracy testing on the `TheBloke/neural-chat-7B-v3-1-GGUF` model using the OpenBookQA dataset. Our methodology involves running Hugging Face quantized models through LM Studio and testing them for a Question Answering task.

### Open in Collab

{:.table2}
| Category               | Hub                           | Task                              |  Datset Used                                                                                                                                                                                                                                   | Open In Colab |
| ----------------------------------- |
| **Robustness**, **Accuracy** | LM Studio                            | Question-Answering  | `OpenBookQA` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/LM-Studio-Demo.ipynb)               |

<div class="main-docs" markdown="1">



## Config Used

```yml
evaluation:
  hub: openai
  metric: llm_eval
  model: gpt-3.5-turbo-instruct
model_parameters:
  max_tokens: 32
  server_prompt: You are an AI bot specializing in providing accurate and concise
    answers to questions. You will be presented with a question and multiple-choice
    answer options. Your task is to choose the correct answer. Ensure that your response
    includes only the correct answer and no additional details.
  stream: false
  temperature: 0.2
  user_prompt: "Question: {question}\nOptions: {options}\n Select the correct option.\
    \ Keep your response short and precise. Avoid additional explanations.\nYour Answer:"
tests:
  defaults:
    min_pass_rate: 0.65
  robustness:
    add_ocr_typo:
      min_pass_rate: 0.75
    add_speech_to_text_typo:
      min_pass_rate: 0.75
    uppercase:
      min_pass_rate: 0.75

```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


