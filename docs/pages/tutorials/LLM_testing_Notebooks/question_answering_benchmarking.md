---
layout: docs
header: true
seotitle: NLP Tutorials | Question-Answering Benchmarking | John Snow Labs
title: Question Answering Benchmarking
key: llm_testing_notebooks
permalink: /docs/pages/tutorials/llm_testing_notebooks/question_answering_benchmarking
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

This notebook provides a comprehensive overview of benchmarking Language Models (LLMs) in Question-Answering tasks. we are dpoing  conducting robustness and accuracy tests to evaluate LLM performance. we're conducting Robustness and Accuracy testing on the `mistralai/Mistral-7B-Instruct-v0.1` model for the OpenBookQA dataset. 
### Open in Collab

{:.table2}
| Category               | Hub                           | Task                              |  Datset Used                                                                                                                                                                                                                                   | Open In Colab |
| ----------------------------------- |
| **Robustness**, **Accuracy** | Hugging Face Inference API                            | Question-Answering  | `OpenBookQA` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/benchmarks/Question-Answering.ipynb)               |

<div class="main-docs" markdown="1">



## Config Used

```yml
evaluation:
  hub: openai
  metric: llm_eval
  model: gpt-3.5-turbo-instruct
model_parameters:
  max_tokens: 32
  user_prompt: "You are an AI bot specializing in providing accurate and concise answers\
    \ to questions. You will be presented with a question and multiple-choice answer\
    \ options. Your task is to choose the correct answer.\nNote: Do not explain your\
    \ answer.\nQuestion: {question}\nOptions: {options}\n Answer:"
tests:
  defaults:
    min_pass_rate: 0.65
  robustness:
    add_abbreviation:
      min_pass_rate: 0.75
    add_ocr_typo:
      min_pass_rate: 0.75
    add_slangs:
      min_pass_rate: 0.75
    add_speech_to_text_typo:
      min_pass_rate: 0.75
    add_typo:
      min_pass_rate: 0.75
    adjective_synonym_swap:
      min_pass_rate: 0.75
    dyslexia_word_swap:
      min_pass_rate: 0.75
    lowercase:
      min_pass_rate: 0.75
    titlecase:
      min_pass_rate: 0.75
    uppercase:
      min_pass_rate: 0.75
```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


