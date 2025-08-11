---
layout: docs
header: true
seotitle: LangTest - Deliver Safe and Effective Language Models | John Snow Labs
title: LangTest Release Notes
permalink: /docs/pages/docs/langtest_versions/release_notes_2_2_0
key: docs-release-notes
modify_date: 2024-04-02
---

<div class="h3-box" markdown="1">

## 2.2.0
------------------
## 📢 Highlights

John Snow Labs is excited to announce the release of LangTest 2.2.0! This update introduces powerful new features and enhancements to elevate your language model testing experience and deliver even greater insights.

- 🏆 **Model Ranking & Leaderboard**: LangTest introduces a comprehensive model ranking system. Use harness.get_leaderboard() to rank models based on various test metrics and retain previous rankings for historical comparison.

- 🔍 **Few-Shot Model Evaluation:** Optimize and evaluate your models using few-shot prompt techniques. This feature enables you to assess model performance with minimal data, providing valuable insights into model capabilities with limited examples.

- 📊 **Evaluating NER in LLMs:** This release extends support for Named Entity Recognition (NER) tasks specifically for Large Language Models (LLMs). Evaluate and benchmark LLMs on their NER performance with ease.

- 🚀 **Enhanced Data Augmentation:** The new DataAugmenter module allows for streamlined and harness-free data augmentation, making it simpler to enhance your datasets and improve model robustness.

- 🎯 **Multi-Dataset Prompts:** LangTest now offers optimized prompt handling for multiple datasets, allowing users to add custom prompts for each dataset, enabling seamless integration and efficient testing.

</div><div class="h3-box" markdown="1">

## 🔥 Key Enhancements:

### **🏆 Comprehensive Model Ranking & Leaderboard**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/benchmarks/Benchmarking_with_Harness.ipynb)
The new Model Ranking & Leaderboard system offers a comprehensive way to evaluate and compare model performance based on various metrics across different datasets. This feature allows users to rank models, retain historical rankings, and analyze performance trends.

**Key Features:**
- **Comprehensive Ranking**: Rank models based on various performance metrics across multiple datasets.
- **Historical Comparison**: Retain and compare previous rankings for consistent performance tracking.
- **Dataset-Specific Insights**: Evaluate model performance on different datasets to gain deeper insights.

**How It Works:**

The following are steps to do model ranking and visualize the leaderboard for `google/flan-t5-base` and `google/flan-t5-large` models.
**1.** Setup and configuration of the Harness are as follows:

```yaml
# config.yaml
model_parameters:
  max_tokens: 64
  device: 0
  task: text2text-generation
tests:
  defaults:
    min_pass_rate: 0.65
  robustness:
    add_typo:
      min_pass_rate: 0.7
    lowercase:
      min_pass_rate: 0.7
```
```python
from langtest import Harness

harness = Harness(
    task="question-answering",
    model={
        "model": "google/flan-t5-base",
        "hub": "huggingface"
    },
    data=[
        {
            "data_source": "MedMCQA"
        },
        {
            "data_source": "PubMedQA"
        },
        {
            "data_source": "MMLU"
        },
        {
            "data_source": "MedQA"
        }
    ],
    config="config.yml",
    benchmarking={
        "save_dir":"~/.langtest/leaderboard/" # required for benchmarking 
    }
)
```

**2**. generate the test cases, run on the model, and get the report as follows:
```python
harness.generate().run().report()
```
![image](https://github.com/JohnSnowLabs/langtest/assets/23481244/d8055592-5501-4139-ad90-55baa4fecbfc)

**3**. Similarly, do the same steps for the `google/flan-t5-large` model with the same `save_dir` path for benchmarking and the same `config.yaml`

**4**. Finally, the leaderboard can show the model rank by calling the below code.
```python
harness.get_leaderboard()
```
![image](https://github.com/JohnSnowLabs/langtest/assets/23481244/ff741d8e-4fc0-4f94-bcc3-9c67653aaba8)

**Conclusion:**
The Model Ranking & Leaderboard system provides a robust and structured method for evaluating and comparing models across multiple datasets, enabling users to make data-driven decisions and continuously improve model performance.


### **🔍 Efficient Few-Shot Model Evaluation**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Fewshot_QA_Notebook.ipynb)
Few-Shot Model Evaluation optimizes and evaluates model performance using minimal data. This feature provides rapid insights into model capabilities, enabling efficient assessment and optimization with limited examples.

**Key Features:**
- **Few-Shot Techniques**: Evaluate models with minimal data to gauge performance quickly.
- **Optimized Performance**: Improve model outputs using targeted few-shot prompts.
- **Efficient Evaluation**: Streamlined process for rapid and effective model assessment.

**How It Works:**
**1.** Set up few-shot prompts tailored to specific evaluation needs.
```yaml
# config.yaml
prompt_config:
  "BoolQ":
    instructions: >
      You are an intelligent bot and it is your responsibility to make sure 
      to give a concise answer. Answer should be `true` or `false`.
    prompt_type: "instruct" # instruct for completion and chat for conversation(chat models)
    examples:
      - user:
          context: >
            The Good Fight -- A second 13-episode season premiered on March 4, 2018. 
            On May 2, 2018, the series was renewed for a third season.
          question: "is there a third series of the good fight?"
        ai:
          answer: "True"
      - user:
          context: >
            Lost in Space -- The fate of the castaways is never resolved, 
            as the series was unexpectedly canceled at the end of season 3.
          question: "did the robinsons ever get back to earth"
        ai:
          answer: "True"
  "NQ-open":
    instructions: >
      You are an intelligent bot and it is your responsibility to make sure 
      to give a short concise answer.
    prompt_type: "instruct" # completion
    examples:
      - user:
          question: "where does the electron come from in beta decay?"
        ai:
          answer: "an atomic nucleus"
      - user:
          question: "who wrote you're a grand ol flag?"
        ai:
          answer: "George M. Cohan"

tests:
  defaults:
    min_pass_rate: 0.8
  robustness:
    uppercase:
      min_pass_rate: 0.8
    add_typo:
      min_pass_rate: 0.8
```
**2.** Initialize the Harness with `config.yaml` file as below code
```python
harness = Harness(
                  task="question-answering", 
                  model={"model": "gpt-3.5-turbo-instruct","hub":"openai"}, 
                  data=[{"data_source" :"BoolQ",
                        "split":"test-tiny"},
                        {"data_source" :"NQ-open",
                         "split":"test-tiny"}],
                  config="config.yaml"
                  )
```
**3.** Generate the test cases, run them on the model, and then generate the report.

```python
harness.generate().run().report()
```
![image](https://github.com/JohnSnowLabs/langtest/assets/23481244/4bae4008-621c-4d1c-a303-218f9df2700d)

**Conclusion:**
Few-Shot Model Evaluation provides valuable insights into model capabilities with minimal data, allowing for rapid and effective performance optimization. This feature ensures that models can be assessed and improved efficiently, even with limited examples.


### **📊 Evaluating NER in LLMs**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/NER%20Casual%20LLM.ipynb)
Evaluating NER in LLMs enables precise extraction and evaluation of entities using Large Language Models (LLMs). This feature enhances the capability to assess LLM performance on Named Entity Recognition tasks.

**Key Features:**
- **LLM-Specific Support**: Tailored for evaluating NER tasks using LLMs.
- **Accurate Entity Extraction**: Improved techniques for precise entity extraction.
- **Comprehensive Evaluation**: Detailed assessment of entity extraction performance.

**How It Works:**
**1.** Set up NER tasks for specific LLM evaluation.
```python
# Create a Harness object
harness = Harness(task="ner",
            model={
                "model": "gpt-3.5-turbo-instruct",
                "hub": "openai", },
            data={
                "data_source": 'path/to/conll03.conll'
            },
            config={
                "model_parameters": {
                    "temperature": 0,
                },
                "tests": {
                    "defaults": {
                        "min_pass_rate": 1.0
                    },
                    "robustness": {
                        "lowercase": {
                            "min_pass_rate": 0.7
                        }
                    },
                    "accuracy": {
                        "min_f1_score": {
                            "min_score": 0.7,
                        },
                    }
                }
            }
            )
```
**2.** Generate the test cases based on the configuration in the Harness, run them on the model, and get the report.
```python
harness.generate().run().report()
```
![image](https://github.com/JohnSnowLabs/langtest/assets/23481244/9435fa17-d3f7-4d47-934c-4cd483b11a53)

Examples:
![image](https://github.com/JohnSnowLabs/langtest/assets/23481244/2ceb3390-9f07-4b17-b9e7-b32504ad1afe)

**Conclusion:**
Evaluating NER in LLMs allows for accurate entity extraction and performance assessment using LangTest's comprehensive evaluation methods. This feature ensures thorough and reliable evaluation of LLMs on Named Entity Recognition tasks.


### **🚀 Enhanced Data Augmentation**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Data_Augmenter_Notebook.ipynb)
Enhanced Data Augmentation introduces a new `DataAugmenter` class, enabling streamlined and harness-free data augmentation. This feature simplifies the process of enriching datasets to improve model robustness and performance.

**Key Features:**
- **Harness-Free Augmentation**: Perform data augmentation without the need for harness testing.
- **Improved Workflow**: Simplified processes for enhancing datasets efficiently.
- **Robust Models**: Increase model robustness through effective data augmentation techniques.

**How It Works:**
The following are steps to import the `DataAugmenter` class from LangTest.
**1.** Create a config.yaml for the data augmentation.
```yaml
# config.yaml
parameters:
    type: proportion
    style: new
tests:
    robustness:
        uppercase:
            max_proportion: 0.2
        lowercase:
            max_proportion: 0.2

```
**2.** Initialize the `DataAugmenter` class and apply various tests for augmentation to your datasets.
```python
from langtest.augmentation import DataAugmenter
from langtest.tasks.task import TaskManager

data_augmenter = DataAugmenter(
    task=TaskManager("ner"), # use the ner, text-classification, question-answering...
    config="config.yaml",
)
```
**3.** Provide the training dataset to `data_augmenter`.
```python
data_augmenter.augment(data={
    'data_source': 'path/to/conll03.conll'
})
```
**4.** Then, save the augmented dataset. 
```
data_augmenter.save("augmented.conll")
```
**Conclusion:**
Enhanced Data Augmentation capabilities in LangTest ensure that your models are more robust and capable of handling diverse data scenarios. This feature simplifies the augmentation process, leading to improved model performance and reliability.


### **🎯Multi-Dataset Prompts**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/MultiPrompt_MultiDataset.ipynb)
Multi-Dataset Prompts streamline the process of integrating and testing various data sources by allowing users to define custom prompts for each dataset. This enhancement ensures efficient prompt handling across multiple datasets, enabling comprehensive performance evaluations.

**Key Features:**

- **Custom Prompts:** Add tailored prompts for each dataset to enhance testing accuracy.
- **Seamless Integration:** Easily incorporate multiple datasets into your testing environment.
- **Improved Efficiency:** Simplified workflows for handling diverse data sources.

**How It Works:**
**1.** Initiate the Harness with `BoolQ` and `NQ-open` datasets.
```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(
    task="question-answering",
    model={"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
    data=[
        {"data_source": "BoolQ", "split": "dev-tiny"},
        {"data_source": "NQ-open", "split": "test-tiny"}
    ],
)
```
**2.** Configure prompts specific to each dataset, allowing tailored evaluations.
```python
harness.configure(
    {
        "model_parameters": {
            "user_prompt": {
                "BoolQ": "Answer the following question with a True or False. {context}\nQuestion {question}",
                "NQ-open": "Answer the following question. Question {question}",
            }
        },
        "tests": {
            "defaults": {"min_pass_rate": 0.65},
            "robustness": {
                "uppercase": {"min_pass_rate": 0.66},
                "dyslexia_word_swap": {"min_pass_rate": 0.60},
                "add_abbreviation": {"min_pass_rate": 0.60},
                "add_slangs": {"min_pass_rate": 0.60},
                "add_speech_to_text_typo": {"min_pass_rate": 0.60},
            },
        }
    }
)
```
**3.** Generate the test cases, run them on the model, and get the report.
```python
harness.generate().run().report()
```
![image](https://github.com/JohnSnowLabs/langtest/assets/23481244/a961d98d-a229-439e-a9eb-92395dde6f62)

**Conclusion:**
Multi-dataset prompts in LangTest empower users to efficiently manage and test multiple data sources, resulting in more effective and comprehensive language model evaluations.

## 📒 New Notebooks

{:.table2}
| Notebooks          | Colab Link |
|--------------------|-------------|
| Model Ranking & Leaderboard       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/benchmarks/Benchmarking_with_Harness.ipynb)|
| Fewshot Model Evaluation     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Fewshot_QA_Notebook.ipynb) |
| Evaluating NER in LLMs    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/NER%20Casual%20LLM.ipynb) |
| Data Augmenter    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Data_Augmenter_Notebook.ipynb) |
| Multi-Dataset Prompts   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/MultiPrompt_MultiDataset.ipynb) |


## 🐛 Fixes

- Fixed bugs in Random Age test [#1020]
- Fixed bugs in Performance tests [#1015]

## ⚡ Enhancements

- Improved the importing edit_testcases into Harness [#1022]
- Code Organization and Readability in Augmentation Module [#1025]

## What's Changed
* User prompt handling for multi-dataset testing by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1010
* Bug fix/performance tests by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1015
* NER task support for casuallm models from huggingface, web, and lm-studio by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1009
* `random_age` Class not returning test cases by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1020
* Feature/data augmentation allow access without harness testing by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1016
* Improvements/load and save benchmark report by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1012
* Refactor: Improved the `import_edited_testcases()` functionality in Harness. by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1022
* Implementation of prompt techniques by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1018
* Fix: Summary class to update summary dataframe and handle file path by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1024
* Refactor: Improve Code Organization and Readability by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1025
* Improved: `rank_by` argument add to `harness.get_leaderboard()` by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1027
* website updates by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1023
* updated: langtest version in pip by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1028
* Release/2.2.0 by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1029


**Full Changelog**: https://github.com/JohnSnowLabs/langtest/compare/2.1.0...2.2.0

</div>
{%- include docs-langtest-pagination.html -%}
