---
layout: docs
header: true
seotitle: LangTest - Deliver Safe and Effective Language Models | John Snow Labs
title: LangTest Release Notes
permalink: /docs/pages/docs/langtest_versions/release_notes_2_1_0
key: docs-release-notes
modify_date: 2024-04-02
---

<div class="h3-box" markdown="1">

## 2.1.0

## 📢 Highlights

John Snow Labs is thrilled to announce the release of LangTest 2.1.0! This update brings exciting new features and improvements designed to streamline your language model testing workflows and provide deeper insights.

- **🔗 Enhanced API-based LLM Integration:** LangTest now supports testing API-based Large Language Models (LLMs). This allows you to seamlessly integrate diverse LLM models with LangTest and conduct performance evaluations across various datasets.

- **📂 Expanded File Format Support:** LangTest 2.1.0 introduces support for additional file formats, further increasing its flexibility in handling different data structures used in LLM testing.

- **📊 Improved Multi-Dataset Handling:** We've made significant improvements in how LangTest manages multiple datasets. This simplifies workflows and allows for more efficient testing across a wider range of data sources.

- **🖥️ New Benchmarking Commands**: LangTest now boasts a set of new commands specifically designed for benchmarking language models. These commands provide a structured approach to evaluating model performance and comparing results across different models and datasets.

</div><div class="h3-box" markdown="1">

## 🔥 Key Enhancements:

### **🔗 Streamlined Integration and Enhanced Functionality for API-Based Large Language Models:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Generic_API-Based_Model_Testing_Demo.ipynb)

This feature empowers you to seamlessly integrate virtually any language model hosted on an external API platform. Whether you prefer OpenAI, Hugging Face, or even custom vLLM solutions, LangTest now adapts to your workflow. `input_processor` and `output_parser` functions are not required for openai api compatible server.

#### Key Features:

- **Effortless API Integration:** Connect to any API system by specifying the API URL, parameters, and a custom function for parsing the returned results. This intuitive approach allows you to leverage your preferred language models with minimal configuration.

- **Customizable Parameters:** Define the URL, parameters specific to your chosen API, and a parsing function tailored to extract the desired output. This level of control ensures compatibility with diverse API structures.

- **Unparalleled Flexibility:** Generic API Support removes platform limitations. Now, you can seamlessly integrate language models from various sources, including OpenAI, Hugging Face, and even custom vLLM solutions hosted on private platforms.

#### How it Works:

**Parameters:**
Define the `input_processer` function for creating a payload and the `output_parser` function is used to extract the output from the response.

```python
GOOGLE_API_KEY = "<YOUR API KEY>"
model_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GOOGLE_API_KEY}"

# headers
headers = {
    "Content-Type": "application/json",
}

# function to create a payload
def input_processor(content):
    return {"contents": [
        {
            "role": "user",
            "parts": [
                {
                    "text": content
                }
            ]
        }
    ]}

# function to extract output from model response
def output_parser(response):
    try:
        return response['candidates'][0]['content']['parts'][0]['text']
    except:
        return ""
```

To take advantage of this feature, users can utilize the following setup code:

```python
from langtest import Harness

# Initialize Harness with API parameters
harness = Harness(
    task="question-answering",
    model={
        "model": {
            "url": url,
            "headers": headers,
            "input_processor": input_processor,
            "output_parser": output_parser,
        },
        "hub": "web",
    },
    data={
        "data_source": "OpenBookQA",
        "split": "test-tiny",
    }
)
# Generate, Run and get Report
harness.generate().run().report()
```
![image](https://github.com/JohnSnowLabs/langtest/assets/23481244/9754c506-e715-4e2c-8b9d-dfd98f0695e5)


### 📂 Streamlined Data Handling and Evaluation

This feature streamlines your testing workflows by enabling LangTest to process a wider range of file formats directly.

#### Key Features:

- **Effortless File Format Handling:** LangTest now seamlessly ingests data from various file formats, including pickles (.pkl) in addition to previously supported formats. Simply provide the data source path in your harness configuration, and LangTest takes care of the rest.

- **Simplified Data Source Management**: LangTest intelligently recognizes the file extension and automatically selects the appropriate processing method. This eliminates the need for manual configuration, saving you time and effort.

- **Enhanced Maintainability**: The underlying code structure is optimized for flexibility. Adding support for new file formats in the future requires minimal effort, ensuring LangTest stays compatible with evolving data storage practices.

#### How it works:

```python
from langtest import Harness 

harness = Harness(
    task="question-answering",
    model={
        "model": "http://localhost:1234/v1/chat/completions",
        "hub": "lm-studio",
    },
    data={
        "data_source": "path/to/file.pkl", #
    },
)
# generate, run and report
harness.generate().run().report()
```
### 📊 Multi-Dataset Handling and Evaluation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Multiple_dataset.ipynb)

This feature empowers you to efficiently benchmark your language models across a wider range of datasets.

#### Key Features:

- **Effortless Multi-Dataset Testing:** LangTest now seamlessly integrates and executes tests on multiple datasets within a single harness configuration. This streamlined approach eliminates the need for repetitive setups, saving you time and resources.

- **Enhanced Fairness Evaluation**: By testing models across diverse datasets, LangTest helps identify and mitigate potential biases. This ensures your models perform fairly and accurately on a broader spectrum of data, promoting ethical and responsible AI development.

- **Robust Accuracy Assessment:** Multi-dataset support empowers you to conduct more rigorous accuracy testing. By evaluating models on various datasets, you gain a deeper understanding of their strengths and weaknesses across different data distributions. This comprehensive analysis strengthens your confidence in the model's real-world performance.

#### How it works:

Initiate the Harness class
```python
harness = Harness(
    task="question-answering",
    model={"model": "gpt-3.5-turbo-instruct", "hub": "openai"},
    data=[
        {"data_source": "NQ-open", "split": "test-tiny",},
        {"data_source": "MedQA", "split": "test-tiny"},
        {"data_source": "LogiQA", "split": "test-tiny"},
    ],
)
```
Configure the accuracy tests in Harness class
```python
harness.configure(
    {
        "tests": {
            "defaults": {"min_pass_rate": 0.65},
           
            "accuracy": {
                "llm_eval": {"min_score": 0.60},
                "min_rouge1_score": {"min_score": 0.60},
                "min_rouge2_score": {"min_score": 0.60},
                "min_rougeL_score": {"min_score": 0.60},
                "min_rougeLsum_score": {"min_score": 0.60},
            },
        }
    }
)
```
harness.generate() generates testcases, .run() executes them, and .report() compiles results.
```python
harness.generate().run().report()
```
![image](https://github.com/JohnSnowLabs/langtest/assets/23481244/0d48be2f-e5bc-4971-b0a1-2756a10d3f24)

### 🖥️ Streamlined Evaluation Workflows with Enhanced CLI Commands
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/benchmarks/Langtest_Cli_Eval_Command.ipynb)

LangTest's evaluation capabilities, focusing on report management and leaderboards. These enhancements empower you to:

- **Streamlined Reporting and Tracking:** Effortlessly save and load detailed evaluation reports directly from the command line using `langtest eval`, enabling efficient performance tracking and comparative analysis over time, with manual file review options in the `~/.langtest` or `./.langtest` folder.

- **Enhanced Leaderboards:** Gain valuable insights with the new langtest show-leaderboard command. This command displays existing leaderboards, providing a centralized view of ranked model performance across evaluations.

- **Average Model Ranking:** Leaderboard now include the average ranking for each evaluated model. This metric provides a comprehensive understanding of model performance across various datasets and tests.

### How it works:

First, create the `parameter.json` or `parameter.yaml` in the working directory

**JSON Format**
```json
{
    "task": "question-answering",
    "model": {
        "model": "google/flan-t5-base",
        "hub": "huggingface"
    },
    "data": [
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
    "config": {
        "model_parameters": {
            "max_tokens": 64,
            "device": 0,
            "task": "text2text-generation"
        },
        "tests": {
            "defaults": {
                "min_pass_rate": 0.70
            },
            "robustness": {
                "add_typo": {
                    "min_pass_rate": 0.70
                }
            }
        }
    }
}
```
**Yaml Format**
```yaml
task: question-answering
model:
  model: google/flan-t5-base
  hub: huggingface
data:
- data_source: MedMCQA
- data_source: PubMedQA
- data_source: MMLU
- data_source: MedQA
config:
  model_parameters:
    max_tokens: 64
    device: 0
    task: text2text-generation
  tests:
    defaults:
      min_pass_rate: 0.70
    robustness:
      add_typo:
        min_pass_rate: 0.7

```
And open the terminal or cmd in your system 
```bash
langtest eval --model <your model name or endpoint> \
              --hub <model hub like hugging face, lm-studio, web ...> \
              -c < your configuration file like parameter.json or parameter.yaml>
```
Finally, we can know the leaderboard and rank of the model.
![image](https://github.com/JohnSnowLabs/langtest/assets/23481244/a405d0c6-5ef1-4efb-924c-0ba8667ebe43)

----

To visualize the leaderboard anytime using the CLI command
```bash
langtest show-leaderboard
```
![image](https://github.com/JohnSnowLabs/langtest/assets/23481244/f357c173-e4b1-4dc8-86ad-98438046b89c)

## 📒 New Notebooks

{:.table2}
| Notebooks          | Colab Link |
|--------------------|-------------|
| Generic API-based Model Testing         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Generic_API-Based_Model_Testing_Demo.ipynb)|
| Multi-Dataset     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Multiple_dataset.ipynb) |
| Langtest Eval Cli Command     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/benchmarks/Langtest_Cli_Eval_Command.ipynb) |


## 🐛 Fixes

- Fixed multi-dataset support for accuracy task [#998]
- Fixed bugs in langtest package [#1003][#1004]


## ⚡ Enhancements
- Improved the error handling in Harness run method [#990]
- Websites Updates [#1001]
- Updated new version for dependencies   [#992]
- Improved the data augmentation for Question-Answering task [#991]

## What's Changed

* Feautre/integration with web api by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/986
* Refactor TestFactory class to handle exceptions in async tests by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/990
* data augmentation support for question-answering task by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/991
* Updated dependencies by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/992
* Fix/implement the multiple dataset support for accuracy tests by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/998
* Feature/add support for other file formats by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/993
* Bug Fix: Generated results are none by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1000
* Feature/implement load & save for benchmark reports by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/999
* Fix/bug fixes langtest 2 1 0 rc1 by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1003
* website updates by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1001
* Fix/bug fixes langtest 2 1 0 rc1 by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1004
* Release/2.0.1 by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1005


**Full Changelog**: https://github.com/JohnSnowLabs/langtest/compare/2.0.0...2.1.0

## ⚒️ Previous Versions
</div>
{%- include docs-langtest-pagination.html -%}
