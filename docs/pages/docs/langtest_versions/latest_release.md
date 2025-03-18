---
layout: docs
header: true
seotitle: LangTest - Deliver Safe and Effective Language Models | John Snow Labs
title: LangTest Release Notes
permalink: /docs/pages/docs/langtest_versions/latest_release
key: docs-release-notes
modify_date: 2024-12-02
---

<div class="h3-box" markdown="1">

## 2.6.0
------------------
## 📢 Highlights

We are excited to introduce the latest langtest release, bringing you a suite of improvements designed to streamline model evaluation and enhance overall performance:

- **🛠 De-biasing Data Augmentation:**  
  We’ve integrated de-biasing techniques into our data augmentation process, ensuring more equitable and representative model assessments.

- **🔄 Evaluation with  Structured Outputs:**  
  LangTest now supports structured output APIs for both OpenAI and Ollama, offering greater flexibility and precision when processing model responses.

- **🏥 Confidence Testing with Med Halt Tests:**  
  Introducing med halt tests for confidence evaluation, enabling more robust insights into your LLMs’ reliability under diverse conditions.

- **📖 Expanded Task Support for JSL LLM Models:**  
  QA and Summarization tasks are now fully supported for JSL LLM models, enhancing their capabilities for real-world applications.

- **🔒Security Enhancements:**  
  Critical vulnerabilities and security issues have been addressed, reinforcing the LangTest overall stability and safety.

- **🐛 Resolved Bugs:**  
  We’ve fixed issues with templatic augmentation to ensure consistent, accurate, and reliable outputs across your workflows.


## 🔥 Key Enhancements  

### 🛠 De-biasing Data Augmentation  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/misc/Dataset_Debiasing.ipynb)

We’ve integrated de-biasing techniques into our data augmentation process, ensuring more equitable and representative model assessments.  

**Key Features:**  
- Eliminates biases in training data to improve model fairness.  
- Enhances diversity in augmented datasets for better generalization.  

**How it works:**  
To load the dataset
```python
from datasets import load_dataset

dataset = load_dataset("RealTimeData/bbc_news_alltime", "2024-12", split="train")

# sample dataset with 500 rows
df = dataset.to_pandas()
sample = df.sample(500)

# to avoid the errors at context overflow
sample = sample[sample['content'].apply(lambda x: len(x) < 1000)
```

```python
# let's set up the debiasing 
from langtest.augmentation.debias import DebiasTextProcessing 

processing = DebiasTextProcessing(
    model="gpt-4o-mini",
    hub="openai",
    model_kwargs={
        "temperature": 0,
    }
)
```

```python
import pandas as pd

processing.initialize(
    input_dataset = sample,
    output_dataset = pd.DataFrame({}),
    text_column="content",
    
)

output, reason = processing.apply_bias_correction(bias_tolerance_level=2)

output.head()
```
![image](https://github.com/user-attachments/assets/5e56275c-14c6-42a8-9c1a-3e50f57f08f6)


### 🔄Evaluation with  Structured Outputs
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/misc/Evaluation_with_Structured_Outputs.ipynb)
 
Now supporting structured output APIs for OpenAI, Ollama, and Azure-OpenAI, offering greater flexibility and precision when processing model responses.  

**Key Features:**  
- Supports structured LLM outputs for better parsing and analysis.  
- Integrates effortlessly with OpenAI, Ollama, and Azure-OpenAI.

**How it works:**  

Pydantic Model Setup:

```python
from pydantic import BaseModel
from langtest import Harness

class Answer(BaseModel):
    
    class Rationale(BaseModel):
        """Explanation for an answer. why the answer is correct or incorrect with a valid reasons, a score, and a summary."""
        reason: str
        score: float
        summary: str

    answer: bool
    rationale: Rationale

    def __eq__(self, other: 'Answer') -> bool:
        return self.answer == other.answer

```
**Harness Setup:**
```python
harness = Harness(
    task='question-answering',
    model={
        'model': 'llama3.1',
        'hub': 'ollama',
        'type': 'chat',
        'output_schema': Answer,
    },
    data={
        "data_source": "BoolQ",
        "split": "test-tiny",
    },
    config={
        "tests": {
            "defaults": {
                "min_pass_rate": 0.5,
            },
            "robustness": {
                "uppercase": {
                    "min_pass_rate": 0.8,
                },
                "add_ocr_typo": {
                    "min_pass_rate": 0.8,
                },
                "add_tabs": {
                    "min_pass_rate": 0.8,
                }
            }
        }
    }
)

harness.generate().run().report()

```
![image](https://github.com/user-attachments/assets/9e4d7971-bae5-4a27-ab7a-5b0d3192bace)


### 🏥 Confidence Testing with Med Halt Tests  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/Med_Halt_Tests.ipynb)

Gain deeper insights into your LLMs’ robustness and reliability under diverse conditions with our upgraded Med Halt tests. This release focuses on refining confidence assessments in LLMs.

**Key Features:**  
- Identifies and prevents overconfident, incorrect responses in critical scenarios.
- To enhance confidence evaluation with these tests.

Test Name | Description
-- | --
**FCT** <br>(False Confidence Test) | Detects when an AI model is overly confident in incorrect answers by swapping answer options and including a "None of the Above" option.
**FQT** <br> (Fake Questions Test) | Evaluates the model's ability to handle questions presented out of their original context by exchanging contextual information.
**NOTA** <br> Test | Assesses whether the model can recognize insufficient information by replacing the correct answer with a "None of the Above" option.

**How it works:**  
```python
from langtest import Harness 


harness = Harness(
    task="question-answering",
    model={
        "model": "phi4-mini",
        "hub": "ollama",
        "type": "chat"
        # "model": "gpt-4o-mini",
        # "hub": "openai",
    },
    data={
        "data_source": "MMLU",
        "split": "clinical",
    },
    config={
        "model_parameters": {
            "user_prompt": (
                    "You are a knowledgeable AI Assistant. Please provide the best possible choice (A or B or C or D) from the options"
                    "to the following MCQ question with the given options. Note: only provide the choice and don't given any explanations\n"
                    "Question:\n{question}\n"
                    "Options:\n{options}\n"
                    "Correct Choice (A or B or C or D): "
                    
            )
        },
        "tests": {
            
            "defaults": {
                "min_pass_rate": 0.75,

            },
            "clinical": {
                "nota": {"min_pass_rate": 0.75},
            }
        },
        "evaluation": {
            "metric": "llm_eval",
            "model": "gpt-4o-mini",
            "hub": "openai",
        }
    }
)
```

Generate and Execute the test cases:
```python
harness.generate().run()
```

Report
```python
harness.generated_results()
```
![image](https://github.com/user-attachments/assets/a077d64d-0dd0-46ad-b97c-ca1f60d2337c)

```python
harness.report()
```
![image](https://github.com/user-attachments/assets/f3042fe7-2cc9-41b2-b57d-435d6cb22b56)


### 📖 QA and Summarization Support for JSL LLM Models  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/JSL_Medical_LLM.ipynb)

JSL LLM models now support both Question Answering (QA) and Summarization tasks, which makes testing more practical in real-world scenarios

**Key Features:**  
- Tests the model's ability to deliver clear and accurate answers.
- Evaluates the model's skill in creating concise summaries from longer texts

**How it works:**  

Pipeline Setup:

```python
document_assembler = MultiDocumentAssembler()\
    .setInputCols("question", "context")\
    .setOutputCols("document_question", "document_context")

med_qa = MedicalQuestionAnswering().pretrained("clinical_notes_qa_base_onnx", "en", "clinical/models")\
    .setInputCols(["document_question", "document_context"])\
    .setCustomPrompt(("You are an AI bot specializing in providing accurate and concise answers to questions"
                      ". You will be presented with a medical question and multiple-choice answer options."
                      " Your task is to choose the correct answer.\nQuestion: {question}\nOptions: {options}\n Answer:"))\
    .setOutputCol("answer")

pipeline = Pipeline(stages=[document_assembler, med_qa])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)
```

```python
import os 
# for evaluation
os.environ["OPENAI_API_KEY"] = "<API KEY>"
```
Harness Setup:
```python
from langtest import Harness 

harness = Harness(
    task="question-answering",
    model={
        "model": model,
        "hub": "johnsnowlabs",
    },
    data={
        "data_source": "PubMedQA",
        "subset": "pqaa",
        "split": "test",
    },
    config={  
        "tests": {
            "defaults": {
                "min_pass_rate": 0.5,
            },
            "robustness": {
                "uppercase": {
                    "min_pass_rate": 0.5,
                },
                "lowercase": {
                    "min_pass_rate": 0.5,
                },
                "add_ocr_typo": {
                    "min_pass_rate": 0.5,
                },
                "add_slangs": {
                    "min_pass_rate": 0.5,
                }
            }
        },
        "evaluation": {
            "metric": "llm_eval",
            "model": "gpt-4o-mini",
            "hub": "openai"
        }
    }
)
```
generate and run testcases
```python
harness.generate().run().report()
```
Results
![image](https://github.com/user-attachments/assets/26c2c110-ccaa-47b4-942e-c72efeb2bd15)
Report
![image](https://github.com/user-attachments/assets/5c59742f-1475-48f1-b198-303abfcfb043)

### 🔒 Security Enhancements  
Critical vulnerabilities and security issues have been resolved, reinforcing the overall stability and safety of our platform. In this update, we upgraded dependencies to fix vulnerabilities, ensuring a more secure and reliable environment for our users.

## 📒 New Notebooks

| Notebooks          | Colab Link |
|--------------------|-------------|
| **De-biasing Data Augmentation**     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/misc/Dataset_Debiasing.ipynb) |
| **🔄Evaluation with  Structured Outputs**       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/misc/Evaluation_with_Structured_Outputs.ipynb) |
| **Confidence Testing with Med Halt Tests**    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/Med_Halt_Tests.ipynb) |
| **JSL Medical LLM Models**    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/JSL_Medical_LLM.ipynb) |


## 🐛 Fixes
* fix: better handling of extra model params in Harness by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1183
* fixes: resolving the bugs 2_6_0rc versions by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1182
* Fix vulnerabilities and security issues by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1160
* fix(bug): update model handling in OpenAI and AzureOpenAI configurations by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1178


## ⚡ Enhancements
* vulnerabilities and security issues by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1161
* chore: update certifi, idna, zipp versions and add extras in poetry.lock by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1162
* updated the openai dependencies  by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1172
* feat: add support for generating templates using Ollama provider by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1180


## What's Changed
* website updates for public view by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1158
* Fix vulnerabilities and security issues by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1160
* vulnerabilities and security issues by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1161
* chore: update certifi, idna, zipp versions and add extras in poetry.lock by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1162
* Update the Medical_Dataset NB by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1169
* Feature/data augmentation for de biasing by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1164
* updated the openai dependencies  by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1172
* feat: enhance model handling with additional info and output schema s… by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1168
* feat: add support for question answering model in JSL model handler by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1174
* fix(bug): update model handling in OpenAI and AzureOpenAI configurations by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1178
* Feature/add integration to deepseek by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1176
* Feature/implement med halt tests for robust model evaluation by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1170
* feat: add support for generating templates using Ollama provider by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1180
* fixes: resolving the bugs 2_6_0rc versions by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1182
* fix: better handling of extra model params in Harness by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1183
* chore: update version to 2.6.0 by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1185
* Release/2.6.0 by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1184


**Full Changelog**: https://github.com/JohnSnowLabs/langtest/compare/2.5.0...2.6.0
</div>
{%- include docs-langtest-pagination.html -%}
