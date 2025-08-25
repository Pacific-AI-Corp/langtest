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

## 2.5.0
------------------
## **📢 Highlights**  
We are thrilled to announce the latest release, packed with exciting updates and enhancements to empower your AI model evaluation and development workflows!

- **🔗 Spark DataFrames and Delta Live Tables Support**  
We've expanded our capabilities with support for **Spark DataFrames** and **Delta Live Tables** from Databricks, allowing seamless integration and efficient data processing for your projects.

- **🧪 Performance Degradation Analysis in Robustness Testing**  
Introducing **Performance Degradation Analysis** in robustness tests! Gain insights into how your models handle edge cases and ensure consistent performance under challenging scenarios.

- **🖼 Enhanced Image Robustness Testing**  
We've added **new test types for Image Robustness** to evaluate your vision models rigorously. the models can test with diverse image perturbations and assess their ability to adapt.

- **🛠 Customizable Templates for LLMs**  
Personalize your workflows effortlessly with **customizable templates** for large language models (LLMs) from Hugging Face. Tailor prompts and configurations to meet your specific needs.

- **💬 Improved LLM and VQA Model Functionality**  
Enhancements to **chat and completion functionality** make interactions with LLMs and Vision Question Answering (VQA) models more robust and user-friendly.

- **✔ Improved Unit Tests and Type Annotations**  
We've bolstered **unit tests and type annotations** across the board, ensuring better code quality, reliability, and maintainability.

- **🌐 Website Updates**  
The website has been updated with new content highlighting Databricks integration, including support for Spark DataFrames and Delta Live Tables tutorials.


## 🔥 Key Enhancements


### 🔗 Spark DataFrames and Delta Live Tables Support  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/LangTest_Databricks_Integration.ipynb)

We've expanded our capabilities with support for Spark DataFrames and Delta Live Tables from Databricks, enabling seamless integration and efficient data processing for your projects.  

#### Key Features  
- **Seamless Integration**: Easily incorporate Spark DataFrames and Delta Live Tables into your workflows.  
- **Enhanced Efficiency**: Optimize data processing with Databricks' powerful tools.  

#### How it Works: 

```python
from pyspark.sql import DataFrame

 # Load the dataset into a Spark DataFrame
 df: DataFrame = spark.read.json("<FILE_PATH>")

df.printSchema()
```

**Tests Config:**

```python
prompt_template = (
    "You are an AI bot specializing in providing accurate and concise answers to questions. "
    "You will be presented with a medical question and multiple-choice answer options. "
    "Your task is to choose the correct answer.\n"
    "Question: {question}\n"
    "Options: {options}\n"
    "Answer: "
)

```
```python
from langtest.types import HarnessConfig

test_config: HarnessConfig = {
    "evaluation": {
        "metric": "llm_eval",
        "model": "gpt-4o", # for evaluation
        "hub": "openai",
    },
    "tests": {
        "defaults": {
            "min_pass_rate": 1.0,
            "user_prompt": prompt_template,
        },
        "robustness": {
            "add_typo": {"min_pass_rate": 0.8},
            "add_ocr_typo": {"min_pass_rate": 0.8},
            "add_speech_to_text_typo":{"min_pass_rate": 0.8},
            "add_slangs": {"min_pass_rate": 0.8},
            "uppercase": {"min_pass_rate": 0.8},
        },
    },
}
```

**Dataset Config:**
```python
input_data = {
     "data_source": df,
     "source": "spark",
     "spark_session": spark # make sure that spark session is started or not
 }
```
**Model Config:**
```python
model_config = {
     "model": {
         "endpoint": "databricks-meta-llama-3-1-70b-instruct",
     },
     "hub": "databricks",
     "type": "chat"
 }
```
Harness Setup:
```python
from langtest import Harness 

 harness = Harness(
     task="question-answering",
     model=model_config,
     data=input_data,
     config=test_config
 )
```
```python
harness.generate().run().report()
```
![image](https://github.com/user-attachments/assets/ef2f0eab-49bd-417a-a4ee-5e7efab7a4ea)


To Review and Store in DLT
```python
testcases= harness.testcases()
testcases
```

```python
testcases_dlt_df = spark.createDataFrame(testcases)

testcases_dlt_df.write.format("delta").save("<FILE_PATH>")
```

```python
generated_results = harness.generated_results()
generated_results
```

```python
# write into delta tables.
results_dlt_df = spark.createDataFrame(generated_results)

# Choose a file model based on the requirements
# to append results into the existing table or 
# overwrite the table.
results_dlt_df.write.format("delta").save("<FILE_PATH>")
```

###  🧪 Performance Degradation Analysis in Robustness Testing  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Degradation_Analysis_Test.ipynb)

Introducing Performance Degradation Analysis in robustness tests! Gain insights into how your models handle edge cases and ensure consistent performance under challenging scenarios.  

#### Key Features  
- **Edge Case Insights**: Understand model behavior in extreme conditions.  
- **Performance Consistency**: Ensure reliability across diverse inputs.  

#### How it Works: 

```python
from langtest.types import HarnessConfig
from langtest import Harness
```

```python
test_config = HarnessConfig({
    "tests": {
        "defaults": {
            "min_pass_rate": 0.6,
        },
        "robustness": {
            "uppercase": {
                "min_pass_rate": 0.7,
            },
            "lowercase": {
                "min_pass_rate": 0.7,
            },
            "add_slangs": {
                "min_pass_rate": 0.7,
            },
            "add_ocr_typo": {
                "min_pass_rate": 0.7,
            },
            "titlecase": {
                "min_pass_rate": 0.7,
            }
        },
        "accuracy": {
            "degradation_analysis": {
                "min_score": 0.7,
            }
        }
    }
})

# data config
data = {
    "data_source": "BoolQ",
    "split": "dev-tiny",
}
```
Setup Harness: 
```python
harness = Harness(
    task="question-answering", 
    model={
        "model": "llama3.1:latest", 
        "hub": "ollama",
        "type": "chat",
    },
    config=test_config,
    data=data
)

harness.generate().run()
```
Harness Report 
```python
harness.report()
```
![image](https://github.com/user-attachments/assets/88352c00-e94c-49d2-b8ab-d13cdaa1c716)

###  🖼 Enhanced Image Robustness Testing  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Visual_QA_II.ipynb)

We've added new test types for Image Robustness to evaluate your vision models rigorously. Could you challenge your models with diverse image perturbations and assess their ability to adapt?  

#### Key Features  
- **Diverse Perturbations**: Evaluate performance with new image robustness tests.  
- **Vision Model Assessment**: Test adaptability under varied visual conditions.  


Perturbation | Description
-- | --
`image_translate` | Shifts the image horizontally or vertically to evaluate model robustness against translations.
`image_shear` | Applies a shearing transformation to test how the model handles distortions in perspective.
`image_black_spots` | Introduces random black spots to simulate damaged or obscured image regions.
`image_layered_mask` | Adds layers of masking to obscure parts of the image, testing recognition under occlusion.
`image_text_overlay` | Places text on the image to evaluate the model's resilience to textual interference.
`image_watermark` | Adds a watermark to test how the model performs with watermarked images.
`image_random_text_overlay` | Randomly places text at varying positions and sizes, testing model robustness to overlays.
`image_random_line_overlay` | Draws random lines over the image to check the model's tolerance for line obstructions.
`image_random_polygon_overlay` | Adds random polygons to the image, simulating graphical interference or shapes.


#### How it Works:

```python
from langtest.types import HarnessConfig
from langtest import Harness
```

```python
test_config = HarnessConfig(
{
    "evaluation": {
        "metric": "llm_eval",
        "model": "gpt-4o-mini",
        "hub": "openai"

    },
    "tests": {
        "defaults": {
            "min_pass_rate": 0.5,
            "user_prompt": "{question}?\n {options}\n",
        },
        "robustness": {
            "image_random_line_overlay": {
                "min_pass_rate": 0.5,
            },
            "image_random_polygon_overlay": {
                "min_pass_rate": 0.5,
            },
            "image_random_text_overlay": {
                "min_pass_rate": 0.5,
                "parameters": {
                    "color": [123, 144, 123],
                    "opacity": 0.8
                }
            },
            "image_watermark": {
                "min_pass_rate": 0.5,
            },
        }
    }
}
)
```
Setup Harness: 
```python
from langtest import Harness

harness = Harness(
    task="visualqa",
    model={
        "model": "gpt-4o-mini",
        "hub": "openai"
    },
    data={"data_source": 'MMMU/MMMU',
          # "subset": "Clinical_Medicine",
          "subset": "Art",
          "split": "dev",
          "source": "huggingface"
    },
    config=test_config
)

harness.generate().run()
```

```python
from IPython.display import display, HTML

res_df = harness.generated_results()
html=res_df.sample(5).to_html(escape=False)

display(HTML(html))
```
![image](https://github.com/user-attachments/assets/242e2e7f-f0be-4a0e-a759-b3621fd0c19e)


report 
```python
harness.report()
```
![image](https://github.com/user-attachments/assets/3f1d9069-1d12-4d0d-b787-915d0fef2d74)


###  🛠 Customizable Templates for LLMs  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Custom_Chat_Template_Config.ipynb)

Personalize your workflows effortlessly with customizable templates for large language models (LLMs) from Hugging Face. Tailor prompts and configurations to meet your specific needs.  

#### Key Features  
- **Workflow Personalization**: Customize LLM templates to suit your tasks.  
- **Enhanced Usability**: Simplify configurations with pre-built templates.  

#### How it Works:

```python
from langtest.types import HarnessConfig
from langtest import Harness

import os 

os.environ["HUGGINGFACE_API_KEY"] = "<YOUR HUGGINGFACE API>"
os.environ["OPENAI_API_KEY"] = "<your-openai-api-key>"
```

```python
# only jinja template supported
meta_template = """
{% raw %}
{{- bos_token }}\n

{%- if messages[0]['role'] == 'system' %} 
    {%- set system_message = messages[0]['content']|trim %} 
    {%- set messages = messages[1:] %} 
{%- else %} 
    {%- set system_message = "You are a helpful assistant. Provide a short answer based on the given context and question in plain text." %} 
{%- endif %}

{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\\n" }}
{{- system_message }}
{{- "<|eot_id|>" }}

{%- for message in messages %} 
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n' + message['content'] | trim + '<|eot_id|>' }} 
{%- endfor %} 
{{- '<|start_header_id|>assistant<|end_header_id|>\\n' }}
{% endraw %}
"""

# few shot prompt config
prompt_config =  {
    "NQ-open": {
        "prompt_type": "chat",
        "instructions": "Write a short answer based on the given context and question in plain text.",
        "user_prompt": "You are a helpful assistant. Provide a short answer based on the given context and question.\n {question}",
        "examples": [{
            "user": {
                "question": "What is the capital of France?",
                "context": "France is a country in Europe."
            },
            "ai": {
                "answer": "Paris"
            }
      }]
    }
}
```

Test Config:
```python
from langtest.types import HarnessConfig


test_config: HarnessConfig = {
    "evaluation": {
        "metric": "llm_eval",
        "model": "gpt-4o",
        "hub": "openai",
    },
    "prompt_config": prompt_config,
    "model_parameters": {
        "chat_template": meta_template,
        "max_tokens": 50,
        "task": "text-generation",
        "device": 0, # Use GPU 0
    },
    "tests": {
        "defaults": {
            "min_pass_rate": 0.6,
        },
        "robustness": {
            "uppercase": {
                "min_pass_rate": 0.7,
            },
            "add_slangs": {
                "min_pass_rate": 0.7,
            },
            "add_ocr_typo": {
                "min_pass_rate": 0.7,
            },
        },
    }
}
```

Harness Setup:

```python
harness = Harness(
    task="question-answering",
    model={
        "model": "meta-llama/Llama-3.2-3B-Instruct", 
        "hub": "huggingface",
        "type": "chat",
        },
    data={"data_source": "NQ-open",
          "split": "test-tiny"},
    config=test_config,
)
```

```python
harness.generate().run().report()
```
![image](https://github.com/user-attachments/assets/5d572aed-06f6-47db-af0e-e1f78a1efa7a)

```python
harness.generated_results()
```
![image](https://github.com/user-attachments/assets/73986530-31d7-4ca6-a94e-3ee7cbb46040)


###  💬 Improved LLM and VQA Model Functionality  
We have enhanced the chat and completion functionality, making interactions with LLMs and Vision Question Answering (VQA) models more robust and intuitive. These improvements enable smoother conversational experiences with LLMs and deliver better performance for VQA tasks. The updates focus on creating a more user-friendly and efficient interaction framework, ensuring high-quality results for diverse applications.

###  ✔ Improved Unit Tests and Type Annotations  
We have strengthened unit tests and implemented clearer type annotations throughout the codebase to ensure improved quality, reliability, and maintainability. These updates enhance testing coverage and robustness, making the code more resilient and dependable. Additionally, the use of precise type annotations supports better readability and easier maintenance, contributing to a more efficient development process.


### 🌐 Website Updates  
The website has been updated to feature new content emphasizing Databricks integration. It now includes tutorials that showcase working with Spark DataFrames and Delta Live Tables, providing users with practical insights and step-by-step guidance. These additions aim to enhance the learning experience by offering comprehensive resources tailored to Databricks users. The updated content highlights key features and capabilities, ensuring a more engaging and informative experience.


## 📒 New Notebooks

| Notebooks          | Colab Link |
|--------------------|-------------|
| **LangTest-Databricks Integration**     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/LangTest_Databricks_Integration.ipynb) |
| **Degradation Analysis Test**       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Degradation_Analysis_Test.ipynb)|
| **Custom Chat Template Config**    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Custom_Chat_Template_Config.ipynb) |


## What's Changed
* Websites Changes in v2.1.0 by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1006
* updates web pages by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1032
* adding workflow for github pages by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1051
* websites updates  with fixes by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1079
* Website Updates for 2.4.0 by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1126
* Fix/basic setup within datrabricks using azure openai by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1128
* Feature/implement accuracy drop tests on robustness and bias by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1129
* Feature/add support for chat and instruct model types by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1131
* updated: model_kwargs handling for evaluation model by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1133
* updated: acclerate and spacy packages by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1135
* Feature/enhance harness report to include detailed score counts and grouped results by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1132
* Feature/random masking on images tests by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1138
* Unit testing/add new unit tests to enhance test coverage and reliability by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1140
* added new overlay classes for enhanced image robustness by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1141
* Annotations/improve the type annotation for config by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1143
* fix: enhance model loading logic and update dependencies for by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1145
* fix: improve model_report function to handle numeric values and initi… by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1146
* Feature/support for loading datasets from dlt within databricks by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1148
* feat: update dependency version constraints in pyproject.toml  by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1149
* feat: enhance DegradationAnalysis to support question-answering task by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1153
* Chore/final website updates for 2.5.0 by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1150
* Chore/final website updates by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1155
* Release/2.5.0 by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1144


**Full Changelog**: https://github.com/JohnSnowLabs/langtest/compare/2.4.0...2.5.0

</div>
{%- include docs-langtest-pagination.html -%}
