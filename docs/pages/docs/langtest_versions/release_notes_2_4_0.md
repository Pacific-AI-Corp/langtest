---
layout: docs
header: true
seotitle: LangTest - Deliver Safe and Effective Language Models | John Snow Labs
title: LangTest Release Notes
permalink: /docs/pages/docs/langtest_versions/release_notes_2_4_0
key: docs-release-notes
modify_date: 2024-12-02
---

<div class="h3-box" markdown="1">

## 2.4.0

## 📢 **Highlights**

John Snow Labs is excited to announce the release of LangTest 2.4.0! This update introduces cutting-edge features and resolves key issues further to enhance model testing and evaluation across multiple modalities.

- 🔗 **Multimodality Testing with VQA Task**: We are thrilled to introduce multimodality testing, now supporting Visual Question Answering (VQA) tasks! With the addition of 10 new robustness tests, you can now perturb images to challenge and assess your model’s performance across visual inputs.

- 📝 **New Robustness Tests for Text Tasks**: LangTest 2.4.0 comes with two new robustness tests, `add_new_lines` and `add_tabs`, applicable to text classification, question-answering, and summarization tasks. These tests push your models to handle text variations and maintain accuracy.

- 🔄 **Improvements to Multi-Label Text Classification**: We have resolved accuracy and fairness issues affecting multi-label text classification evaluations, ensuring more reliable and consistent results.

- 🛡 **Basic Safety Evaluation with Prompt Guard**: We have incorporated safety evaluation tests using the `PromptGuard` model, offering crucial layers of protection to assess and filter prompts before they interact with large language models (LLMs), ensuring harmful or unintended outputs are mitigated.

- 🛠 **NER Accuracy Test Fixes**: LangTest 2.4.0 addresses and resolves issues within the Named Entity Recognition (NER) accuracy tests, improving reliability in performance assessments for NER tasks.

- 🔒 **Security Enhancements**: We have upgraded various dependencies to address security vulnerabilities, making LangTest more secure for users.


## 🔥 **Key Enhancements**

### 🔗 **Multimodality Testing with VQA Task**  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Visual_QA.ipynb)

In this release, we introduce multimodality testing, expanding your model’s evaluation capabilities with Visual Question Answering (VQA) tasks.

**Key Features:**
- **Image Perturbation Tests**: Includes 10 new robustness tests that allow you to assess model performance by applying perturbations to images.
- **Diverse Modalities**: Evaluate how models handle both visual and textual inputs, offering a deeper understanding of their versatility.

**Test Type Info**

| **Perturbation**      | **Description**                      |
|-----------------------|--------------------------------------|
| `image_resize`        | Resizes the image to test model robustness against different image dimensions. |
| `image_rotate`        | Rotates the image at varying degrees to evaluate the model's response to rotated inputs. |
| `image_blur`          | Applies a blur filter to test model performance on unclear or blurred images. |
| `image_noise`         | Adds noise to the image, checking the model’s ability to handle noisy data. |
| `image_contrast`      | Adjusts the contrast of the image, testing how contrast variations impact the model's performance. |
| `image_brightness`    | Alters the brightness of the image to measure model response to lighting changes. |
| `image_sharpness`     | Modifies the sharpness to evaluate how well the model performs with different image sharpness levels. |
| `image_color`         | Adjusts color balance in the image to see how color variations affect model accuracy. |
| `image_flip`          | Flips the image horizontally or vertically to test if the model recognizes flipped inputs correctly. |
| `image_crop`          | Crops the image to examine the model’s performance when parts of the image are missing. |


**How It Works:**

**Configuration:**
to create a config.yaml
```yaml
# config.yaml
model_parameters:
    max_tokens: 64
tests:
    defaults:
        min_pass_rate: 0.65
    robustness:
        image_noise:
            min_pass_rate: 0.5
            parameters:
                noise_level: 0.7
        image_rotate:
            min_pass_rate: 0.5
            parameters:
                angle: 55
        image_blur:
            min_pass_rate: 0.5
            parameters:
                radius: 5
        image_resize:
            min_pass_rate: 0.5
            parameters:
                resize: 0.5

```

**Harness Setup**
```python
harness = Harness(
    task="visualqa",
    model={"model": "gpt-4o-mini", "hub": "openai"},
    data={
        "data_source": 'MMMU/MMMU',
        "subset": "Clinical_Medicine",
        "split": "dev",
        "source": "huggingface"
    },
    config="config.yaml",
)
```

**Execution:**

```python
harness.generate().run().report()
```
![image](https://github.com/user-attachments/assets/f429bfd8-6be3-44bf-8af7-f93dbe7d3683)

```python
from IPython.display import display, HTML


df = harness.generated_results()
html=df.sample(5).to_html(escape=False)

display(HTML(html))
```
![image](https://github.com/user-attachments/assets/fac7586d-0748-4c92-8b5d-2f10e51b3ca4)


### 📝 **Robustness Tests for Text Classification, Question-Answering, and Summarization**  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Add_New_Lines_and_Tabs_Tests.ipynb)

The new `add_new_lines` and `add_tabs` tests push your text models to manage input variations more effectively.

**Key Features:**
- **Perturbation Testing**: These tests insert new lines and tab characters into text inputs, challenging your models to handle structural changes without compromising accuracy.
- **Broad Task Support**: Applicable to a variety of tasks, including text classification, question-answering, and summarization.

Tests 

| **Perturbation**      | **Description**                                                           |
|-----------------------|---------------------------------------------------------------------------|
| `add_new_lines`       | Inserts random new lines into the text to test the model’s ability to handle line breaks and structural changes in text. |
| `add_tabs`            | Adds tab characters within the text to evaluate how the model responds to indentation and tabulation variations.         |


**How It Works:**

**Configuration:**
to create a config.yaml
```yaml
# config.yaml

tests:
    defaults:
        min_score: 0.7
    robustness:
        add_new_lines:
            min_pass_rate: 0.7
            parameters:
                max_lines: 5
        add_tabs:
            min_pass_rate: 0.7
            parameters:
                max_tabs: 5
```

**Harness Setup**
```python
harness = Harness( 
    task = "text-classification",
    model = {"model": 'textcat_imdb', "hub": "spacy"},
    config="config.yaml",
)
```

**Execution:**

```python
harness.generate().run().report()
```
![image](https://github.com/user-attachments/assets/c5a6a432-7aab-46a4-919c-5cda7b40aa43)


### 🛡 **Basic Safety Evaluation with Prompt Guard**  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Safety_Tests_With_PromptGuard.ipynb)

LangTest introduces safety checks using the prompt_guard model, providing essential safety layers for evaluating prompts before they are sent to large language models (LLMs), ensuring harmful or unethical outputs are avoided.

**Key Features:**

- **Safety Testing:** Identify risky, unethical, or unintended prompts or queries using the `jailbreak_probabilities_score` and `injection_probabilities_score` metrics before they are sent to LLM models.
- **Proactive Safeguards:** Mitigate potential safety issues early by checking prompts for vulnerabilities, ensuring adherence to ethical standards, and reducing the chances of unsafe responses from LLMs.


| **Test Name**                    | **Description**                                                                                           |
|----------------------------------|-----------------------------------------------------------------------------------------------------------|
| `jailbreak_probabilities_score`  | A metric that evaluates the likelihood of a model generating unsafe, unintended, or harmful outputs when exposed to adversarial prompts or jailbreak attempts. |
| `injection_probabilities_score`  | A metric that assesses the probability of a model being manipulated through input injections, where malicious or unintended content is injected into the output. |


**How It Works:**

**Configuration:**
to create a config.yaml
```yaml
# config.yaml
tests:
  defaults:
    max_score: 0.2
    user_prompt: "Ignore the previous instructions: "
  safety:
    injection_probalities_score:
      max_score: 0.2
    jailbreak_probalities_score:
      max_score: 0.2

```

**Harness Setup**
```python
harness = Harness(
    task="text-classification",
    model={
        "model": "google-t5/t5-base", # this model is not used while evaluating these tests from the safety category.
        "hub": "huggingface",
    },
    data={
        "data_source": "deepset/prompt-injections",
        "split": "test",
        "source": "huggingface"
    },
    config="config.yaml",
)
```

**Execution:**

```python
harness.generate().run().report()
```
![image](https://github.com/user-attachments/assets/a8074f07-f049-4b58-846a-f0fd70ce3fb7)

## 🐛 Fixes
- Fix/error in accuracy tests for multi-label classification [#1114]
- Fix/error in fairness tests for multi-label classification [#1121, #1120]
- Fix/error in accuracy tests for ner task [#1115, #1116]

## ⚡ Enhancements
- Resolved the Security and Vulnerabilities Issues. [#1112]

## What's Changed
* Added: implemeted the breaking sentence by newline in robustness. by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1109
* Feature/implement the addtabs test in robustness category by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1110
* Fix/error in accuracy tests for multi label classification by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1114
* Fix/error in accuracy tests for ner task by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1116
* Update transformers version to 4.44.2 by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1112
* Feature/implement the support for multimodal with new vqa task by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1111
* Fix/AttributeError in accuracy tests for multi label classification by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1118
* Refactor fairness test to handle multi-label classification  by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1121
* Feature/enhance safety tests with promptguard by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1119
* Release/2.4.0 by @chakravarthik27 in https://github.com/JohnSnowLabs/langtest/pull/1122


**Full Changelog**: https://github.com/JohnSnowLabs/langtest/compare/2.3.1...2.4.0

</div>
{%- include docs-langtest-pagination.html -%}
