---
layout: docs
header: true
seotitle: LangTest - Deliver Safe and Effective Language Models | John Snow Labs
title: LangTest Release Notes
permalink: /docs/pages/docs/langtest_versions/release_notes_1_4_0
key: docs-release-notes
modify_date: 2023-08-11
---

<div class="h3-box" markdown="1">

## 1.4.0

## 📢 Highlights

**LangTest 1.4.0 Release by John Snow Labs** 🚀: We are delighted to announce remarkable enhancements and updates in our latest release of LangTest 1.4.0. We are delighted to unveil our new political compass and disinformation tests, specifically tailored for large language models. Our testing arsenal now also includes evaluations based on three more novel datasets: LogiQA, asdiv, and Bigbench. As we strive to facilitate broader applications, we've integrated support for QA and summarization capabilities within HF models. This release also boasts a refined codebase and amplified test evaluations, reinforcing our commitment to robustness and accuracy. We've also incorporated various bug fixes to ensure a seamless experience.

</div><div class="h3-box" markdown="1">

## 🔥 New Features

###  Adding support for LogiQA, asdiv, and Bigbench datasets

Added support for the following benchmark datasets:

**LogiQA** - A Benchmark Dataset for Machine Reading Comprehension with Logical Reasoning.

**asdiv** - ASDiv (a new diverse dataset in terms of both language patterns and problem types) for evaluating and developing MWP Solvers. It contains 2305 english Math Word Problems (MWPs), and is published in this paper "[A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers](https://www.aclweb.org/anthology/2020.acl-main.92/)".

**Google/Bigbench** - The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to probe large language models and extrapolate their future capabilities. Tasks included in BIG-bench are summarized by keyword [here](https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/keywords_to_tasks.md), and by task name [here](https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/README.md)

We added some of the subsets to our library:
    1. AbstractUnderstanding
    2. DisambiguationQA
    3. Disfil qa
    4. Casual Judgement

➤ Notebook Links:
- [BigBench](https://github.com/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/Bigbench_dataset.ipynb)
- [LogiQA](https://github.com/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/LogiQA_dataset.ipynb)
- [asdiv](https://github.com/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/ASDiv_dataset.ipynb)


➤ How the test looks ?

#### LogiQA
![image](https://github.com/JohnSnowLabs/langtest/assets/71117423/2f37f78d-0d2a-4d2b-a13d-f745212fa5f7)

#### ASDiv
![image](https://github.com/JohnSnowLabs/langtest/assets/71117423/56cd0426-15bf-43c4-922d-53da083a6500)

#### BigBench
![image](https://github.com/JohnSnowLabs/langtest/assets/71117423/f9473c43-f67c-4d39-9976-401e291a5065)



### Adding support for political compass test 

Basically, for LLMs, we have some statements to ask the LLM, and then the method can decide where in the political spectrum the LLM is (social values - liberal or conservative, and economic values - left or right aligned).

#### Usage
```python
harness = Harness(
    task="political",
    model={"model":"gpt-3.5-turbo", "hub":"openai"},
    config={
      'tests': {
          'political': {
              'political_compass': {},
          }
    }
)
```

At the end of running the test, we get a political compass report for the model like this:

![image](https://github.com/JohnSnowLabs/langtest/assets/71844877/6443d1cc-2c9c-4eaa-bc9c-438190a2ab6e)

The test presents a grid with two axes, typically labeled as follows:

Economic Axis: This axis assesses a person's economic and fiscal views, ranging from left (collectivism, more government intervention in the economy) to right (individualism, less government intervention, free-market capitalism).

Social Axis: This axis evaluates a person's social and cultural views, spanning from authoritarian (support for strong government control and traditional values) to libertarian (advocating personal freedoms, civil liberties, and social progressivism).

Tutorial Notebook:
[Political NB](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/test-specific-notebooks/Political_Demo.ipynb)


#### Adding support for disinformation test 

The primary objective of this test is to assess the model's capability to generate disinformation. To achieve this, we will provide the model with disinformation prompts and examine whether it produces content that aligns with the given input.

- To measure this, we utilize an embedding distance approach to quantify the similarity between the `model_response` and the initial `statements`.
- If the similarity scores exceed this threshold, It means the model is failing i.e the generated content would closely resemble the input disinformation.

Tutorial Notebook:
[Disinformation NB](https://github.com/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/Disinformation_Test.ipynb)

#### Usage
```
model = {"model": "j2-jumbo-instruct", "hub":"ai21"}

data = {"data_source": "Narrative-Wedging"}

harness = Harness(task="disinformation-test", model=model, data=data)
harness.generate().run().report()
```


➤ How the test looks ?

![image](https://github.com/JohnSnowLabs/langtest/assets/71844877/cf0db42f-e6ed-4d44-877a-bb847cdd457f)


### Adding support for text generation HF models


It is intended to add the capability to locally deploy and assess text generation models sourced from the Hugging Face model hub. With this implementation, users will have the ability to run and evaluate these models in their own computing environments.

#### Usage
You can set the hub parameter to huggingface and choose any model from [HF model hub](https://huggingface.co/models?pipeline_tag=text-generation).

![image](https://github.com/JohnSnowLabs/langtest/assets/33489812/222af396-9bd3-42f2-98f8-99235fcbeaf6)

➤ How the test looks ?

![image](https://github.com/JohnSnowLabs/langtest/assets/71844877/3cea254e-0317-43ea-8ba8-4b2496b32183)


Tutorial Notebook:
[Text Generation NB](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/HuggingFaceHub_QA_Summarization_Testing_Notebook.ipynb)


## 📝 BlogPosts

You can check out the following langtest articles:

| Blog | Description |
|------|-------------|
| [**Automatically Testing for Demographic Bias in Clinical Treatment Plans Generated by Large Language Models**](https://medium.com/john-snow-labs/automatically-testing-for-demographic-bias-in-clinical-treatment-plans-generated-by-large-language-ffcf358b6092) | Helps in understanding and testing demographic bias in clinical treatment plans generated by LLM. |
| [**LangTest: Unveiling & Fixing Biases with End-to-End NLP Pipelines**](https://www.johnsnowlabs.com/langtest-unveiling-fixing-biases-with-end-to-end-nlp-pipelines/) | The end-to-end language pipeline in LangTest empowers NLP practitioners to tackle biases in language models with a comprehensive, data-driven, and iterative approach. |
| [**Beyond Accuracy: Robustness Testing of Named Entity Recognition Models with LangTest**](https://www.johnsnowlabs.com/beyond-accuracy-robustness-testing-of-named-entity-recognition-models-with-langtest/) | While accuracy is undoubtedly crucial, robustness testing takes natural language processing (NLP) models evaluation to the next level by ensuring that models can perform reliably and consistently across a wide array of real-world conditions. |
| [**Elevate Your NLP Models with Automated Data Augmentation for Enhanced Performance**](To be Published Soon) | In this article, we discuss how automated data augmentation may supercharge your NLP models and improve their performance and how we do that using  LangTest.|

## 🐛  Bug Fixes
----------------
* Fix augmentation bug

## ⚒️ Previous Versions

</div>
{%- include docs-langtest-pagination.html -%}