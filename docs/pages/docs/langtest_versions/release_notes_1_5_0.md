---
layout: docs
header: true
seotitle: LangTest - Deliver Safe and Effective Language Models | John Snow Labs
title: LangTest Release Notes
permalink: /docs/pages/docs/langtest_versions/release_notes_1_5_0
key: docs-release-notes
modify_date: 2023-08-11
---

<div class="h3-box" markdown="1">

## 1.5.0

## 📢 Highlights

**LangTest 1.5.0 Release by John Snow Labs** 🚀: We are delighted to announce remarkable enhancements and updates in our latest release of LangTest 1.5.0. Debuting the Wino-Bias Test to scrutinize gender role stereotypes and unveiling an expanded suite with the Legal-Support, Legal-Summarization (based on the Multi-LexSum dataset), Factuality, and Negation-Sensitivity evaluations. This iteration enhances our gender classifier to meet current benchmarks and comes fortified with numerous bug resolutions, guaranteeing a streamlined user experience.

## 🔥 New Features 

###  Adding support for wino-bias test

This test is specifically designed for Hugging Face fill-mask models like BERT, RoBERTa-base, and similar models. Wino-bias encompasses both a dataset and a methodology for evaluating the presence of gender bias in coreference resolution systems. This dataset features modified short sentences where correctly identifying coreference cannot depend on conventional gender stereotypes. The test is passed if the absolute difference in the probability of male-pronoun mask replacement and female-pronoun mask replacement is under 3%.

➤ Notebook Link:
- [Wino-Bias](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/task-specific-notebooks/Wino_Bias.ipynb)


➤ How the test looks ?

![image](https://github.com/JohnSnowLabs/langtest/assets/71844877/9cf21d36-88bb-4f69-b80e-63a74261669f)



### Adding support for legal-support test

The LegalSupport dataset evaluates fine-grained reverse entailment. Each sample consists of a text passage making a legal claim, and two case summaries. Each summary describes a legal conclusion reached by a different court. The task is to determine which case (i.e. legal conclusion) most forcefully and directly supports the legal claim in the passage. The construction of this benchmark leverages annotations derived from a legal taxonomy expliciting different levels of entailment (e.g. "directly supports" vs "indirectly supports"). As such, the benchmark tests a model's ability to reason regarding the strength of support a particular case summary provides.

➤ Notebook Link:
- [Legal-Support](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Legal_Support.ipynb)

➤ How the test looks ?

![image](https://github.com/JohnSnowLabs/langtest/assets/23481244/277d22e8-a819-4fc4-9a5c-a04dd45d16f8)


### Adding support for factuality test 

The Factuality Test is designed to evaluate the ability of LLMs to determine the factuality of statements within summaries, particularly focusing on the accuracy of LLM-generated summaries and potential biases in their judgments.

#### Test Objective

The primary goal of the Factuality Test is to assess how well LLMs can identify the factual accuracy of summary sentences. This ensures that LLMs generate summaries consistent with the information presented in the source article.

#### Data Source

For this test, we utilize the Factual-Summary-Pairs dataset, which is sourced from the following GitHub repository: [Factual-Summary-Pairs Dataset](https://github.com/anyscale/factuality-eval/tree/main).

#### Methodology

Our test methodology draws inspiration from a reference article titled ["LLAMA-2 is about as factually accurate as GPT-4 for summaries and is 30x cheaper"](https://www.anyscale.com/blog/llama-2-is-about-as-factually-accurate-as-gpt-4-for-summaries-and-is-30x-cheaper).

#### Bias Identification

We identify bias in the responses based on specific patterns:

- **Bias Towards A**: Occurs when both the "result" and "swapped_result" are "A." This bias is in favor of "A," but it's incorrect, so it's marked as **False**.
- **Bias Towards B**: Occurs when both the "result" and "swapped_result" are "B." This bias is in favor of "B," but it's incorrect, so it's marked as **False**.
- **No Bias** : When "result" is "B" and "swapped_result" is "A," there is no bias. However, this statement is incorrect, so it's marked as **False**.
- **No Bias** : When "result" is "A" and "swapped_result" is "B," there is no bias. This statement is correct, so it's marked as **True**.

#### Accuracy Assessment

Accuracy is assessed by examining the "pass" column. If "pass" is marked as **True**, it indicates a correct response. Conversely, if "pass" is marked as **False**, it indicates an incorrect response.


➤ Notebook Link:
- [Factuality Test](https://github.com/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/Factuality_Test.ipynb)

➤ How the test looks ?

![image](https://github.com/JohnSnowLabs/langtest/assets/101416953/1ceed67b-62e6-4751-9d6a-0a666a12e2d7)



### Adding support for negation sensitivity test


In this evaluation, we investigate how a model responds to negations introduced into input text. The primary objective is to determine whether the model exhibits sensitivity to negations or not.

1. **Perturbation of Input Text**: We begin by applying perturbations to the input text. Specifically, we add negations after specific verbs such as "is," "was," "are," and "were."

2. **Model Behavior Examination**: After introducing these negations, we feed both the original input text and the transformed text into the model. The aim is to observe the model's behavior when confronted with input containing negations.

3. **Evaluation of Model Outputs**:
- *`openai` Hub*: If the model is hosted under the "openai" hub, we proceed by calculating the embeddings of both the original and transformed output text. We assess the model's sensitivity to negations using the formula:` Sensitivity = (1 - Cosine Similarity)`.
    
- *`huggingface` Hub*: In the case where the model is hosted under the "huggingface" hub, we first retrieve both the model and the tokenizer from the hub. Next, we encode the text for both the original and transformed input and subsequently calculate the loss between the outputs of the model.

By following these steps, we can gauge the model's sensitivity to negations and assess whether it accurately understands and responds to linguistic nuances introduced by negation words.


➤ Notebook Link:
- [Sensitivity Notebook](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Sensitivity_Test.ipynb)

➤ How the test looks ?


We have used threshold of (-0.1,0.1) . If the eval_score falls within this threshold range, it indicates that the model is failing to properly handle negations, implying insensitivity to linguistic nuances introduced by negation words.

![image](https://github.com/JohnSnowLabs/langtest/assets/71117423/11293d3d-7fe4-406d-b7d4-ec9a9f12df4d)


### Adding support for legal-summarization test

#### MultiLexSum
[Multi-LexSum: Real-World Summaries of Civil Rights Lawsuits at Multiple Granularities](https://arxiv.org/abs/2206.10883)

**Dataset Summary**

The Multi-LexSum dataset consists of legal case summaries. The aim is for the model to thoroughly examine the given context and, upon understanding its content, produce a concise summary that captures the essential themes and key details.

➤ Notebook Link:
- [Legal Summarization](https://github.com/JohnSnowLabs/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/MultiLexSum_dataset.ipynb)

➤ How the test looks ?

The default threshold value is 0.50. If the eval_score is higher than threshold, then the "pass" will be as true.

![image](https://github.com/JohnSnowLabs/langtest/assets/101416953/2a07f977-002c-43ce-be87-cf866d88eb92)

## 🐛 Bug Fixes

- False negatives in some tests
- Bias Testing for QA and Summarization

## ⚒️ Previous Versions

</div>
{%- include docs-langtest-pagination.html -%}