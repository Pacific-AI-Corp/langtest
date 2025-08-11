---
layout: docs
header: true
seotitle: Tutorials | LangTest | John Snow Labs
title: Test Specific Notebooks
key: docs-test_specific_notebooks
permalink: /docs/pages/tutorials/test_specific_notebooks/
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
show_edit_on_github: true
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">
The following table gives an overview of the different tutorial notebooks. We have test specific (Accuracy, Fairness, Robustness, Representation, Bias etc.) notebooks listed below.

</div><div class="h3-box" markdown="1">

{:.table2}
| Tutorial Description                | Hub                           | Task                              | Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
| [**Accuracy test**](accuracy) :  In this notebook we are evaluating `ner.dl` model on accuracy tests.                      | John Snow Labs                    | NER                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/test-specific-notebooks/Accuracy_Demo.ipynb)                                |
|  [**Bias test**](bias) : In this notebook we are evaluating `ner.dl` model on bias tests.                         | John Snow Labs                    | NER                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/test-specific-notebooks/Bias_Demo.ipynb)                                    |
|  [**Fairness test**](fairness) : In this notebook we are evaluating `ner.dl` model on fairness tests.                    | John Snow Labs                    | NER                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/test-specific-notebooks/Fairness_Demo.ipynb)                                |
|  [**Representation test**](representation) : In this notebook we are evaluating `ner.dl` model on representation tests.                | John Snow Labs                    | NER                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/test-specific-notebooks/Representation_Demo.ipynb)                          |
|  [**Robustness test**](robustness): In this notebook we are evaluating `ner.dl` model on robustness tests.                    | John Snow Labs                    | NER                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/test-specific-notebooks/Robustness_DEMO.ipynb)                              |
| [**Performace test**](performance) : In this notebook we are testing time taken to complete the tests in LangTest on the datasets with Models.                        | Hugging Face/John Snow Labs/Spacy | NER                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/PerformanceTest_Notebook.ipynb)                                          |
| [**Translation test**](translation) : In this section, we dive into testing translation models. We will use the Hugging Face Transformers library/John Snow Labs to load the translation models.   | Hugging Face/John Snow Labs       | Translation                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/task-specific-notebooks/Translation_Notebook.ipynb)                         |
| [**CrowS Pairs test**](stereotype) : In this notebook we are measuring the degree to which stereotypical biases are present in masked language models using Crows Pairs dataset                         | Hugging Face                      | Fill-Mask                     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/task-specific-notebooks/Crows_Pairs_Notebook.ipynb)                         |
| [**Stereoset test**](stereoset#gender-occupational-stereotype-notebook) : In this notebook we are evaluating Hugging Face models on StereoSet. StereoSet is a dataset and a method to evaluate the bias in LLM's. This dataset uses pairs of sentences, where one of them is more stereotypic and the other one is anti-stereotypic.                           | Hugging Face                      | Question-Answering                         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/task-specific-notebooks/StereoSet_Notebook.ipynb)                           |
| [**Wino-Bias test**](stereotype#wino-bias-huggingface-notebook) : In this tutorial, we assess the model on gender occupational stereotype statements using Hugging Face fill mask models.                          | Hugging Face                      | Fill-Mask                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/task-specific-notebooks/Wino_Bias.ipynb)                                    |
| [**Grammar test**](grammar) :  In this notebook we are evaluating `lvwerra/distilbert-imdb` model on grammar test.                             | Hugging Face                      | Text-Classification                      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/test-specific-notebooks/Grammar_Demo.ipynb)                                   |


</div><div class="h3-box" markdown="1">