---
layout: docs
header: true
seotitle: Data | LangTest | John Snow Labs
title: Data
key: docs-examples
permalink: /docs/pages/docs/data
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

**data**: `dict`

The provided code initializes an instance of the Harness class. It accepts a data parameter, which can be specified as a `dictionary` with the following attributes.

```python
{
   "data_source": "",
   "subset": "",
   "feature_column": "",
   "target_column": "",
   "split": "",
   "source": "huggingface"
}
```
<br/>

{:.table2}
| Key                          | Description                                                          |
| ---------------------------- | -------------------------------------------------------------------- |
| **data_source**(mandatory)   | Represents the name of the dataset being used.                       |
| **subset**(optional)         | Indicates the subset of the dataset being considered.                |
| **feature_column**(optional) | Specifies the column that contains the input features.               |
| **target_column**(optional)  | Represents the column that contains the target labels or categories. |
| **split**(optional)          | Denotes which split of the dataset should be used.                   |
| **source**(optional)         | Set to ‘huggingface’ when loading Hugging Face dataset.              |

</div><div class="h3-box" markdown="1">

#### Supported File formats

The following table provides an overview of the compatible data sources for each specific task.

{:.table2}
| Task                    | Supported Data Inputs                                    |
| ----------------------- | -------------------------------------------------------- |
| [**ner**](/docs/pages/task/ner)  | CoNLL, CSV and HuggingFace Datasets                      |
| [**text-classification**](/docs/pages/task/text-classification) | CSV and HuggingFace Datsets                              |
| [**question-answering**](/docs/pages/task/question-answering)  | benchmark datasets, curated datasets, CSV, HuggingFace Datsets |
| [**summarization**](/docs/pages/task/summarization)      |benchmark datasets, CSV, HuggingFace Datsets |
| [**fill-mask**](/docs/pages/task/fill-mask)    | curated datasets                         |
| [**translation**](/docs/pages/task/translation)            |curated datasets                    |
| [**text-generation**](/docs/pages/task/text-generation)       | curated datasets                          |


> Note: **data_source** formats are `task` and `category` dependent.

</div><div class="h3-box" markdown="1">

## NER

There are three options for datasets to test NER models: **`CoNLL`**, **`CSV`** and **HuggingFace** datasets. Here are some details of what these may look like:

#### CoNLL Format for NER

```bash
LEICESTERSHIRE NNP B-NP B-ORG
TAKE           NNP I-NP O
OVER           IN  B-PP O
AT             NNP B-NP O
TOP            NNP I-NP O
AFTER          NNP I-NP O
INNINGS        NNP I-NP O
VICTORY        NNP I-NP O
```

</div><div class="h3-box" markdown="1">

#### CSV Format for NER

{:.table2}
| Supported "text" column names               | Supported "ner" column names                                                     | Supported "pos" column names                     | Supported "chunk" column names |
| ------------------------------------------- | -------------------------------------------------------------------------------- |
| ['text', 'sentences', 'sentence', 'sample'] | ['label', 'labels ', 'class', 'classes', 'ner_tag', 'ner_tags', 'ner', 'entity'] | ['pos_tags', 'pos_tag', 'pos', 'part_of_speech'] | ['chunk_tags', 'chunk_tag']    |

</div><div class="h3-box" markdown="1">

#### Passing a NER Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task='ner',
                  model={'model': 'en_core_web_sm', 'hub':'spacy'},
                  data={"data_source":'test.conll'},
                  config='config.yml') #Either of the two formats can be specified.
```

#### Passing a Hugging Face Dataset for NER to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task="ner",
                  model={"model": "en_core_web_sm", "hub": "spacy"},
                  data={"data_source":'wikiann',
                  "subset":"en",
                  "feature_column":"tokens",
                  "target_column":'ner_tags',
                  "split":"test",
                  "source": "huggingface"
                  })
```


</div><div class="h3-box" markdown="1">

## Text Classification

There are 2 options for datasets to test Text Classification models: **`CSV`** datasets or loading **`HuggingFace Datasets`** containing the name, subset, split, feature_column and target_column for loading the HF datasets. Here are some details of what these may look like:

#### CSV Format for Text Classification

Here's a sample dataset:

{:.table2}
| text                                                                                                                                                                                                                                                                                                                                                                                                           | label |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| I thoroughly enjoyed Manna from Heaven. The hopes and dreams and perspectives of each of the characters is endearing and we, the audience, get to know each and every one of them, warts and all. And the ending was a great, wonderful and uplifting surprise! Thanks for the experience; I'll be looking forward to more.                                                                                    | 1     |
| Absolutely nothing is redeeming about this total piece of trash, and the only thing worse than seeing this film is seeing it in English class. This is literally one of the worst films I have ever seen. It totally ignores and contradicts any themes it may present, so the story is just really really dull. Thank god the 80's are over, and god save whatever man was actually born as "James Bond III". | 0     |

For `CSV` files, we support different variations of the column names. They are shown below :

{:.table2}
| Supported "text" column names               | Supported "label" column names           |
| ------------------------------------------- | ---------------------------------------- |
| ['text', 'sentences', 'sentence', 'sample'] | ['label', 'labels ', 'class', 'classes'] |

</div><div class="h3-box" markdown="1">

#### Passing a CSV Text Classification Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task='text-classification',
                  model={'model': 'mrm8488/distilroberta-finetuned-tweets-hate-speech', 'hub':'huggingface'},
                  data={"data_source":'sample.csv'},
                  config='config.yml')
```

</div><div class="h3-box" markdown="1">

#### Passing a Hugging Face Dataset for Text Classification to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task="text-classification",
                  model={'model': 'mrm8488/distilroberta-finetuned-tweets-hate-speech', 'hub':'huggingface'},
                  data={"data_source":'glue',
                  "subset":"sst2",
                  "feature_column":"sentence",
                  "target_column":'label',
                  "split":"train",
                  "source": "huggingface"
                  })
```

</div><div class="h3-box" markdown="1">

## Question Answering

Question Answering task contains various test-categories, and by default, the question answering task supports robustness, accuracy, fairness, representation, and bias for the benchmark dataset. However, if you want to access a specific sub-task (Category) within the question answering task, it is data-dependent.

Supported test categories and their corresponding supported data inputs are outlined below:

> Note: For bias we only support **data_source**:`BoolQ` and **split**:`bias`

{:.table2}
| Supported Test Categories                     | Supported Data                                           |
|-----------------------------------------------|----------------------------------------------------------|
| **[Robustness](/docs/pages/task/question-answering#robustness), [Accuracy](/docs/pages/task/question-answering#accuracy), [Fairness](/docs/pages/task/question-answering#fairness), [Representation](/docs/pages/task/question-answering#representation), [Grammar](/docs/pages/task/question-answering#grammar)** | Benchmark datasets, CSV, HuggingFace Datasets       |
| **[Bias](/docs/pages/task/question-answering#bias)**                                      | BoolQ (split: bias)                                      |
| **[Factuality](/docs/pages/task/question-answering#factuality)**                                | Factual-Summary-Pairs                                    |
| **[Ideology](/docs/pages/task/question-answering#ideology)**                                  | Curated list                                             |
| **[Legal](/docs/pages/task/question-answering#legal)**                                     | Legal-Support                                            |
| **[Sensitivity](/docs/pages/task/question-answering#sensitivity)**                               | NQ-Open, OpenBookQA, wikiDataset                         |
| **[Stereoset](/docs/pages/task/question-answering#stereoset)**                                 | StereoSet                                                |
| **[Sycophancy](/docs/pages/task/question-answering#sycophancy)**                                | synthetic-math-data, synthetic-nlp-data                  |

</div><div class="h3-box" markdown="1">

For the default Question Answering task, the user is meant to select a benchmark dataset. You can see the benchmarks page for all available benchmarks:
[Benchmarks](/docs/pages/benchmarks/benchmark). You can access the tutorial notebooks to get a quick start on your preferred dataset here: [Dataset Notebooks](/docs/pages/tutorials/Benchmark_Dataset_Notebook_Notebooks)

</div><div class="h3-box" markdown="1">

#### Passing a Question Answering Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task="question-answering", 
                  model={"model": "gpt-3.5-turbo-instruct", "hub":"openai"}, 
                  data={"data_source" :"BBQ", "split":"test-tiny"}, config='config.yml')

```


</div><div class="h3-box" markdown="1">

### Ideology

This test evaluates the model's political orientation. There is one default dataset used for this test.

#### Datasets

{:.table2}
| Dataset                        | Source                                                                                  | Description                                                      |Notebook        |   
| ------------------------------ | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |----------------|
| **Ideology Compass Questions** | [3 Axis Political Compass Test](https://github.com/SapplyValues/SapplyValues.github.io) | Political Compass questions, containing 40 questions for 2 axes. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/test-specific-notebooks/Political_Demo.ipynb) |

</div><div class="h3-box" markdown="1">

#### Passing a Disinformation Dataset to the Harness

In ideology test, the data is automatically loaded since there is only one dataset available for now:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task={"task":"question-answering", "category":"ideology"}, 
            model={'model': "gpt-3.5-turbo-instruct", "hub": "openai"})
```

</div><div class="h3-box" markdown="1">

### Factuality

The Factuality Test is designed to evaluate the ability of LLMs to determine the factuality of statements within summaries, particularly focusing on the accuracy of LLM-generated summaries and potential biases in their judgments. Users should choose a benchmark dataset from the provided list.

#### Datasets

{:.table2}
| Dataset                   | Source                                                                                                                                                                                             | Description                                             | Notebook        |   
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |-------------|
| **Factual-Summary-Pairs** | [LLAMA-2 is about as factually accurate as GPT-4 for summaries and is 30x cheaper](https://www.anyscale.com/blog/llama-2-is-about-as-factually-accurate-as-gpt-4-for-summaries-and-is-30x-cheaper) | Factual-Summary-Pairs, containing 371 labeled examples. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Factuality_Test.ipynb) |

</div><div class="h3-box" markdown="1">


#### Passing a Factuality Test Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness  =  Harness(task={"task":"question-answering", "category":"factuality"}, 
                    model = {"model": "gpt-3.5-turbo-instruct", "hub":"openai"},
                    data = {"data_source": "Factual-Summary-Pairs"})
```
</div><div class="h3-box" markdown="1">

### Legal

The Legal test assesses LLMs' ability to discern the level of support provided by various case summaries for a given legal claim.

#### Datasets

{:.table2}
| Dataset                   | Source                                                                                                                                                                                             | Description                                             | Notebook        |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |-------------|
| **legal-support** | [legal Support Scenario](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/legal_support_scenario.py) | The legal-support dataset includes 100 labeled examples designed to evaluate models' performance in discerning the level of support provided by different case summaries for a given legal claim. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Legal_Support.ipynb) |


</div><div class="h3-box" markdown="1">

#### Passing a Legal Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness  =  Harness(task={"task":"question-answering", "category":"legal"}, 
                    model = {"model": "gpt-3.5-turbo-instruct", "hub":"openai"},
                    data = {"data_source": "legal-support"})
```
</div><div class="h3-box" markdown="1">

### Sensitivity

The Sensitivity Test comprises two distinct evaluations: one focusing on assessing a model's responsiveness to toxicity, particularly when toxic words are introduced into the input text, and the other aimed at gauging its sensitivity to negations, especially when negations are inserted after verbs like "is," "was," "are," and "were". Users should choose a benchmark dataset from the provided list.

#### Test and Dataset Compatibility

{:.table2}

| Test Name | Supported Dataset                                                               |       split         | Notebook                                                                                                                                                                                                   |
| --------- | ------------------------------------------------------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Add Toxic Words](/docs/pages/tests/sensitivity#add-toxic-words)  | [wikiDataset](https://huggingface.co/datasets/wikitext)                              |  test, test-tiny | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Sensitivity_Test.ipynb) |
| [Add Negation](/docs/pages/tests/sensitivity#add-negation)  | [NQ-open](https://aclanthology.org/Q19-1026/) | test, test-tiny, combined  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Sensitivity_Test.ipynb) |
| [Add Negation](/docs/pages/tests/sensitivity#add-negation)  | [OpenBookQA](https://arxiv.org/abs/1809.02789) | test, test-tiny | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Sensitivity_Test.ipynb) |

#### Passing a Sensitivity Test Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness  =  Harness(task={"task":"question-answering", "category":"sensitivity"}, 
                    model = {"model": "gpt-3.5-turbo-instruct", "hub":"openai"},
                    data={"data_source" :"NQ-open","split":"test-tiny"})
```
### Stereoset

StereoSet test is designed to evaluate the ability of LLMs to measure stereotypical biases in four domains: gender, profession, race, and religion. The dataset consists of pairs of sentences, with one sentence being more stereotypical and the other being anti-stereotypical.

#### Datasets

{:.table2}
| Dataset                   | Source                                                                                                                                                                                             | Description                                             | Notebook        |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |-------------|
| **StereoSet** | [StereoSet: Measuring stereotypical bias in pretrained language models](https://paperswithcode.com/dataset/stereoset) | StereoSet dataset contains 4229 samples. This dataset uses pairs of sentences, where one of them is more stereotypic and the other one is anti-stereotypic. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/task-specific-notebooks/StereoSet_Notebook.ipynb) |

</div><div class="h3-box" markdown="1">

#### Passing a Stereoset Math Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(
    task={"task":"question-answering", "category":"stereoset"},
    model={"model": "bert-base-uncased","hub":"huggingface"},
    data ={"data_source":"StereoSet"})
```
</div><div class="h3-box" markdown="1">


### Sycophancy

Sycophancy is an undesirable behavior where models tailor their responses to align with a human user's view even when that view is not objectively correct. In this notebook, we propose a simple synthetic data intervention to reduce this behavior in language models.

#### Test and Dataset Compatibility

{:.table2}
| Test Name       | Supported Dataset    | Notebook                                                                                                                                                                                                  |
| --------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| sycophancy_math | sycophancy-math-data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Sycophancy_test.ipynb) |
| sycophancy_nlp  | sycophancy-nlp-data  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Sycophancy_test.ipynb) |

#### Passing a Sycophancy Math Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task={"task":"question-answering", "category":"sycophancy"},
                  model={"model": "gpt-3.5-turbo-instruct","hub":"openai"}, 
                  data={"data_source": 'synthetic-math-data',})
```
</div><div class="h3-box" markdown="1">


## Summarization

To test Summarization models, the user is meant to select a benchmark dataset from the available ones:
[Benchmarks](/docs/pages/benchmarks/benchmark). You can access the tutorial notebooks to get a quick start with your preferred dataset here: [Dataset Notebooks](/docs/pages/tutorials/Benchmark_Dataset_Notebook_Notebooks)

> Note: For bias we only support **data_source**:`BoolQ` and **split**:`bias`

</div><div class="h3-box" markdown="1">

#### Passing a Summarization Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task="summarization", 
                  model={"model": "gpt-3.5-turbo-instruct","hub":"openai"}, 
                  data={"data_source" :"XSum", "split":"test-tiny"},
                  config='config.yml')
   

```
</div><div class="h3-box" markdown="1">

#### Passing a Hugging Face Dataset for Summarization to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task="summarization", 
                  model={'model': 'gpt-3.5-turbo-instruct', 'hub':'openai'}, 
                  data={"data_source":'samsum',
                  "feature_column":"dialogue",
                  "target_column":'summary',
                  "split":"test",
                  "source": "huggingface"
                  })
```
</div><div class="h3-box" markdown="1">

## Fill Mask 

Fill Mask task currently supports only Stereotype test categories. Accessing a specific test within the Stereotype category depends on the dataset. The supported test categories and their corresponding data inputs are outlined below:

{:.table2}
| Supported Test Category | Supported Data                                  |
|-------------------------|-------------------------------------------------|
| [**Stereotype**](/docs/pages/task/fill-mask#stereotype)          | Wino-test, Crows-Pairs          |

### Stereotype

Stereotype tests play a crucial role in assessing the performance of models when it comes to common gender stereotypes and occupational biases. 

{:.table2}
| Test Name       | Supported Dataset    | Notebook                                                                                                                                                                                                  |
| --------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| wino-bias | Wino-test | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/task-specific-notebooks/Wino_Bias.ipynb) |
| crows-pairs  | Crows-Pairs  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/task-specific-notebooks/Crows_Pairs_Notebook.ipynb) |

</div><div class="h3-box" markdown="1">
#### Passing a Wino Bias Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python

# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(
                  task={"task": "fill-mask", "category": "wino-bias"}, 
                  model={"model" : "bert-base-uncased", "hub":"huggingface" } ,
                  data ={"data_source":"Wino-test"}
                  )
```

</div><div class="h3-box" markdown="1">
#### Passing a Crows Pairs Dataset to the Harness

```python

# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(
               task={"task": "fill-mask", "category": "crows-pairs"},
               model={"model" : "bert-base-uncased", "hub":"huggingface" } ,
               data = {"data_source":"Crows-Pairs"}
               )
```
</div><div class="h3-box" markdown="1">


## Text-generation

Text Generation task contains various test-categories. Accessing a specific sub-task (category) within the text generation task depends on the dataset. Supported test categories and their corresponding supported data inputs are outlined below:

{:.table2}
| Supported Test Category | Supported Data                                  |
|-------------------------|-------------------------------------------------|
| [**Clinical**](/docs/pages/task/text-generation#clinical)                |  Medical-files, Gastroenterology-files, Oromaxillofacial-files                           |
| [**Disinformation**](/docs/pages/task/text-generation#disinformation)      | Narrative-Wedging |
| [**Security**](/docs/pages/task/text-generation#security)            | Prompt-Injection-Attack |
| [**Toxicity**](/docs/pages/task/text-generation#toxicity)          | Real Toxicity Prompts |

</div><div class="h3-box" markdown="1">

### Clinical

Clinical test assesses LLMs' capability to detect demographic bias, which involves unfair treatment based on factors like age, gender, or race, regardless of patients' medical conditions.

#### Datasets

{:.table2}
| Dataset                   | Source                                                                                                                                                                                             | Description                                             | Notebook        |   
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |-------------|
| **Medical-files** | curated dataset | Medical-files, containing 49 labeled examples. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Clinical_Tests.ipynb) |
| **Gastroenterology-files** | curated dataset | Gastroenterology-files, containing 49 labeled examples. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Clinical_Tests.ipynb) |
| **Oromaxillofacial-files** | curated dataset | Oromaxillofacial-files, containing 49 labeled examples. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Clinical_Tests.ipynb) |

</div><div class="h3-box" markdown="1">

#### Passing a Clinical Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

model = {"model": "gpt-3.5-turbo-instruct", "hub": "openai"}

data = {"data_source": "Clinical", "split":"Medical-files"}

task = {"task": "text-generation", "category": "clinical"},

harness = Harness(task=task, model=model, data=data)
```
</div><div class="h3-box" markdown="1">

### Disinformation

This test evaluates the model's disinformation generation capability. Users should choose a benchmark dataset from the provided list.

#### Datasets

{:.table2}
| Dataset               | Source                                                                                                                                            | Notebook    |Description                                                |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------ |
| **Narrative-Wedging** | [Truth, Lies, and Automation How Language Models Could Change Disinformation](https://cset.georgetown.edu/publication/truth-lies-and-automation/) | Narrative-Wedging dataset, containing 26 labeled examples. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Disinformation_Test.ipynb) |

</div><div class="h3-box" markdown="1">

#### Passing a Disinformation Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness  =  Harness(task={"task":"text-generation", "category":"disinformation"}, 
                    model={"model": "j2-jumbo-instruct", "hub":"ai21"},
                    data = {"data_source": "Narrative-Wedging"})

```
</div><div class="h3-box" markdown="1">

### Security

The Security Test assesses LLMs' capability to identify and mitigate prompt injection vulnerabilities, which involve malicious prompts attempting to extract personal information or launch attacks on databases. 

#### Datasets

{:.table2}
| Dataset                   | Source                                                                                                                                                                                             | Description                                             | Notebook        |   
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |-------------|
| **Prompt-Injection-Attack** | curated dataset |Prompt-Injection-Attack, containing 17 examples. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Prompt_Injections_Tests.ipynb) |

</div><div class="h3-box" markdown="1">

#### Passing a Security Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

model={'model': "gpt-3.5-turbo-instruct", "hub": "openai"}

data = {"data_source": "Prompt-Injection-Attack", "split":"test"}

task={"task": "text-generation", "category": "security"}

harness = Harness(task=task, model=model, data=data)
```
</div><div class="h3-box" markdown="1">

### Toxicity

This test checks the toxicity of the completion., the user is meant to select a benchmark dataset from the following list:

#### Datasets

{:.table2}
| Dataset                | Source                                                                     | Description                                                                   |  Notebook  |
| ---------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------|
| **Toxicity** | [Real Toxicity Prompts](https://aclanthology.org/2020.findings-emnlp.301/) | Truncated set from the Real Toxicity Prompts Dataset, containing 80 examples. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/Toxicity_NB.ipynb)|

</div><div class="h3-box" markdown="1">

#### Passing a Toxicity Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python
# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task={"task":"text-generation", "category":"toxicity"}, 
                  model={"model": "gpt-3.5-turbo-instruct","hub":"openai"}, 
                  data={"data_source" :'Toxicity', "split":"test"})

```

</div><div class="h3-box" markdown="1">

## Translation


#### Datasets

{:.table2}
| Dataset                   | Source                                                                                                                                                                                             | Description                                             | Notebook        |   
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |-------------|
| **Translation** | [Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://paperswithcode.com/dataset/tatoeba) | Translation, containing 4400 examples. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/task-specific-notebooks/Translation_Notebook.ipynb) |

</div><div class="h3-box" markdown="1">

#### Passing a Translation Dataset to the Harness

In the Harness, we specify the data input in the following way:

```python

# Import Harness from the LangTest library
from langtest import Harness

harness = Harness(task="translation",
                  model={"model":'t5-base', "hub": "huggingface"},
                  data={"data_source": "Translation"})
```
</div><div class="h3-box" markdown="1">