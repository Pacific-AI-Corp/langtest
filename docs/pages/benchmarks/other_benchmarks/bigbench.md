---
layout: docs
header: true
seotitle: BigBench Benchmark | LangTest | John Snow Labs
title: BigBench
key: benchmarks-bigbench
permalink: /docs/pages/benchmarks/other_benchmarks/bigbench/
aside:
    toc: true
sidebar:
    nav: benchmarks
show_edit_on_github: true
nav_key: benchmarks
modify_date: "2019-05-16"
---

<div class="h3-box" markdown="1">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/llm_notebooks/dataset-notebooks/Bigbench_dataset.ipynb)

**Source:** [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models](https://arxiv.org/abs/2206.04615)

BigBench is a large-scale benchmark for measuring the performance of language models across a wide range of natural language understanding and generation tasks. It consists of many subtasks, each with its own evaluation metrics and scoring system. The subsets included in LangTest are:
- Abstract-narrative-understanding
- DisambiguationQA
- DisflQA
- Causal-judgment


You can see which subsets and splits are available below.

</div><div class="h3-box" markdown="1">

### Abstract-narrative-understanding

{:.table2}
| Split                                                   | Details                                                                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **test**      | Testing set from the Bigbench/Abstract Narrative Understanding dataset, containing 1000 question and answers examples.                         |
| **test-tiny** | Truncated version of the test set from the Bigbench/Abstract Narrative Understanding dataset, containing 50 question and answers examples. |

#### Example

In the evaluation process, we start by fetching *original_context* and *original_question* from the dataset. The model then generates an *expected_result* based on this input. To assess model robustness, we introduce perturbations to the *original_context* and *original_question*, resulting in *perturbed_context* and *perturbed_question*. The model processes these perturbed inputs, producing an *actual_result*. The comparison between the *expected_result* and *actual_result* is conducted using the `llm_eval` approach (where llm is used to evaluate the model response). Alternatively, users can employ metrics like **String Distance** or **Embedding Distance** to evaluate the model's performance in the Question-Answering Task within the robustness category. For a more in-depth exploration of these approaches, you can refer to this [notebook](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Evaluation_Metrics.ipynb) discussing these three methods.


{:.table3}
| category   | test_type    | original_context                                         | original_question                  | perturbed_context                                           | perturbed_question                     | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | dyslexia_word_swap | The man who owned the little corner diner for fifty years decided to redecorate and update the place. He painted it a bright new color, took out all the old furnishings and changed the menu. Sadly, people stopped coming in for dinner. They loved the old, original nostalgic look. They didn't like the bright new design of the diner. | This narrative is a good illustration of the following proverb: <br>1. Don't put new wine into old bottles<br>2. A cat may look at a king<br>3. There's no accounting for tastes<br>4. Never judge a book by its cover<br>5. Silence is golden | th^e m^an v/ho owned the litt1e corner diner f^r fifty years decided t^o redecorate a^d update the placc. He painted i^t a bright n^ew cohor, took o^ut all the old furnishings a^d changed the menu. Sadly, pcople stopped coming i^n f^r dinncr. t^hey loved the old, original nostalgic l^ook. They didn't likc the bright new delign of the diner. | th1s narrative is a g00d illustration of t^e following proverb: <br>1. Don't x)ut neAv wine inlo old bottles<br>2. A cat m^ay l^ook at a king<br>3. There's n^o accounting f^r tastes<br>4. nevei judge a b6ok by it^s covcr<br>5. Silence is golden|4. Never judge a book by its cover | 4. Never judge a book by its cover. | True  |


> Generated Results for `gpt-3.5-turbo-instruct` model from `OpenAI`

</div><div class="h3-box" markdown="1">

### DisambiguationQA

{:.table2}
| Split                                                   | Details                                                                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **test**                      | Testing set from the Bigbench/DisambiguationQA dataset, containing 207 question answers examples.                                          |
| **test-tiny**                 | Truncated version of the test set from the Bigbench/DisambiguationQA dataset, containing 50 question and answers examples.                 |

#### Example

In the evaluation process, we start by fetching *original_question* from the dataset. The model then generates an *expected_result* based on this input. To assess model robustness, we introduce perturbations to the *original_question*, resulting in *perturbed_question*. The model processes these perturbed inputs, producing an *actual_result*. The comparison between the *expected_result* and *actual_result* is conducted using the `llm_eval` approach (where llm is used to evaluate the model response). Alternatively, users can employ metrics like **String Distance** or **Embedding Distance** to evaluate the model's performance in the Question-Answering Task within the robustness category. For a more in-depth exploration of these approaches, you can refer to this [notebook](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Evaluation_Metrics.ipynb) discussing these three methods.


{:.table3}
| category   | test_type    |  original_question                  | perturbed_question                     | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | add_abbreviation | The patient was referred to the specialist because he had a rare skin condition.<br> choice: Ambiguous<br> choice: The patient had a skin condition<br> choice: The specialist had a skin condition<br>Pronoun identification: Which of the following is correct?<br><br>1. The patient had a skin condition<br>2. The specialist had a skin condition<br>3. Ambiguous | da patient wuz referred 2 tdaspecialist cos he had a rare skin condition.<br> choice: Ambiguous<br> choice: Thdaatient had a skin condition<br> choice: Thedaecialist had a skin condition<br>Pronoun identification: Which of the dalowing is correct?<br><br>1. The pdaent had a skin condition<br>2. The spdaalist had a skin condition<br>3. Ambiguous | 1. The patient had a skin condition |3. Ambiguous  |False


> Generated Results for `gpt-3.5-turbo-instruct` model from `OpenAI`

</div><div class="h3-box" markdown="1">

### DisflQA

{:.table2}
| Split                                                   | Details                                                                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **test**                               | Testing set from the Bigbench/DisflQA dataset, containing 1000 question answers examples.                                                  |
| **test-tiny**                               | Truncated version of the test set from the Bigbench/DisflQA dataset, containing 50 question and answers examples.                          |

#### Example

In the evaluation process, we start by fetching *original_context* and *original_question* from the dataset. The model then generates an *expected_result* based on this input. To assess model robustness, we introduce perturbations to the *original_context* and *original_question*, resulting in *perturbed_context* and *perturbed_question*. The model processes these perturbed inputs, producing an *actual_result*. The comparison between the *expected_result* and *actual_result* is conducted using the `llm_eval` approach (where llm is used to evaluate the model response). Alternatively, users can employ metrics like **String Distance** or **Embedding Distance** to evaluate the model's performance in the Question-Answering Task within the robustness category. For a more in-depth exploration of these approaches, you can refer to this [notebook](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Evaluation_Metrics.ipynb) discussing these three methods.


{:.table3}
| category   | test_type    | original_context                                         | original_question                  | perturbed_context                                           | perturbed_question                     | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | add_abbreviation | The common allotrope of elemental oxygen on Earth is called dioxygen , O 2 . It is the form that is a major part of the Earth ' s atmosphere ( see Occurrence ) . O2 has a bond length of 121 pm and a bond energy of 498 kJ · mol − 1 , which is smaller than the energy of other double bonds or pairs of single bonds in the biosphere and responsible for the exothermic reaction of O2 with any organic molecule . Due to its energy content , O2 is used by complex forms of life , such as animals , in cellular respiration ( see Biological role ) . Other aspects of O 2 are covered in the remainder of this article . | What part the composition of the Earth ' s biosphere is comprised of oxygen no sorry the Earth ' s atmosphere ? | da common allotrope of elemental oxygen on Earth is called dioxygen , O 2 . It is tdaform that is a major part of thdaarth ' s atmosphere ( see Occurrence ) . O2 has a bond length of 121 pm and a bond energy of 498 kJ · mol − 1 , which is smaller than thedaergy of other double bonds or pairs of single bonds in the dasphere and responsible 4 the edahermic reaction of O2 with ne organic molecule . Due 2 its energy content , O2 is used by complex forms of life , such as animals , in cellular respiration ( see Biological role ) . Other aspects of O 2 r covered in the redander of this article . |wat part da composition of tdaEarth ' s biosphere is comprised of oxygen no sry thdaarth ' s atmosphere ? | The Earth's atmosphere is comprised of dioxygen, O2. | Dioxygen, O2, is a major part of the Earth's atmosphere.  | True |


> Generated Results for `gpt-3.5-turbo-instruct` model from `OpenAI`


</div><div class="h3-box" markdown="1">

### Causal-judgment

{:.table2}
| Split                                                   | Details                                                                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **test**                       | Testing set from the Bigbench/Causal Judgment dataset, containing 190 question and answers examples.                                       |
| **test-tiny**                  | Truncated version of the test set from the Bigbench/Causal Judgment dataset, containing 50 question and answers examples.                  |


#### Example

In the evaluation process, we start by fetching *original_context* and *original_question* from the dataset. The model then generates an *expected_result* based on this input. To assess model robustness, we introduce perturbations to the *original_context* and *original_question*, resulting in *perturbed_context* and *perturbed_question*. The model processes these perturbed inputs, producing an *actual_result*. The comparison between the *expected_result* and *actual_result* is conducted using the `llm_eval` approach (where llm is used to evaluate the model response). Alternatively, users can employ metrics like **String Distance** or **Embedding Distance** to evaluate the model's performance in the Question-Answering Task within the robustness category. For a more in-depth exploration of these approaches, you can refer to this [notebook](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Evaluation_Metrics.ipynb) discussing these three methods.


{:.table3}
| category   | test_type    | original_context                                         | original_question                  | perturbed_context                                           | perturbed_question                     | expected_result                | actual_result                  | pass   |
|-----------|-------------|---------------------------------------------------------|-----------------------------------|------------------------------------------------------------|---------------------------------------|-------------------------------|-------------------------------|-------|
| robustness | add_ocr_typo | The CEO of a company is sitting in his office when his Vice President of R&D comes in and says, 'We are thinking of starting a new programme. It will help us increase profits, but it will also harm the environment.' The CEO responds that he doesn't care about harming the environment and just wants to make as much profit as possible. The programme is carried out, profits are made and the environment is harmed | Did the CEO intentionally harm the environment? | tle CEO of a c0mpany is sitting i^n hif ofﬁve v\^hen hif Vice prefident of R&D comes in a^nd says, 'We are thinking of starting a neiv programme. i^t wiUi hclp us incrcase profits, b^ut it wiUi aUso harm tle environment.' tle CEO responds j^that he doesn't caie aboui harming tle environment and jus^t wants t^o makc as mueh profit as pofsible. tle programme is carried o^ut, profits are made and tle environment is harmed |Did tle CEO intentionally harm tle environment? | No. | No  | True |


> Generated Results for `gpt-3.5-turbo-instruct` model from `OpenAI`

</div>