---
layout: docs
header: true
seotitle: LangTest - Deliver Safe and Effective Language Models | John Snow Labs
title: LangTest Release Notes
permalink: /docs/pages/docs/langtest_versions/release_notes_1_10_0
key: docs-release-notes
modify_date: 2023-10-17
---

<div class="h3-box" markdown="1">

## 1.10.0

## 📢 Highlights


🌟 **LangTest 1.10.0 Release by John Snow Labs**

We're thrilled to announce the latest release of LangTest, introducing remarkable features that elevate its capabilities and user-friendliness. This update brings a host of enhancements:

- **Evaluating RAG with LlamaIndex and Langtest**: LangTest seamlessly integrates LlamaIndex for constructing a RAG and employs LangtestRetrieverEvaluator, measuring retriever precision (Hit Rate) and accuracy (MRR) with both standard and perturbed queries, ensuring robust real-world performance assessment.

- **Grammar Testing for NLP Model Evaluation:** This approach entails creating test cases through the paraphrasing of original sentences. The purpose is to evaluate a language model's proficiency in understanding and interpreting the nuanced meaning of the text, enhancing our understanding of its contextual comprehension capabilities.


- **Saving and Loading the Checkpoints:** LangTest now supports the seamless saving and loading of checkpoints, providing users with the ability to manage task progress, recover from interruptions, and ensure data integrity.

- **Extended Support for Medical Datasets:** LangTest adds support for additional medical datasets, including LiveQA, MedicationQA, and HealthSearchQA. These datasets enable a comprehensive evaluation of language models in diverse medical scenarios, covering consumer health, medication-related queries, and closed-domain question-answering tasks.


- **Direct Integration with Hugging Face Models:**  Users can effortlessly pass any Hugging Face model object into the LangTest harness and run a variety of tasks. This feature streamlines the process of evaluating and comparing different models, making it easier for users to leverage LangTest's comprehensive suite of tools with the wide array of models available on Hugging Face.


</div><div class="h3-box" markdown="1">

##  🔥 Key Enhancements:

### 🚀Implementing and Evaluating RAG with LlamaIndex and Langtest
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/JohnSnowLabs/langtest/blob/main/demo/tutorials/RAG/RAG_OpenAI.ipynb)   

LangTest seamlessly integrates LlamaIndex, focusing on two main aspects: constructing the RAG with LlamaIndex and evaluating its performance. The integration involves utilizing LlamaIndex's generate_question_context_pairs module to create relevant question and context pairs, forming the foundation for retrieval and response evaluation in the RAG system.

To assess the retriever's effectiveness, LangTest introduces LangtestRetrieverEvaluator, employing key metrics such as Hit Rate and Mean Reciprocal Rank (MRR). Hit Rate gauges the precision by assessing the percentage of queries with the correct answer in the top-k retrieved documents. MRR evaluates the accuracy by considering the rank of the highest-placed relevant document across all queries. This comprehensive evaluation, using both standard and perturbed queries generated through LangTest, ensures a thorough understanding of the retriever's robustness and adaptability under various conditions, reflecting its real-world performance.

```
from langtest.evaluation import LangtestRetrieverEvaluator

retriever_evaluator = LangtestRetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)
     
retriever_evaluator.setPerturbations("add_typo","dyslexia_word_swap", "add_ocr_typo") 

# Evaluate
eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)

retriever_evaluator.display_results()

```

### 📚Grammar Testing in Evaluating and Enhancing NLP Models
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/JohnSnowLabs/langtest/blob/main/demo/tutorials/test-specific-notebooks/Grammar_Demo.ipynb)   

Grammar Testing is a key feature in LangTest's suite of evaluation strategies, emphasizing the assessment of a language model's proficiency in contextual understanding and nuance interpretation. By creating test cases that paraphrase original sentences, the goal is to gauge the model's ability to comprehend and interpret text, thereby enriching insights into its contextual mastery.

{:.table3}
| Category | Test Type  | Original                                                                                                                                                    | Test Case                                                                                                                         | Expected Result | Actual Result | Pass  |
|----------|------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:|------------------|---------------|-------|
| grammar  | paraphrase | This program was on for a brief period when I was a kid, I remember watching it whilst eating fish and chips.<br /><br />Riding on the back of the Tron hype this series was much in the style of streethawk, manimal and the like, except more computery. There was a geeky kid who's computer somehow created this guy - automan. He'd go around solving crimes and the lot.<br /><br />All I really remember was his fancy car and the little flashy cursor thing that used to draw the car and help him out generally.<br /><br />When I mention it to anyone they can remember very little too. Was it real or maybe a dream? | I remember watching a show from my youth that had a Tron theme, with a nerdy kid driving around with a little flashy cursor and solving everyday problems. Was it a genuine story or a mere dream come true? | NEGATIVE         | POSITIVE      | false |

### 🔥 Saving and Loading the Checkpoints
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Saving_Checkpoints.ipynb)     
Introducing a robust checkpointing system in LangTest! The `run` method in the `Harness` class now supports checkpointing, allowing users to save intermediate results, manage batch processing, and specify a directory for storing checkpoints and results. This feature ensures data integrity, providing a mechanism for recovering progress in case of interruptions or task failures.
```
harness.run(checkpoint=True, batch_size=20,save_checkpoints_dir="imdb-checkpoint")
```
The `load_checkpoints` method facilitates the direct loading of saved checkpoints and data, providing a convenient mechanism to resume testing tasks from the point where they were previously interrupted, even in the event of runtime failures or errors.
```
harness = Harness.load_checkpoints(save_checkpoints_dir="imdb-checkpoint",
                                   task="text-classification",
                                   model = {"model": "lvwerra/distilbert-imdb" , "hub":"huggingface"}, )
```

### 🏥 Added Support for More Medical Datasets

#### LiveQA
The LiveQA'17 medical task focuses on consumer health question answering. It consists of constructed medical question-answer pairs for training and testing, with additional annotations. LangTest now supports LiveQA for comprehensive medical evaluation.

##### How the dataset looks:

{:.table3}
| category   | test_type | original_question                                   | perturbed_question                                      | expected_result                                                | actual_result                                              | eval_score | pass |
|------------|-----------|------------------------------------------------------|-----------------------------------------------------------|-----------------------------------------------------------------|------------------------------------------------------------|------------|------|
| robustness | uppercase | Do amphetamine salts 20mg tablets contain gluten?    | DO AMPHETAMINE SALTS 20MG TABLETS CONTAIN GLUTEN?           | No, amphetamine salts 20mg tablets do not contain gluten.       | No, Amphetamine Salts 20mg Tablets do not contain gluten.    | 1.0        | true |

#### MedicationQA
The MedicationQA dataset consists of commonly asked consumer questions about medications. It includes annotations corresponding to drug focus and interactions. LangTest now integrates MedicationQA for thorough evaluation of models in medication-related scenarios.

##### How the dataset looks:

{:.table3}
| category   | test_type | original_question                                  | perturbed_question                                     | expected_result                                                                                                                                     | actual_result                                                                                                                                                                    | eval_score          | pass |
|------------|-----------|-----------------------------------------------------|--------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|------|
| robustness | uppercase | how does rivatigmine and otc sleep medicine interact | HOW DOES RIVATIGMINE AND OTC SLEEP MEDICINE INTERACT     | Rivastigmine is a cholinesterase inhibitor and OTC (over-the-counter) sleep medicine is a sedative. There is a risk of excessive sedation when taking both of these medications together. Patients should consult their doctor before taking both of these medications together. | Rivastigmine and OTC sleep medicines may interact. Rivastigmine can increase the effects of sedatives, including OTC sleep medicines, resulting in increased drowsiness. It's important to talk to your doctor before taking OTC sleep medicines while taking Rivastigmine. | 0.9365371414708803 | true |


#### HealthSearchQA
HealthSearchQA, a new free-response dataset of medical questions sought online, with six existing open-question answering datasets covering professional medical exams, research, and consumer queries. The HealthsearchQA dataset consists of 3375 frequently-asked consumer questions and was curated using seed medical diagnoses and their related symptoms.

##### How the dataset looks:

{:.table3}
| category   | test_type | original_question                       | perturbed_question                          | expected_result                                                                                                                | actual_result                                                                                                              | eval_score          | pass |
|------------|-----------|------------------------------------------|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|---------------------|------|
| robustness | uppercase | Are dental abscesses serious?            | ARE DENTAL ABSCESSES SERIOUS?               | Yes, dental abscesses are serious and can lead to infection and damage to the surrounding tissue if left untreated. Treatment typically involves antibiotics and/or draining the abscess. If left untreated, the infection can spread to other parts of the body. | Dental abscesses can be serious and require prompt medical attention. Left untreated, they can cause swelling, spreading infections, and damage to the surrounding teeth and bone. | 0.9457038739103363 | true |



### 🚀Direct Integration with Hugging Face Models

Users can effortlessly pass any Hugging Face model object into the LangTest harness and run a variety of tasks. This feature streamlines the process of evaluating and comparing different models, making it easier for users to leverage LangTest's comprehensive suite of tools with the wide array of models available on Hugging Face.

![image](https://github.com/JohnSnowLabs/langtest/assets/71844877/adef09b7-e33d-42ec-86f3-a96dea85387e)


## 🚀 New LangTest Blogs:

{:.table2}
| Blog | Description |
| --- | --- |
| [LangTest: A Secret Weapon for Improving the Robustness of Your Transformers Language Models](https://www.johnsnowlabs.com/langtest-a-secret-weapon-for-improving-the-robustness-of-your-transformers-language-models/) | Explore the robustness of Transformers Language Models with LangTest Insights. |
| [Testing the Robustness of LSTM-Based Sentiment Analysis Models](https://medium.com/john-snow-labs/testing-the-robustness-of-lstm-based-sentiment-analysis-models-67ed84e42997) | Explore the robustness of custom models with LangTest Insights. |

## 🐛 Bug Fixes

- Fixed LangTestCallback errors
- Fixed QA, Default Config, and Transformer Model for QA
- Fixed multi-model evaluation
- Fixed datasets format

## ⚒️ Previous Versions

</div>
{%- include docs-langtest-pagination.html -%}
