---
layout: docs
header: true
seotitle: LangTest - Deliver Safe and Effective Language Models | John Snow Labs
title: LangTest Release Notes
permalink: /docs/pages/docs/langtest_versions/release_notes_1_9_0
key: docs-release-notes
modify_date: 2023-10-17
---

<div class="h3-box" markdown="1">

## 1.9.0

## 📢 Highlights

**LangTest 1.9.0 Release by John Snow Labs 🚀**

We're excited to announce the latest release of LangTest, featuring significant enhancements that bolster its versatility and user-friendliness. This update introduces the seamless integration of Hugging Face Callback, empowering users to effortlessly utilize this renowned platform. Another addition is our Enhanced Templatic Augmentation with Automated Sample Generation. We also expanded LangTest's utility in language testing by conducting comprehensive benchmarks across various models and datasets, offering deep insights into performance metrics. Moreover, the inclusion of additional Clinical Datasets like MedQA, PubMedQA, and MedMCQ broadens our scope to cater to diverse testing needs. Coupled with insightful blog posts and numerous bug fixes, this release further cements LangTest as a robust and comprehensive tool for language testing and evaluation.

-  Integration of Hugging Face's callback class in LangTest facilitates seamless incorporation of an automatic testing callback into transformers' training loop for flexible and customizable model training experiences.

- Enhanced Templatic Augmentation with Automated Sample Generation: A key addition in this release is our innovative feature that auto-generates sample templates for templatic augmentation. By setting generate_templates to True, users can effortlessly create structured templates, which can then be reviewed and customized with the show_templates option. 

- In our Model Benchmarking initiative, we conducted extensive tests on various models across diverse datasets (MMLU-Clinical, OpenBookQA, MedMCQA, MedQA), revealing insights into their performance and limitations, enhancing our understanding of the landscape for robustness testing.

- Enhancement: Implemented functionality to save model responses (actual and expected results) for original and perturbed questions from the language model (llm) in a pickle file. This enables efficient reuse of model outputs on the same dataset, allowing for subsequent evaluation without the need to rerun the model each time.

- Optimized API Efficiency with Bug Fixes in Model Calls.

</div><div class="h3-box" markdown="1">

##  🔥 Key Enhancements:

### 🤗 Hugging Face Callback Integration
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/HF_Callback_NER.ipynb)     
We introduced the callback class for utilization in transformers model training. Callbacks in transformers are entities that can tailor the training loop's behavior within the PyTorch or Keras Trainer. These callbacks have the ability to examine the training loop state, make decisions (such as early stopping), or execute actions (including logging, saving, or evaluation). LangTest effectively leverages this capability by incorporating an automatic testing callback. This class is both flexible and adaptable, seamlessly integrating with any transformers model for a customized experience.

Create a callback instance with one line and then use it in the callbacks of trainer:
```python
my_callback = LangTestCallback(...)
trainer = Trainer(..., callbacks=[my_callback])
```

{:.table2}
| Parameter             | Description |
| --------------------- | ----------- |
| **task**              | Task for which the model is to be evaluated (text-classification or ner) |
| **data**              | The data to be used for evaluation. A dictionary providing flexibility and options for data sources. It should include the following keys: <br>• data_source (mandatory): The source of the data.<br>• subset (optional): The subset of the data.<br>• feature_column (optional): The column containing the features.<br>• target_column (optional): The column containing the target labels.<br>• split (optional): The data split to be used.<br>• source (optional): Set to 'huggingface' when loading Hugging Face dataset. |
| **config**            | Configuration for the tests to be performed, specified in the form of a YAML file. |
| **print_reports**     | A bool value that specifies if the reports should be printed. |
| **save_reports**      | A bool value that specifies if the reports should be saved. |
| **run_each_epoch**    | A bool value that specifies if the tests should be run after each epoch or the at the end of training |

</div><div class="h3-box" markdown="1">

### 🚀 Enhanced Templatic Augmentation with Automated Sample Generation
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Templatic_Augmentation_Notebook.ipynb)     

Users can now enable the automatic generation of sample templates by setting generate_templates to True. This feature utilizes the advanced capabilities of LLMs to create structured templates that can be used for templatic augmentation.To ensure quality and relevance, users can review the generated templates by setting show_templates to True. 

![image](https://github.com/JohnSnowLabs/langtest/assets/71844877/1ca52964-7ad2-4778-a76e-642c425b1c13)

###  🚀 Benchmarking Different Models

In our Model Benchmarking initiative, we conducted comprehensive tests on a range of models across diverse datasets. This rigorous evaluation provided valuable insights into the performance of these models, pinpointing areas where even large language models exhibit limitations. By scrutinizing their strengths and weaknesses, we gained a deeper understanding of the landscape

#### MMLU-Clinical

We focused on extracting clinical subsets from the MMLU dataset, creating a specialized MMLU-clinical dataset. This curated dataset specifically targets clinical domains, offering a more focused evaluation of language understanding models. It includes questions and answers related to clinical topics, enhancing the assessment of models' abilities in medical contexts. Each sample presents a question with four choices, one of which is the correct answer. This curated dataset is valuable for evaluating models' reasoning, fact recall, and knowledge application in clinical scenarios.

#### How the Dataset Looks

{:.table3}
| category   | test_type | original_question                                   | perturbed_question                                 | expected_result | actual_result | pass |
|------------|-----------|------------------------------------------------------|----------------------------------------------------|------------------|---------------|------|
| robustness | uppercase | Fatty acids are transported into the mitochondria bound to:\nA. thiokinase. B. coenzyme A (CoA). C. acetyl-CoA. D. carnitine. | FATTY ACIDS ARE TRANSPORTED INTO THE MITOCHONDRIA BOUND TO: A. THIOKINASE. B. COENZYME A (COA). C. ACETYL-COA. D. CARNITINE. | D. carnitine.                | B. COENZYME A (COA).             | False |

![mmlu](https://github.com/JohnSnowLabs/langtest/assets/71117423/ef056c0b-76dd-4dc1-b201-f6d97b880d1f)


#### OpenBookQA

The OpenBookQA dataset is a collection of multiple-choice questions that require complex reasoning and inference based on general knowledge, similar to an “open-book” exam. The questions are designed to test the ability of natural language processing models to answer questions that go beyond memorizing facts and involve understanding concepts and their relations. The dataset contains 500 questions, 
each with four answer choices and one correct answer. The questions cover various topics in science, such as biology, chemistry, physics, and astronomy.

#### How the Dataset Looks

{:.table3}
| category   | test_type | original_question                                                                                 | perturbed_question                                                                            | expected_result | actual_result | pass |
|------------|-----------|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|------------------|----------------|------|
| robustness | uppercase | There is most likely going to be fog around: A. a marsh B. a tundra C. the plains D. a desert"                                                      | THERE IS MOST LIKELY GOING TO BE FOG AROUND: A. A MARSH B. A TUNDRA C. THE PLAINS D. A DESERT" |A marsh                                                 | A MARSH                                            | True |



![openbook1](https://github.com/JohnSnowLabs/langtest/assets/101416953/fcc6a1d3-2b8b-444f-b4fc-4ced0d9ecd60)

#### MedMCQA

The MedMCQA is a large-scale benchmark dataset of Multiple-Choice Question Answering (MCQA) dataset designed to address real-world medical entrance exam questions. 

#### How the Dataset Looks

{:.table3}
| category   | test_type | original_question                                   | perturbed_question                                 | expected_result | actual_result | pass |
|------------|-----------|------------------------------------------------------|----------------------------------------------------|------------------|---------------|------|
| robustness | uppercase | Most common site of direct hernia\nA. Hesselbach's triangle\nB. Femoral gland\nC. No site predilection\nD. nan | MOST COMMON SITE OF DIRECT HERNIA A. HESSELBACH'S TRIANGLE B. FEMORAL GLAND C. NO SITE PREDILECTION D. NAN | A                | A             | True |

![medmcq](https://github.com/JohnSnowLabs/langtest/assets/101416953/b155fc64-308b-4a80-bc78-68afe37e3933)
**Dataset info:**
- subset: MedMCQA-Test
- Split: Medicine, Anatomy, Forensic_Medicine, Microbiology, Pathology, Anaesthesia, Pediatrics, Physiology, Biochemistry, Gynaecology_Obstetrics, Skin, Surgery, Radiology


#### MedQA

The MedQA is a benchmark dataset of Multiple choice question answering based on the United States Medical License Exams (USMLE). The dataset is collected from the professional medical board exams.

#### How the Dataset Looks

{:.table3}
| original_question                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | perturbed_question                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | expected_result | actual_result | pass |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|----------------|------|
| A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon.......Which of the following is the correct next action for the resident to take?\nA. Disclose the error to the patient but leave it out of the operative report\nB. Disclose the error to the patient and put it in the operative report\nC. Tell the attending that he cannot fail to disclose this mistake\nD. Report the physician to the ethics committee\nE. Refuse to dictate the operative report | A JUNIOR ORTHOPAEDIC SURGERY RESIDENT IS COMPLETING A CARPAL TUNNEL REPAIR WITH THE DEPARTMENT CHAIRMAN AS THE ATTENDING PHYSICIAN. DURING THE CASE, THE RESIDENT INADVERTENTLY CUTS A FLEXOR TENDON......WHICH OF THE FOLLOWING IS THE CORRECT NEXT ACTION FOR THE RESIDENT TO TAKE? A. DISCLOSE THE ERROR TO THE PATIENT BUT LEAVE IT OUT OF THE OPERATIVE REPORT B. DISCLOSE THE ERROR TO THE PATIENT AND PUT IT IN THE OPERATIVE REPORT C. TELL THE ATTENDING THAT HE CANNOT FAIL TO DISCLOSE THIS MISTAKE D. REPORT THE PHYSICIAN TO THE ETHICS COMMITTEE E. REFUSE TO DICTATE THE OPERATIVE REPORT | B                | C              | False|




![medqa](https://github.com/JohnSnowLabs/langtest/assets/101416953/3e486804-b184-43cd-ab8e-5289987f2db9)

## 🚀 Community Contributions:

Our team has published the below enlightening blogs on Hugging Face's community platform:

- [Streamlining ML Workflows: Integrating MLFlow Tracking with LangTest for Enhanced Model Evaluations](https://huggingface.co/blog/arshaan-nazir/streamlining-ml-workflows-langtest)

- [Evaluating Large Language Models on Gender-Occupational Stereotypes Using the Wino Bias Test](https://huggingface.co/blog/Rakshit122/gender-occupational-stereotypes)

## 🚀 New LangTest Blogs:

{:.table2}
| Blog | Description |
| --- | --- |
| [LangTest Insights: A Deep Dive into LLM Robustness on OpenBookQA](https://medium.com/john-snow-labs/langtest-insights-a-deep-dive-into-llm-robustness-on-openbookqa-ab0ddcbd2ab1) | Explore the robustness of Language Models (LLMs) on the OpenBookQA dataset with LangTest Insights.| 
| Unveiling Sentiments: Exploring LSTM-based Sentiment Analysis with PyTorch on the IMDB Dataset (To be Published ) | Explore the robustness of custom models with LangTest Insights. |
| LangTest: A Secret Weapon for Improving the Robustness of Your Transformers Language Models (To be Published ) | Explore the robustness of Transformers Language Models with LangTest Insights. |


## 🐛 Bug Fixes

* fixed LangTestCallback 
* Add predict_raw method to PretrainedCustomModel

## ⚒️ Previous Versions

</div>
{%- include docs-langtest-pagination.html -%}
