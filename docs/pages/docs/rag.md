---
layout: docs
seotitle: RAG | LangTest | John Snow Labs
title: RAG
permalink: /docs/pages/docs/rag
key: docs-callback
modify_date: "2023-03-28"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/RAG/RAG_OpenAI.ipynb)                

It is divided into two main sections: constructing the RAG with LlamaIndex and assessing its performance using a combination of LlamaIndex and Langtest. Our primary focus is on the evaluation as a pivotal metric for gauging the accuracy and reliability of the RAG application across a variety of queries and data sources.

To measure the effectiveness of the retriever component, we introduce **LangtestRetrieverEvaluator** utilizing two key metrics: `Hit Rate` and `Mean Reciprocal Rank (MRR)`. Hit Rate assesses the percentage of queries where the correct answer is found within the top-k retrieved documents, essentially measuring the retriever's precision. Mean Reciprocal Rank, on the other hand, evaluates the rank of the highest-placed relevant document across all queries, providing a nuanced view of the retriever's accuracy. We use LlamaIndex's **generate_question_context_pairs** module for creating relevant question and context pairs, serving as the foundation for both retrieval and response evaluation phases in the RAG system.

The evaluation process employs a dual approach by assessing the system with both standard and perturbed queries generated through Langtest. This methodology ensures a thorough understanding of the retriever's robustness and adaptability under various conditions, reflecting its performance in real-world scenarios.

```python
from langtest.evaluation import LangtestRetrieverEvaluator

retriever_evaluator = LangtestRetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], 
    retriever=retriever
)
     
retriever_evaluator.setPerturbations("add_typo","dyslexia_word_swap", "add_ocr_typo") 

# Evaluate
eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)

retriever_evaluator.display_results()

```





</div></div>