---
layout: docs
header: true
seotitle: NLP Tutorials | Translation Test | John Snow Labs
title: Translation Notebook
key: test_specific_notebooks
permalink: /docs/pages/tutorials/test_specific_notebooks/translation
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In the Translation section, we're delving into the details of testing translation models by utilizing the Hugging Face Transformers library, alongside John Snow Labs. However, we're taking it a step further. Beyond the standard evaluation, we're introducing various perturbations to the input text, such as lowercasing and uppercasing. This approach allows us to assess the models' robustness and their ability to accurately translate text into different languages when faced with these variations. It's a methodical examination, ensuring these models can maintain accuracy even when encountering diverse linguistic challenges.

## Open in Collab

{:.table2}
| Category               | Hub                           | Task                              | Open In Colab                                                                                                                                                                                                                                    |
| ----------------------------------- |
| **Translation** :  | Hugging Face/John Snow Labs       | Translation                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/task-specific-notebooks/Translation_Notebook.ipynb)                         |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Config Used

```yml 
model_parameters:
    target_language: de
tests:
    defaults:
    min_pass_rate: 0.65
    robustness:
    lowercase:
        min_pass_rate: 0.66
    uppercase:
        min_pass_rate: 0.66
```

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Supported Tests

- **`uppercase`**: capitalization of the test set is turned into uppercase

- **`lowercase`**: capitalization of the test set is turned into lowercase

- **`titlecase`**: capitalization of the test set is turned into title case

- **`add_punctuation`**: special characters at end of each sentence are replaced by other special characters, if no
special character at the end, one is added

- **`strip_punctuation`**: special characters are removed from the sentences (except if found in numbers, such as '2.5')

- **`add_typo`**: typos are introduced in sentences

- **`add_contraction`**: contractions are added where possible (e.g. 'do not' contracted into 'don't')

- **`add_context`**: tokens are added at the beginning and at the end of the sentences

- **`swap_entities`**: named entities replaced with same entity type with same token count from terminology

- **`swap_cohyponyms`**: Named entities replaced with co-hyponym from the WordNet database

- **`american_to_british`**: American English will be changed to British English

- **`british_to_american`**: British English will be changed to American English

- **`number_to_word`**: Converts numeric values in sentences to their equivalent verbal representation.

- **`add_ocr_typo`**: Ocr typos are introduced in sentences

- **`add_speech_to_text_typo`**: Introduce common conversion errors from SSpeech to Text conversion.

- **`add_abbreviation`**:Replaces words or expressions in texts with their abbreviations

- **`multiple_perturbations`** : Transforms the given sentences by applying multiple perturbations in a specific sequence.

- **`adjective_synonym_swap`** : Transforms the adjectives in the given sentences to their synonyms.

- **`adjective_antonym_swap`** : Transforms the adjectives in the given sentences to their antonyms.

- **`strip_all_punctuation`**: Strips all punctuation from the sentences.

</div></div>