---
layout: docs
header: true
seotitle: Tutorials | Editing Testcases | LangTest | John Snow Labs
title: Editing Testcases
key: miscellaneous_notebooks
permalink: /docs/pages/tutorials/miscellaneous_notebooks/editing_testcases
sidebar:
    nav: tutorials
aside:
    toc: true
nav_key: tutorials
show_edit_on_github: true
modify_date: "2023-11-15"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Overview

In the Editing Testcases notebook, `.edit_testcases()` method of harness is explained and used. When it is called after the generation of the testcases, it exports the testcases into a csv and allows the user to edit the testcases before running the tests. Then the user can import the edited testcases back into the harness object with `.import_edited_testcases()` and run the tests.

## Open in Collab

{:.table2}
| Category                                                                                          | Hub          | Task | Open In Colab                                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------- |
| **Editing Testcases**  | Hugging Face | NER  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/misc/Editing_TestCases_Notebook.ipynb) |

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## Config Used

```yml 
tests:
  defaults:
    min_pass_rate: 0.65
  robustness:
    lowercase:
      min_pass_rate: 0.66
    uppercase:
      min_pass_rate: 0.66
```
