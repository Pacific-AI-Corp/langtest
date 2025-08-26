
<div class="h3-box" markdown="1">

## Custom Bias

Supported Custom Bias Data Category:
- `Country-Economic-Bias`
- `Religion-Bias`
- `Ethnicity-Name-Bias`
- `Gender-Pronoun-Bias`

#### How to Add Custom Bias

To add custom bias, you can follow these steps:

```python
# Import Harness from the LangTest library
from langtest import Harness

# Create a Harness object
harness = Harness(
    task="ner",
    model='en_core_web_sm',
    hub="spacy"
)

# Load custom bias data for country economic bias
harness.pass_custom_data(
    file_path='economic_bias_data.json',
    test_name="Country-Economic-Bias",
    task="bias"
)
     
```
When adding custom bias data, it's important to note that each custom bias category may have a different data format for the JSON file. Ensure that the JSON file adheres to the specific format required for each category.

Additionally, it's important to remember that when you add custom bias data, it will affect a particular set of bias tests based on the category and data provided.

To learn more about the data format and how to structure the JSON file for custom bias data, you can refer to the tutorial available [here](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/test-specific-notebooks/Add_Custom_Data_Demo.ipynb).