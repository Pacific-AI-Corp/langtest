
<div class="h3-box" markdown="1">

## Custom Representation

Supported Custom representation Data Category:
- `Country-Economic-Representation`
- `Religion-Representation`
- `Ethnicity-Representation`
- `Label-Representation` (only ner)

#### How to Add Custom Representation

To add custom representation, you can follow these steps:

```python
# Import Harness from the LangTest library
from langtest import Harness

# Create a Harness object
harness = Harness(
    task="ner",
    model='en_core_web_sm',
    hub="spacy"
)

# Load custom representation data for ethnicity representation
harness.pass_custom_data(
    file_path='ethnicity_representation_data.json',
    test_name="Ethnicity-Representation",
    task="representation"
)
     
```
When adding custom representation data, it's important to note that each custom representation category may have a different data format for the JSON file. Ensure that the JSON file adheres to the specific format required for each category.

Additionally, it's important to remember that when you add custom representation data, it will affect a particular set of representation tests based on the category and data provided.

To learn more about the data format and how to structure the JSON file for custom representation data, you can refer to the tutorial available [here](https://colab.research.google.com/github/Pacific-AI-Corp/langtest/blob/main/demo/tutorials/test-specific-notebooks/Add_Custom_Data_Demo.ipynb).