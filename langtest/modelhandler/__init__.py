import importlib
from .modelhandler import ModelAPI
import langtest.modelhandler.custom_modelhandler
import langtest.modelhandler.lmstudio_modelhandler

RENAME_HUBS = {
    "azureopenai": "azure-openai",
    "huggingfacehub": "huggingface-inference-api",
    "sparknlp": "johnsnowlabs",
    "pyspark": "johnsnowlabs",
    "transformers": "huggingface",
}

INSTALLED_HUBS = ["custom", "lm-studio", "web", "openrouter"]

libraries = [
    ("johnsnowlabs", "langtest.modelhandler.jsl_modelhandler"),
    ("sparknlp", "langtest.modelhandler.jsl_modelhandler"),
    ("pyspark", "langtest.modelhandler.jsl_modelhandler"),
    ("johnsnowlabs", "langtest.modelhandler.jsl_modelhandler"),
    ("transformers", "langtest.modelhandler.transformers_modelhandler"),
    ("spacy", "langtest.modelhandler.spacy_modelhandler"),
    ("langchain", "langtest.modelhandler.llm_modelhandler"),
]

for library_name, import_statement in libraries:
    if importlib.util.find_spec(library_name):
        importlib.import_module(import_statement)
        if library_name in RENAME_HUBS:
            INSTALLED_HUBS.append(RENAME_HUBS[library_name])
        else:
            INSTALLED_HUBS.append(library_name)


if "langchain" in INSTALLED_HUBS:
    import langchain

    LANGCHAIN_HUBS = {
        (
            RENAME_HUBS.get(hub.lower(), hub.lower())
            if hub.lower() in RENAME_HUBS
            else hub.lower()
        ): hub
        for hub in langchain.llms.__all__
    }
    LANGCHAIN_HUBS["openrouter"] = "openrouter"

    INSTALLED_HUBS += list(LANGCHAIN_HUBS.keys())
else:
    LANGCHAIN_HUBS = {}
