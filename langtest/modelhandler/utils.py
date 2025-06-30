# This file contains the model classes that are used in the model handler.
# from langchain
from typing import Dict, TypedDict, Union


class ModelInfo(TypedDict):
    module: str  # Module name for the model
    class_name: str  # Class name for the model


class Info(TypedDict):
    chat: ModelInfo
    completion: ModelInfo


MODEL_CLASSES: Dict[str, Union[Info, str]] = {
    "anthropic": {
        "chat": {
            "module": "langchain_anthropic.chat_models",
            "class_name": "ChatAnthropic",
        },
        "completion": {
            "module": "langchain_anthropic.llms",
            "class_name": "Anthropic",
        },
    },
    "anyscale": "ChatAnyscale",
    "azure-openai": {
        "chat": {
            "module": "langchain_openai.chat_models",
            "class_name": "AzureChatOpenAI",
        },
        "completion": {
            "module": "langchain_openai.llms",
            "class_name": "AzureOpenAI",
        },
    },
    "baichuan": "ChatBaichuan",
    "baidu_qianfan_endpoint": "QianfanChatEndpoint",
    "bedrock": "BedrockChat",
    "cohere": {
        "chat": {
            "module": "langchain_cohere.chat_models",
            "class_name": "ChatCohere",
        },
        "completion": {
            "module": "langchain_cohere.llms",
            "class_name": "Cohere",
        },
    },
    "databricks": {
        "chat": {
            "module": "databricks_langchain.chat_models",
            "class_name": "ChatDatabricks",
        },
        "completion": {
            "module": "databricks_langchain.llms",
            "class_name": "Databricks",
        },
    },
    "deepinfra": "ChatDeepInfra",
    "ernie": "ErnieBotChat",
    "everlyai": "ChatEverlyAI",
    "fake": "FakeListChatModel",
    "fireworks": {
        "chat": {
            "module": "langchain_fireworks.chat_models",
            "class_name": "ChatFireworks",
        },
        "completion": {
            "module": "langchain_fireworks.llms",
            "class_name": "Fireworks",
        },
    },
    "gigachat": "GigaChat",
    "google_palm": "ChatGooglePalm",
    "gpt_router": "GPTRouter",
    "huggingface": "ChatHuggingFace",
    "human": "HumanInputChatModel",
    "hunyuan": "ChatHunyuan",
    "javelin_ai_gateway": "ChatJavelinAIGateway",
    "jinachat": "JinaChat",
    "kinetica": "ChatKinetica",
    "konko": "ChatKonko",
    "litellm": "ChatLiteLLM",
    "litellm_router": "ChatLiteLLMRouter",
    "llama_edge": "LlamaEdgeChatService",
    "maritalk": "ChatMaritalk",
    "minimax": "MiniMaxChat",
    "mlflow": "ChatMlflow",
    "mlflow_ai_gateway": "ChatMLflowAIGateway",
    "ollama": {
        "chat": {
            "module": "langchain_ollama.chat_models",
            "class_name": "ChatOllama",
        },
        "completion": {
            "module": "langchain_ollama.llms",
            "class_name": "OllamaLLM",
        },
    },
    "openai": {
        "chat": {
            "module": "langchain_openai.chat_models",
            "class_name": "ChatOpenAI",
        },
        "completion": {
            "module": "langchain_openai.llms",
            "class_name": "OpenAI",
        },
    },
    "pai_eas_endpoint": "PaiEasChatEndpoint",
    "perplexity": "ChatPerplexity",
    "promptlayer_openai": "PromptLayerChatOpenAI",
    "sparkllm": "ChatSparkLLM",
    "tongyi": "ChatTongyi",
    "vertexai": "ChatVertexAI",
    "volcengine_maas": "VolcEngineMaasChat",
    "yandex": "ChatYandexGPT",
    "yuan2": "ChatYuan2",
    "zhipuai": "ChatZhipuAI",
}
