import os
import pickle
import click
import asyncio
from langtest import cli

# from pkg_resources import resource_filename
from importlib import resources
from langtest.config import read_config

try:
    from langtest.evaluation import LangtestRetrieverEvaluator
    from llama_index.readers import SimpleDirectoryReader
    from llama_index.node_parser import SimpleNodeParser
    from llama_index.llms import OpenAI
    from llama_index.evaluation import generate_question_context_pairs
    from llama_index.embeddings import HuggingFaceEmbedding
    from llama_index.indices import VectorStoreIndex
    from llama_index.service_context import ServiceContext, set_global_service_context
except ImportError as e:
    print(e, "please install llama_index using `pip install llama-index`")

default_qa_pkl = resources.files("langtest").joinpath(
    "data/Retrieval_Datasets/qa_dataset.pkl"
)

# set environment variables
os.environ["OPENAI_API_KEY"] = read_config("openai_api_key")


@cli.group("benchmark")
def benchmark():
    """Benchmark NLP and LLM models on various datasets
    with different tests using Langtest CLI."""
    pass


@benchmark.command("embeddings")
@click.option(
    "--model", "-m", type=str, required=True, help="Comma-separated list of models"
)
@click.option("--dataset", "-d", type=str, required=False)
@click.option("--config", "-c", type=str, required=False)
@click.option("--hub", "-h", type=str, required=False)
def embeddings(model, dataset, config, hub):
    """Benchmark embeddings."""
    models = [m.strip() for m in model.split(",")]

    for m in models:
        print(f"Initializing Embedding Model Evaluation with {m}")
        _ = EmbeddingPipeline(m, dataset, config, hub)


class BasePipeline:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        raise NotImplementedError


class EmbeddingPipeline(BasePipeline):
    def __init__(self, embed_model, dataset=None, config=None, hub=None):
        super().__init__()
        self.embed_model = embed_model
        self.dataset = dataset if dataset else default_qa_pkl
        self.config = config
        self.documents = None
        self.nodes = None
        self.llm = None
        self.hub = hub
        self.node_parser = None
        self.qa_dataset = None
        self.retriever_evaluator_HF = None
        self.retriever = None
        self.query_engine = None

        self.load_data()

    def load_data(self):
        print(self.dataset)
        if isinstance(self.dataset, str):
            if self.dataset.endswith(".pkl"):
                with open(self.dataset, "rb") as file:
                    out = pickle.load(file)
                    self.documents = out["documents"]
                    self.nodes = out["nodes"]
                    self.llm = out["llm"]
                    self.node_parser = out["node_parser"]
                    self.qa_dataset = out["qa_dataset"]
            else:
                documents = SimpleDirectoryReader(self.dataset).load_data()

                try:
                    # Define an LLM
                    llm = OpenAI(model="gpt-3.5-turbo")

                except Exception as e:
                    print(
                        f"An error occurred: {e}. Kindly add your key to environment using python -m langtest config set 'OPENAI_API_KEY'= KEY "
                    )

                # Build index with a chunk_size of 512
                node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
                nodes = node_parser.get_nodes_from_documents(documents)

                qa_dataset = generate_question_context_pairs(
                    nodes, llm=llm, num_questions_per_chunk=2
                )

                self.documents = documents
                self.nodes = nodes
                self.llm = llm
                self.node_parser = node_parser
                self.qa_dataset = qa_dataset

                with open(self.dataset, "wb") as file:
                    pickle.dump(
                        {
                            "documents": self.documents,
                            "nodes": self.nodes,
                            "llm": self.llm,
                            "node_parser": self.node_parser,
                            "qa_dataset": self.qa_dataset,
                        },
                        file,
                    )
        self.load_model()

    def load_model(self):
        if self.hub == "huggingface":
            em = HuggingFaceEmbedding(self.embed_model, trust_remote_code=True)
            servicecontext = ServiceContext.from_defaults(embed_model=em)
            set_global_service_context(servicecontext)
            vector_index = VectorStoreIndex(self.nodes, servicecontext=servicecontext)

        else:
            servicecontext = ServiceContext.from_defaults()
            set_global_service_context(servicecontext)
            vector_index = VectorStoreIndex(self.nodes)

        self.query_engine = vector_index.as_query_engine()
        self.retriever = vector_index.as_retriever(similarity_top_k=3)

        self.evaluator_config()

    def evaluator_config(self):
        self.retriever_evaluator_HF = LangtestRetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate"], retriever=self.retriever
        )

        self.retriever_evaluator_HF.setPerturbations(
            "add_typo",
            "dyslexia_word_swap",
            "add_ocr_typo",
            "add_contraction",
            "add_abbreviation",
            "add_speech_to_text_typo",
            "add_slangs",
            "adjective_synonym_swap",
        )

        print("Ready to run evaluation")
        self.run_evaluator()

    def run_evaluator(self):
        print("Running evaluation")

        async def eval_results():
            _ = await self.retriever_evaluator_HF.aevaluate_dataset(self.qa_dataset)
            df = self.retriever_evaluator_HF.display_results()
            return df

        result = asyncio.run(eval_results())
        print(result.to_string())
        return result.to_string()
