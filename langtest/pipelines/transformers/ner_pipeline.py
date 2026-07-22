import os
import logging

from metaflow import FlowSpec, JSONType, Parameter, step
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from langtest import Harness
from langtest.datahandler.datasource import DataFactory
from langtest.pipelines.utils.data_helpers.ner_dataset import NERDataset
from langtest.pipelines.utils.metrics import compute_ner_metrics
from langtest.tasks import TaskManager
from langtest.errors import Warnings


class NEREnd2EndPipeline(FlowSpec):
    """NER pipeline for Huggingface models

    It executes the following workflow in a sequential order:
    - train a model on a given dataset
    - evaluate the model on a given test dataset
    - test the trained model on a set of tests
    - augment the training set based on the tests outcome
    - retrain the model on a the freshly generated augmented training set
    - evaluate the retrained model on the test dataset
    - compare the performance of the two models

    The pipeline can directly be triggered through the CLI via the following one liner:
    ```bash
    python3 langtest/pipelines/transformers_pipelines.py run \
            --model-name="bert-base-uncased" \
            --train-data=tner.csv \
            --eval-data=tner.csv \
            --training-args='{"per_device_train_batch_size": 4, "max_steps": 3}' \
            --feature-col="tokens" \
            --target-col="ner_tags"
    ```
    """

    model_name = Parameter(
        "model-name", help="Name of the pretrained model to load", type=str, required=True
    )
    train_data = Parameter(
        "train-data", help="Path to the train dataset", type=str, required=True
    )
    eval_data = Parameter(
        "eval-data", help="Path to the evaluation dataset", type=str, required=True
    )
    feature_col = Parameter(
        "feature-col",
        help="Name of the feature column to use",
        type=str,
        required=False,
        default="text",
    )
    target_col = Parameter(
        "target-col",
        help="Name of the target column to use",
        type=str,
        required=False,
        default="labels",
    )
    config = Parameter(
        "config", help="Tests configuration", type=JSONType, required=False, default=None
    )
    training_args = Parameter(
        "training-args",
        help="Training arguments to pass to the Trainer",
        type=JSONType,
        required=True,
    )

    @step
    def start(self):
        """Starting step of the flow (required by Metaflow)"""
        self.next(self.setup)

    @step
    def setup(self):
        """Performs all the necessary set up steps"""
        self.task = "ner"
        self.hub = "huggingface"
        self.output_dir = "checkpoints/"

        self.train_datasource = DataFactory(
            file_path={"data_source": self.train_data}, task=TaskManager(self.task)
        )
        self.eval_datasource = DataFactory(
            file_path={"data_source": self.eval_data}, task=TaskManager(self.task)
        )

        self.next(self.train)

    @step
    def train(self):
        """Performs the training procedure of the model"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        samples = self.train_datasource.load_raw()
        self.train_dataset = NERDataset(
            tokens=[sample[self.feature_col] for sample in samples],
            labels=[sample[self.target_col] for sample in samples],
            tokenizer=self.tokenizer,
        )

        samples = self.eval_datasource.load_raw()
        self.eval_dataset = NERDataset(
            tokens=[sample[self.feature_col] for sample in samples],
            labels=[sample[self.target_col] for sample in samples],
            tokenizer=self.tokenizer,
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.train_dataset.label_map),
            id2label=self.train_dataset.id2label,
            label2id=self.train_dataset.label_map,
            ignore_mismatched_sizes=True,
        )

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir=self.output_dir, **self.training_args),
            train_dataset=self.train_dataset,
            processing_class=self.tokenizer,
        )
        trainer.train()
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Performs the evaluation procedure on the given test set"""
        trained_model = AutoModelForTokenClassification.from_pretrained(
            self.output_dir,
            num_labels=len(self.train_dataset.label_map),
            id2label=self.train_dataset.id2label,
            label2id=self.train_dataset.label_map,
            ignore_mismatched_sizes=True,
        )
        self.metrics = Trainer(
            model=trained_model,
            eval_dataset=self.eval_dataset,
            compute_metrics=compute_ner_metrics(
                list(self.train_dataset.label_map.keys())
            ),
        ).evaluate()

        self.next(self.test)

    @step
    def test(self):
        """Performs the testing procedure of the model on a set of tests using langtest"""
        self.harness = Harness(
            task=self.task,
            model={"model": self.output_dir, "hub": self.hub},
            data={"data_source": self.train_data},
        )
        if self.config:
            self.harness.configure(self.config)

        _ = self.harness.generate()
        self.harness.save(save_dir="saved_harness")
        _ = self.harness.run()
        self.harness.report(format="dataframe", save_dir="first_report")

        self.next(self.augment)

    @step
    def augment(self):
        """Performs the data augmentation procedure based on langtest"""
        filename = os.path.basename(self.train_data)
        self.path_augmented_file = os.path.join(os.getcwd(), f"augmented_{filename}")
        self.harness.augment(
            training_data={"data_source": self.train_data},
            save_data_path=self.path_augmented_file,
            export_mode="add",
        )

        self.next(self.retrain)

    @step
    def retrain(self):
        """Performs the training procedure using the augmented data created by langtest"""
        self.augmented_train_datasource = DataFactory(
            file_path={"data_source": self.path_augmented_file},
            task=TaskManager(self.task),
        )
        samples = self.augmented_train_datasource.load_raw()

        self.augmented_train_dataset = NERDataset(
            tokens=[sample["text"] for sample in samples],
            labels=[sample["labels"] for sample in samples],
            tokenizer=self.tokenizer,
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.augmented_train_dataset.label_map),
            id2label=self.augmented_train_dataset.id2label,
            label2id=self.augmented_train_dataset.label_map,
            ignore_mismatched_sizes=True,
        )

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=f"augmented_{self.output_dir}", **self.training_args
            ),
            train_dataset=self.augmented_train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )
        trainer.train()
        self.model.save_pretrained(f"augmented_{self.output_dir}")
        self.tokenizer.save_pretrained(f"augmented_{self.output_dir}")

        self.next(self.reevaluate)

    @step
    def reevaluate(self):
        """Performs the evaluation procedure of the model training on the augmented dataset"""
        newly_trained_model = AutoModelForTokenClassification.from_pretrained(
            f"augmented_{self.output_dir}",
            num_labels=len(self.train_dataset.label_map),
            id2label=self.train_dataset.id2label,
            label2id=self.train_dataset.label_map,
            ignore_mismatched_sizes=True,
        )
        self.augmented_metrics = Trainer(
            model=newly_trained_model,
            eval_dataset=self.eval_dataset,
            compute_metrics=compute_ner_metrics(
                list(self.train_dataset.label_map.keys())
            ),
        ).evaluate()

        self.next(self.compare)

    @step
    def compare(self):
        """Performs the comparison between the two trained models"""
        print("Metrics before augmentation:", self.metrics)
        print("Metrics after augmentation:", self.augmented_metrics)

        self.next(self.end)

    @step
    def end(self):
        """Ending step of the flow (required by Metaflow)"""
        logging.info(Warnings.W011(class_name=self.__class__))


if __name__ == "__main__":
    NEREnd2EndPipeline()
