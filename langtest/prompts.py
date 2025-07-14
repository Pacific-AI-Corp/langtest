from collections import defaultdict
from typing import Dict, List, Union

from pydantic.v1 import BaseModel, validator, Extra, ConfigDict


class MessageType(BaseModel):
    __field_order: List[str] = [
        "content",
        "context",
        "question",
        "original",
        "testcase",
        "options",
        "answer",
    ]

    model_config = ConfigDict(extra=Extra.allow)

    @validator("*", pre=True, allow_reuse=True)
    def add_field(cls, v, values, field, **kwargs):
        if "fields" not in values:
            values["fields"] = []
        values["fields"].append(field)
        return v

    @property
    def get_template(self):
        """Generate a template string based on the dynamic fields of the instance."""

        temp = []
        order_less = []

        sorted_fields = sorted(
            self.__dict__.keys(), key=lambda x: self.__field_order.index(x.lower())
        )

        for field in sorted_fields:
            if field in self.__field_order:
                temp.append(f"{field.title()}: {{{field}}}")
            else:
                order_less.append(f"{field.title()}: {{{field}}}")

        if order_less:
            temp.extend(order_less)
        return "\n" + "\n".join(temp)

    @property
    def get_example(self):
        """Generate an example string based on the dynamic fields of the instance."""
        # return {k: v for k, v in self.__dict__.items() if k != 'fields'}
        temp = {}
        for field in self.__field_order:
            if field in self.__dict__:
                temp[field] = self.__dict__[field]
        return temp

    @property
    def input_variables(self):
        temp = []
        for field in self.__field_order:
            if field in self.__dict__:
                temp.append(field)
        return temp


class Conversion(BaseModel):
    """Conversion model for the conversion of the input and output of the model."""

    user: MessageType
    ai: MessageType

    model_config = ConfigDict(extra=Extra.allow)

    @validator("*", pre=True, allow_reuse=True)
    def add_field(cls, v, values, field, **kwargs):
        if "fields" not in values:
            values["fields"] = []
        values["fields"].append(field)
        return v

    @property
    def get_examples(self):
        """Generate a list of examples based on the dynamic fields of the instance."""
        return {**self.user.get_example, **self.ai.get_example}

    @property
    def get_suffix_user(self):
        if self.user.get_template:
            return self.user.get_template


class PromptConfig(BaseModel):
    instructions: str
    prompt_type: str
    examples: Union[Conversion, List[Conversion]] = None

    @property
    def get_examples(self) -> List[dict]:
        """Generate a list of examples based on the dynamic fields of the instance."""
        if isinstance(self.examples, Conversion):
            return [self.examples.get_examples]
        elif isinstance(self.examples, list):
            return [example.get_examples for example in self.examples]

    @property
    def get_template(self):
        """Generate a template string based on the dynamic fields of the instance."""
        if isinstance(self.examples, Conversion):
            return self.examples.user.get_template
        elif isinstance(self.examples, list):
            return [
                ("human", self.examples[0].user.get_template),
                ("ai", self.examples[0].ai.get_template),
            ]
        # return self.examples.get_template

    @property
    def get_input_variables(self):
        """Generate a list of input variables based on the dynamic fields of the instance."""
        if isinstance(self.examples, Conversion):
            return self.examples.user.input_variables
        elif isinstance(self.examples, list):
            return self.examples[0].user.input_variables

    def prompt_style(self):
        """Generate a prompt based on the prompt type."""
        if self.prompt_type in ["chat", "instruct"]:
            from langchain.prompts import (
                ChatPromptTemplate,
                FewShotChatMessagePromptTemplate,
            )

            example_prompt = ChatPromptTemplate.from_messages(self.get_template)

            few_shot_prompt = FewShotChatMessagePromptTemplate(
                examples=self.get_examples,
                example_prompt=example_prompt,
            )

            final_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.instructions),
                    few_shot_prompt,
                    # ('human', conf),
                    self.get_template[0],
                ]
            )
            return final_prompt

        elif self.prompt_type == "completion":
            from langchain.prompts import FewShotPromptTemplate, PromptTemplate

            template = "".join(v for _, v in self.get_template)
            template = f"{template.replace('Answer:', '')}"
            examples = [v.get_examples for v in self.examples]
            suffix = self.examples[0].get_suffix_user

            example = PromptTemplate.from_template(template)

            final_prompt = FewShotPromptTemplate(
                examples=examples,
                example_prompt=example,
                input_variables=self.get_input_variables,
                suffix=suffix,
                prefix=self.instructions,
            )

            return final_prompt

    def get_prompt(self, hub=None):
        if hub in ("lm-studio", "transformers"):
            return self.lm_studio_prompt()
        return self.prompt_style()

    def get_shot_prompt(self):
        print(self.get_examples)
        return f"{len(self.get_examples)}-shot prompt"

    def lm_studio_prompt(self):
        messages = [
            {"role": "system", "content": self.instructions},
        ]

        for example in self.examples:
            temp_user = {}
            temp_ai = {}

            # user role
            temp_user["role"] = "user"
            temp_user["content"] = example.user.get_template.format(
                **example.user.get_example
            )

            # assistant role
            temp_ai["role"] = "assistant"
            temp_ai["content"] = (
                example.ai.get_template.format(**example.ai.get_example)
                .replace("Answer:", "")
                .strip()
                + "\n\n"
            )

            messages.append(temp_user)
            messages.append(temp_ai)
            # return messages
        return messages


class PromptManager:
    _instance = None
    prompt_configs: Dict[str, PromptConfig] = defaultdict(PromptConfig)
    _default_state = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.prompt_configs = defaultdict(PromptConfig)
        return cls._instance

    @classmethod
    def from_prompt_configs(cls, prompt_configs: dict):
        """Create a prompt manager from a dictionary of prompt configurations."""
        prompt_manager = cls()
        if set(["instructions", "prompt_type", "examples"]).issubset(
            set(prompt_configs.keys())
        ):
            prompt_manager.add_prompt("default", prompt_configs)
            return prompt_manager
        for name, prompt_config in prompt_configs.items():
            prompt_manager.add_prompt(name, prompt_config)

        if len(prompt_manager.prompt_configs) == 1:
            prompt_manager.default_state = list(prompt_manager.prompt_configs.keys())[0]
        return prompt_manager

    def add_prompt(self, name: str, prompt_config: dict):
        """Add a prompt template to the prompt manager."""
        prompt_config_o = PromptConfig(**prompt_config)
        self.prompt_configs[name] = prompt_config_o

    def get_prompt(self, name: str = None, hub: str = None):
        """Get a prompt template based on the name."""
        if name is None and self.default_state is None:
            return None
        if name is None:
            name = self.default_state
        if name in self.prompt_configs:
            prompt_template = self.prompt_configs[name].get_prompt(hub)
            return prompt_template

    @property
    def default_state(self):
        return self._default_state

    @default_state.setter
    def default_state(self, name: str):
        self._default_state = name

    @property
    def get_prompt_shot(self):
        return self.get_prompt().get_shot_prompt()

    def reset(self):
        """Reset the prompt manager to its initial state."""
        self.prompt_configs = defaultdict(PromptConfig)
        self._instance = None
        return self
