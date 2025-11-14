import os.path
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Self

import yaml
from pydantic import BaseModel, Field, model_validator, root_validator

from .episodic_config import EpisodicMemoryConfPartial
from ...common.configuration.embedder_conf import EmbedderConf
from ...common.configuration.log_conf import LogConf
from ...common.configuration.model_conf import LanguageModelConf
from ...common.configuration.reranker_conf import RerankerConf
from ...common.configuration.storage_conf import StorageConf


class SessionDBConf(BaseModel):
    uri: str = Field(
        default="sqlitetest.db",
        description="Database URI",
    )
    storage_id: str = Field(
        default="",
        description="The storage ID to use for session DB",
    )


class ProfileMemoryConf(BaseModel):
    llm_model: str = Field(
        ...,
        description="The language model to use for profile memory",
    )
    embedding_model: str = Field(
        ...,
        description="The embedding model to use for profile memory",
    )
    database: str = Field(
        ...,
        description="The database to use for profile memory",
    )
    prompt: str = Field(
        ...,
        description="The prompt template to use for profile memory",
    )


def _read_txt(filename: str) -> str:
    """
    Reads a text file and returns its contents as a string.

    Behavior:
      - If `filename` is an absolute path, read it directly.
      - If `filename` is a relative path, read it from the current working directory.

    Args:
        filename (str): File name or absolute path. Optional.

    Returns:
        str: The file's content as a string.
    """
    path = Path(filename)
    if not path.is_absolute():
        path = Path.cwd() / path

    with path.open("r", encoding="utf-8") as f:
        return f.read()


class ProfilePrompt(BaseModel):
    update_prompt: str = Field(
        ...,
        description="The prompt template to use for profile update",
    )
    consolidation_prompt: str = Field(
        ...,
        description="The prompt template to use for profile consolidation",
    )

    @staticmethod
    def load_from_module(prompt_module: ModuleType):
        update_prompt = getattr(prompt_module, "UPDATE_PROMPT", "")
        consolidation_prompt = getattr(prompt_module, "CONSOLIDATION_PROMPT", "")

        return ProfilePrompt(
            update_prompt=update_prompt,
            consolidation_prompt=consolidation_prompt,
        )


class PromptConf(BaseModel):
    profile: str = Field(
        default="profile_prompt",
        description="The prompt template to use for profile memory",
    )
    episode_summary_system_prompt_path: str = Field(
        default="",
        description="The prompt template to use for episode summary generation - system part",
    )
    episode_summary_user_prompt_path: str = Field(
        default="",
        description="The prompt template to use for episode summary generation - user part",
    )

    @property
    def episode_summary_system_prompt(self) -> str:
        file_path = self.episode_summary_system_prompt_path
        if not file_path:
            txt = "dft_episode_summary_system_prompt.txt"
            file_path = os.path.join(Path(__file__).parent, txt)
        return _read_txt(file_path)

    @property
    def episode_summary_user_prompt(self) -> str:
        file_path = self.episode_summary_user_prompt_path
        if not file_path:
            txt = "dft_episode_summary_user_prompt.txt"
            file_path = os.path.join(Path(__file__).parent, txt)
        return _read_txt(file_path)

    @property
    def profile_prompt(self) -> "ProfilePrompt":
        prompt_package = "memmachine.server.prompt"
        module_name = f"{prompt_package}.{self.profile}"

        try:
            prompt_module: ModuleType = import_module(module_name)
        except ModuleNotFoundError as e:
            raise ImportError(
                f"Prompt profile '{self.profile}' not found in package '{prompt_package}'."
            ) from e

        try:
            prompt = ProfilePrompt.load_from_module(prompt_module)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load prompt from module '{module_name}': {e}"
            ) from e
        return prompt


class Configuration(EpisodicMemoryConfPartial):
    logging: LogConf
    sessiondb: SessionDBConf
    model: LanguageModelConf
    storage: StorageConf
    profile_memory: ProfileMemoryConf
    embedder: EmbedderConf
    reranker: RerankerConf
    prompt: PromptConf

    def __init__(self, **data):
        data = data.copy()  # avoid mutating caller's dict
        super().__init__(**data)
        self.model = LanguageModelConf.parse_language_model_conf(data)
        self.storage = StorageConf.parse_storage_conf(data)
        self.embedder = EmbedderConf.parse_embedder_conf(data)
        self.reranker = RerankerConf.parse_reranker_conf(data)


def load_config_yml_file(config_file: str) -> Configuration:
    try:
        yaml_config = yaml.safe_load(open(config_file, encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_file} not found")
    except yaml.YAMLError:
        raise ValueError(f"Config file {config_file} is not valid YAML")
    except Exception as e:
        raise e

    def config_to_lowercase(data: Any) -> Any:
        """Recursively converts all dictionary keys in a nested structure
        to lowercase."""
        if isinstance(data, dict):
            return {k.lower(): config_to_lowercase(v) for k, v in data.items()}
        if isinstance(data, list):
            return [config_to_lowercase(i) for i in data]
        return data

    yaml_config = config_to_lowercase(yaml_config)
    return Configuration(**yaml_config)
