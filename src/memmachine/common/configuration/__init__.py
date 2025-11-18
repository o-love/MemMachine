import os.path
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from memmachine.common.configuration.embedder_conf import EmbedderConf
from memmachine.common.configuration.log_conf import LogConf
from memmachine.common.configuration.model_conf import LanguageModelConf
from memmachine.common.configuration.reranker_conf import RerankerConf
from memmachine.common.configuration.storage_conf import StorageConf
from memmachine.semantic_memory.semantic_model import SemanticCategory
from memmachine.semantic_memory.semantic_session_resource import IsolationType
from memmachine.server.prompt.default_prompts import PREDEFINED_SEMANTIC_CATEGORIES

from .episodic_config import EpisodicMemoryConfPartial


class SessionDBConf(BaseModel):
    uri: str = Field(
        default="sqlitetest.db",
        description="Database URI",
    )
    storage_id: str = Field(
        default="",
        description="The storage ID to use for session DB",
    )


class SemanticMemoryConf(BaseModel):
    database: str = Field(
        ...,
        description="The database to use for semantic memory",
    )
    llm_model: str = Field(
        ...,
        description="The default language model to use for semantic memory",
    )
    embedding_model: str = Field(
        ...,
        description="The embedding model to use for semantic memory",
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


class PromptConf(BaseModel):
    profile: list[str] = Field(
        default=["profile_prompt", "writing_assistant_prompt"],
        description="The default prompts to use for semantic user memory",
    )
    role: list[str] = Field(
        default=[],
        description="The default prompts to use for semantic role memory",
    )
    session: list[str] = Field(
        default=[],
        description="The default prompts to use for semantic session memory",
    )
    episode_summary_system_prompt_path: str = Field(
        default="",
        description="The prompt template to use for episode summary generation - system part",
    )
    episode_summary_user_prompt_path: str = Field(
        default="",
        description="The prompt template to use for episode summary generation - user part",
    )

    @classmethod
    def prompt_exists(cls, prompt_name: str) -> bool:
        return prompt_name in PREDEFINED_SEMANTIC_CATEGORIES

    @field_validator("profile", "session", "role", check_fields=True)
    @classmethod
    def validate_profile(cls, v: list[str]) -> list[str]:
        for prompt_name in v:
            if not cls.prompt_exists(prompt_name):
                raise ValueError(f"Prompt {prompt_name} does not exist")
        return v

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
    def default_semantic_categories(
        self,
    ) -> dict[IsolationType, list[SemanticCategory]]:
        semantic_categories = PREDEFINED_SEMANTIC_CATEGORIES

        return {
            IsolationType.SESSION: [
                semantic_categories[s_name] for s_name in self.session
            ],
            IsolationType.ROLE: [semantic_categories[s_name] for s_name in self.role],
            IsolationType.USER: [
                semantic_categories[s_name] for s_name in self.profile
            ],
        }


class Configuration(EpisodicMemoryConfPartial):
    logging: LogConf
    sessiondb: SessionDBConf
    model: LanguageModelConf
    storage: StorageConf
    embedder: EmbedderConf
    reranker: RerankerConf
    prompt: PromptConf = PromptConf()
    semantic_memory: SemanticMemoryConf

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
