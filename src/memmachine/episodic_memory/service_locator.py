from memmachine.common.configuration.episodic_config import EpisodicMemoryConf
from memmachine.common.resource_mgr import ResourceManager

from .episodic_memory import EpisodicMemoryParams
from .long_term_memory.long_term_memory import LongTermMemory
from .long_term_memory.service_locator import (
    long_term_memory_params_from_config,
)
from .short_term_memory.service_locator import (
    short_term_memory_params_from_config,
)
from .short_term_memory.short_term_memory import ShortTermMemory


async def epsiodic_memory_params_from_config(
    config: EpisodicMemoryConf,
    resource_manager: ResourceManager,
) -> EpisodicMemoryParams:
    long_term_memory: LongTermMemory | None = None
    if config.long_term_memory and config.long_term_memory_enabled:
        long_term_memory_params = long_term_memory_params_from_config(
            config.long_term_memory,
            resource_manager,
        )
        long_term_memory = LongTermMemory(long_term_memory_params)

    short_term_memory: ShortTermMemory | None = None
    if config.short_term_memory and config.short_term_memory_enabled:
        short_term_memory_params = short_term_memory_params_from_config(
            config.short_term_memory,
            resource_manager,
        )
        short_term_memory = await ShortTermMemory.create(short_term_memory_params)

    return EpisodicMemoryParams(
        session_key=config.session_key,
        metrics_factory=config.get_metrics_factory(),
        long_term_memory=long_term_memory,
        short_term_memory=short_term_memory,
        enabled=config.enabled,
    )
