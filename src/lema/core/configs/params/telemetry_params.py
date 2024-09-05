from dataclasses import dataclass

from lema.core.configs.params.base_params import BaseParams


@dataclass
class TelemetryParams(BaseParams):
    #: Whether to save telemetry for all ranks.
    #: By default, only the main rank's telemetry stats are saved.
    save_telemetry_for_all_ranks: bool = False
