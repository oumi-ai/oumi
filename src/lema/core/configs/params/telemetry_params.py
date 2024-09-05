from dataclasses import dataclass, field
from typing import Optional

from lema.core.configs.params.base_params import BaseParams


@dataclass
class TelemetryParams(BaseParams):
    #: Directory where the telemetry data will be saved to.
    #: If not specified, then telemetry files will be written
    #: under `output_dir`.
    telemetry_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory where the telemetry data will be saved to. "
                "If not specified, then telemetry files will be written "
                "under `output_dir`."
            )
        },
    )

    #: Whether to save telemetry for all ranks.
    #: By default, only the main rank's telemetry stats are saved.
    save_telemetry_for_all_ranks: bool = False

    #: Whether to record GPU temperature.
    track_gpu_temperature: bool = False
