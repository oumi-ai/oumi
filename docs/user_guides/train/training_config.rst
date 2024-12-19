Training Configuration
======================

.. autoclass:: oumi.core.configs.TrainingConfig
   :members: model, data, training, peft, fsdp
   :no-special-members:
   :no-index:

Model Parameters
----------------

.. autoclass:: oumi.core.configs.ModelParams
   :members:
   :exclude-members: to_lm_harness,__post_init__,__validate__
   :no-special-members:
   :no-index:

Data Parameters
---------------

.. autoclass:: oumi.core.configs.DataParams
   :members: train, test, validation
   :no-special-members:
   :no-index:

.. autoclass:: oumi.core.configs.DatasetSplitParams
   :members: datasets, collator_name, pack, stream, target_col, mixture_strategy, seed
   :no-special-members:
   :no-index:

.. autoclass:: oumi.core.configs.DatasetParams
   :members:
   :exclude-members: __post_init__
   :no-special-members:
   :no-index:

Training Parameters
-------------------

.. autoclass:: oumi.core.configs.TrainingParams
   :members:
   :exclude-members: to_hf, __post_init__, _get_hf_report_to
   :no-special-members:
   :no-index:

PEFT Parameters
---------------

.. autoclass:: oumi.core.configs.PeftParams
   :members:
   :no-special-members:
   :no-index:

FSDP Parameters
---------------

.. autoclass:: oumi.core.configs.FSDPParams
   :members:
   :no-special-members:
   :no-index:

Profiler Parameters
-------------------

.. autoclass:: oumi.core.configs.ProfilerParams
   :members:
   :no-special-members:
   :no-index:

Telemetry Parameters
--------------------

.. autoclass:: oumi.core.configs.TelemetryParams
   :members:
   :no-special-members:
   :no-index:
