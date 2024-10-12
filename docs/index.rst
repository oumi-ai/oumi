Open Universal Machine Intelligence (Oumi) Documentation
========================================================
Welcome to the documentation for Open Universal Machine Intelligence (Oumi) !

Oumi is an open, collaborative modeling platform that allows you to build foundation models end-to-end including data curation/synthesis, pretraining, tuning, and evaluation.

.. .. toctree::

.. toctree::
   :maxdepth: 2
   :caption: Get started

   get_started/installation
   get_started/quickstart
   get_started/configuration

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/training
   user_guide/training_config
   user_guide/trainers
   user_guide/inference
   user_guide/inference_engines
   user_guide/evaluation
   user_guide/lm_harness
   user_guide/judge
   user_guide/oumi_judge
   user_guide/custom_judge
   user_guide/launch
   user_guide/distributed_training

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/notebooks

.. toctree::
   :maxdepth: 2
   :caption: Models

   models/recipes
   models/custom_models
   models/mlp_encoder
   models/cambrian

.. toctree::
   :maxdepth: 2
   :caption: Datasets

   datasets/local_datasets
   datasets/pretraining
   datasets/sft
   datasets/preference_tuning

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/custom_models
   advanced/custom_datasets
   advanced/quantization
   advanced/model_parallelism
   advanced/telemetry
   advanced/performance_optimization
   advance/new_multimodal_dataset

.. toctree::
   :maxdepth: 1
   :caption: FAQ

   faq/troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Python API

   api/index

.. toctree::
   :maxdepth: 2
   :caption: CLI

   cli/commands

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/code_of_conduct

.. toctree::
   :maxdepth: 1
   :caption: About

   about/changelog
   about/acknowledgement
   about/license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Getting Started
---------------

If you're new to Oumi, we recommend starting with the following sections:

1. :doc:`Introduction <links/usage>` - Learn about the core concepts and philosophy behind Oumi.
2. :doc:`Installation <links/dev_setup>` - Get Oumi up and running on your system.
3. `A tour of our library <https://github.com/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20A%20Tour.ipynb>`_ - Take a comprehensive tour of Oumi's features.

Tutorials
---------

Explore our :doc:`tutorials` for in-depth guides on using Oumi for various tasks:

- `Getting started with Finetuning <https://github.com/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Finetuning%20Tutorial.ipynb>`_
- `Using the Oumi job launcher <https://github.com/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Deploying%20a%20Job.ipynb>`_
- `Running jobs remotely <https://github.com/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Running%20Jobs%20Remotely.ipynb>`_
- `Launching jobs on custom clusters <https://github.com/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Launching%20Jobs%20on%20Custom%20Clusters.ipynb>`_
- `Working with Datasets in Oumi <https://github.com/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Datasets%20Tutorial.ipynb>`_

API Reference
-------------

For detailed information about the Oumi library, check out the :doc:`API Reference <apidoc/oumi>` section.

This includes comprehensive documentation for all modules, classes, and functions in the Oumi library.

Contributing
------------

We welcome contributions! See our :doc:`links/contributing` guide for information on how to get involved, including guidelines for code style, testing, and submitting pull requests.

Changelog
---------

.. note::
   This documentation is continuously updated. For the latest version and most recent changes, please visit our `GitHub repository <https://github.com/oumi-ai/oumi>`_.

Need Help?
----------

If you encounter any issues or have questions, please don't hesitate to:

1. Check our `FAQ section <https://github.com/oumi-ai/oumi/blob/main/FAQ.md>`_ for common questions and answers.
2. Open an issue on our `GitHub Issues page <https://github.com/oumi-ai/oumi/issues>`_ for bug reports or feature requests.
3. Join our `community <https://oumi.ai/community>`_ to discuss with other Oumi users and developers.
