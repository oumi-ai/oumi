Oumi Documentation
==================
Welcome to the documentation for Oumi!

Oumi is an open, collaborative modeling platform that allows you to build foundation models end-to-end including data curation/synthesis, pretraining, tuning, and evaluation.


Getting Started
---------------

If you're new to Oumi, we recommend starting with the following sections:

1. :doc:`Introduction <get_started/quickstart>` - Learn about the core concepts and philosophy behind Oumi.
2. :doc:`Installation <get_started/installation>` - Get Oumi up and running on your system.
3. :doc:`A tour of our library <get_started/tour>`  - Take a comprehensive tour of Oumi's features.

Tutorials
---------

Explore our :doc:`tutorials <get_started/tutorials>` for in-depth guides on using Oumi for various tasks:

- :doc:`Getting started with Finetuning <user_guides/train/finetuning>`
- :doc:`Using the Oumi job launcher <user_guides/launch/deploy>`
- :doc:`Running jobs remotely <user_guides/launch/remote>`
- :doc:`Launching jobs on custom clusters <advanced/custom_cluster>`
- :doc:`Working with Datasets in Oumi <advanced/custom_datasets>`

API Reference
-------------

For detailed information about the Oumi library, check out the :doc:`API Reference <api/oumi>` section.

This includes comprehensive documentation for all modules, classes, and functions in the Oumi library.

Contributing
------------

We welcome contributions! See our :doc:`development/contributing` guide for information on how to get involved, including guidelines for code style, testing, and submitting pull requests.

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

Overview
--------
.. .. toctree::

.. toctree::
   :maxdepth: 2
   :caption: Get started

   get_started/installation
   get_started/quickstart
   get_started/configuration
   get_started/tutorials
   get_started/tour

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guides/train/train
   user_guides/infer/infer
   user_guides/evaluate/evaluate
   user_guides/judge/judge
   user_guides/launch/launch

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/customization
   advanced/quantization
   advanced/performance_optimization
   advanced/distributed_training

.. toctree::
   :maxdepth: 2
   :caption: Models

   models/recipes
   models/cambrian

.. toctree::
   :maxdepth: 2
   :caption: Datasets

   datasets/local_datasets
   datasets/pretraining
   datasets/sft
   datasets/preference_tuning
   datasets/vl_sft

.. toctree::
   :maxdepth: 1
   :caption: FAQ

   faq/troubleshooting
   faq/oom
   faq/gpu_sizing

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   Python API <api/oumi>
   CLI Reference <cli/commands>

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/dev_setup
   development/contributing
   development/code_of_conduct
   development/style_guide
   development/git_workflow

.. toctree::
   :maxdepth: 1
   :caption: About

   about/changelog
   about/acknowledgements
   about/license
   about/citations

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
