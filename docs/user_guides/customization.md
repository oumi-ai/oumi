# Customizing Oumi

Often times when using a new framework, you may find that something you'd like to use is
missing. We always welcome [contributions](/development/contributing), but we also understand that sometimes it's
simpler to prototype changes locally. Whether you want to quickly experiment with new
features, test out different implementation approaches, or iterate rapidly on your ideas
without impacting the main codebase, Oumi provides a simple way to support
local customizations without any additional installations.

## The Oumi Registry

We support customization via the {py:class}`oumi.core.registry.Registry`.

You can easily register classes that are then loaded as if they're a native part of the
Oumi framework.

See the diagram below for how we load your custom code:

```{mermaid}
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#f5f5f5'}}}%%
graph LR
    %% Oumi Framework
    FR[Oumi] --> |Read OUMI_EXTRA_DEPS_FILE| RQ[requirements.txt]

    %% Load Custom Files
    RQ --> |Import File| CF1[Custom Class 1 File]
    RQ --> |Import File| CF2[Custom Class 2 File]
    RQ --> |Import File| CF3[        ...        ]

    %% Style for core workflow
    style FR fill:#1565c0,color:#ffffff
    style RQ fill:#1565c0,color:#ffffff
    style CF1 fill:#1565c0,color:#ffffff
    style CF1 fill:#1565c0,color:#ffffff
    style CF2 fill:#1565c0,color:#ffffff
    style CF3 fill:#1565c0,color:#ffffff
```

You can register your custom code in two simple steps:
1. Writing a custom Model, Dataset, Cloud, etc
2. Creating a `requirements.txt` file so your code is available via the CLI

## Writing Custom Classes

### Custom Models

You can easily customize models with oumi if something isn't supported out of the box.
We often hear requests for custom loss and custom model architectures: both are simple
to implement via a custom model.

Check out our guide for an in-depth explanation: {doc}`/resources/models/custom_models`

```{note}
Don't forget to decorate your class with

`@registry.register(..., registry.RegistryType.MODEL)`!
```

### Custom Datasets

Custom datasets are a great way to handle unique dataset formats that Oumi may not yet
support.

See the following snippets for examples of custom datasets:
- [Custom SFT Dataset](/resources/datasets/sft_datasets.md#adding-a-new-sft-dataset)
- [Custom Pre-training Dataset](/resources/datasets/pretraining_datasets.md#adding-a-new-pre-training-dataset)
- [Custom Preference Tuning Dataset](/resources/datasets/preference_datasets.md#creating-custom-preference-dataset)
- [Custom Vision-Language Dataset](/resources/datasets/vl_sft_datasets.md#adding-a-new-vl-sft-dataset)
- [Custom Numpy Dataset](sample-custom-numpy-dataset)

```{note}
Don't forget to decorate your class with `@register_dataset(...)`!
```

### Custom Clouds/Clusters

Adding a custom cloud is perfect for handling local clusters not hosted by major cloud
providers.

For example, we wrote a custom cloud for the Polaris platform. Our research team used
this cloud to schedule jobs seamlessly on a remote super computer.

Take a look at our [custom cluster tutorial here](/user_guides/launch/custom_cluster).

```{note}
Don't forget to decorate your class with `@register_cloud_builder(...)`!
```

### Custom Judge Configs

For quick reference, you can register custom judge configs

You can find [examples of custom judge configs here](https://github.com/oumi-ai/oumi/blob/main/src/oumi/judges/judge_court.py).

```{note}
Don't forget to decorate your function with `@register_judge(...)`!
```


## Enable Your Classes for the CLI

If you're using Oumi as a python library, your custom classes will work out of the box!
However, to use your custom classes from the Oumi CLI, you need to tell Oumi which files
to load when initializing our Registry.

To do this, you must first create a `requirements.txt` file. This file has a simple
structure: each line must be an absolute filepath to the file with your custom class /
function (that you specified with the `@register...` decorator).

For example, if you created two custom classes in files `/path/to/custom_cloud.py` and
`/another/path/to/custom_model.py`, your `requirements.txt` files should look like:

```
/path/to/custom_cloud.py
/another/path/to/custom_model.py
```

With your `requirements.txt` file created, you simply need to set the
`OUMI_EXTRA_DEPS_FILE` environment variable to the location of your file, and you're good to go!

``` {code-block} shell
export OUMI_EXTRA_DEPS_FILE=/another/path/requirements.txt
```

## See Also

- {py:class}`oumi.core.models.BaseModel` - Base class for all Oumi models
- {py:class}`oumi.core.registry.Registry` - Model registration system
- {py:class}`oumi.core.configs.params.model_params.ModelParams` - Base parameters class for models
- {gh}`➿ Training CNN on Custom Dataset <notebooks/Oumi - Training CNN on Custom Dataset.ipynb>` - Sample Jupyter notebook using {py:class}`oumi.models.CNNClassifier` and [Custom Numpy Dataset](sample-custom-numpy-dataset).
