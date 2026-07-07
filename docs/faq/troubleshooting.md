# Troubleshooting

## Getting Help

Running into a problem? Check in with us on [Discord](https://discord.gg/oumi)-we're happy to help!

Still can't find a solution? Let us know by filing a new [GitHub Issue](https://github.com/oumi-ai/oumi/issues).

## Common Issues

### Installing on Windows

If you'd like to use Oumi OSS on Windows, we strongly suggest using
[Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install).

Installing natively on Windows outside of a WSL environment can lead to installation errors such as:

```shell
ERROR: Could not find a version that satisfies the requirement ... (from versions: none)
```

or, when long path support is disabled, an error while installing the `lm_eval` dependency:

```text
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory:
'...\\lm_eval\\tasks\\arabic_leaderboard_complete\\...'
HINT: This error might have occurred since this system does not have Windows Long Path support enabled.
```

or runtime errors like:

```shell
ModuleNotFoundError: No module named 'resource'
```

or a `UnicodeDecodeError` during training, when a dependency reads a UTF-8 file under the
Windows default code page (`cp1252`):

```text
UnicodeDecodeError: 'charmap' codec can't decode byte 0x81 in position 932: character maps to <undefined>
```

#### Running natively on Windows (not recommended)

WSL is the recommended path and avoids all of the issues below. If you cannot use WSL (for
example, on a restricted machine), the following steps address the most common native-Windows
failures. We also recommend the [python.org](https://www.python.org/downloads/windows/) build of
Python rather than the Microsoft Store build, which is sandboxed and does not add package scripts
to your `PATH` by default.

**Enable long path support.** `lm_eval` ships task files whose paths exceed the Windows
260-character `MAX_PATH` limit, which can cause the `OSError` shown above during install. Enable
long paths in an **Administrator** PowerShell, then restart your machine:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

If you also clone the repository, enable long paths in Git so the clone doesn't hit the same limit:

```shell
git config --global core.longpaths true
```

**Enable UTF-8 mode.** Python on Windows defaults to the system code page rather than UTF-8, which
can break both log output (emoji in Oumi's logs) and reads of bundled UTF-8 files (such as `trl`'s
chat templates), producing the `UnicodeDecodeError` shown above. Set UTF-8 mode before running
Oumi:

```shell
export PYTHONUTF8=1
```

In PowerShell, use `$env:PYTHONUTF8 = "1"` instead. To make this permanent in Git Bash, add the
`export` line to `~/.bashrc`.

**Run recipe commands from a cloned repository.** The CLI examples reference config files such as
`configs/recipes/smollm/sft/135m/quickstart_train.yaml` by relative path. These live in the GitHub
repository, not in the installed `oumi` package, so clone the repo and run the commands from inside
it:

```shell
git clone https://github.com/oumi-ai/oumi.git
cd oumi
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml
```

### Installing on macOS

Oumi OSS only supports Apple Silicon Macs, not Intel Macs. This is because PyTorch dropped support for the latter. Installing on Intel Macs can lead to errors like:

```text
Using Python 3.11.11 environment at: /Users/moonshine/miniconda3/envs/oumi
  × No solution found when resolving dependencies:
  ╰─▶ Because only the following versions of torch are available:
          torch<=2.5.0
          torch==2.5.1
          torch>2.6.0
      and torch>=2.5.0,<=2.5.1 has no wheels with a matching platform tag
      (e.g., `macosx_10_16_x86_64`), we can conclude that torch>=2.5.0,<=2.5.1
      cannot be used.
      And because oumi==0.1.dev1313+g33c1fa9 depends on torch>=2.5.0,<2.6.0,
      we can conclude that oumi==0.1.dev1313+g33c1fa9 cannot be used.
      And because only oumi[dev]==0.1.dev1313+g33c1fa9 is available and
      you require oumi[dev], we can conclude that your requirements are
      unsatisfiable.

      hint: Wheels are available for `torch` (v2.5.1) on the following
      platforms: `manylinux1_x86_64`, `manylinux2014_aarch64`,
      `macosx_11_0_arm64`, `win_amd64`
```

### Pre-commit hook errors with VS Code

- When committing changes, you may encounter an error with pre-commit hooks related to missing imports.
- To fix this, make sure to start your vscode instance after activating your conda environment.

     ```shell
     conda activate oumi
     code .  # inside the Oumi OSS directory
     ```

### Out of Memory (OOM)

See {doc}`oom` for more information.

### Launching Remote Jobs Fail due to File Mounts

When running a remote job using a command like:

```shell
oumi launch up -c your/config/file.yaml
```

It's common to see failures with errors like:

```
ValueError: File mount source '~/.netrc' does not exist locally. To fix: check if it exists, and correct the path.
```

These errors indicate that your JobConfig contains a reference to a file that does not exist on your local machine. You can remove the offending line from your yaml file's {py:attr}`~oumi.core.configs.JobConfig.file_mounts` to resolve the error if it's unneeded. Otherwise, here's how to resolve the error for specific files often mounted by Oumi OSS jobs:

- `~/.netrc`: This file contains your Weights and Biases (WandB) credentials, which are needed to log your run's metrics to WandB.
  - To fix, follow {ref}`these instructions <optional-set-up-weights-and-biases>`
  - If you don't require WandB logging, disable either TrainingParams.{py:attr}`~oumi.core.configs.TrainingParams.enable_wandb` or EvaluationConfig.{py:attr}`~oumi.core.configs.EvaluationConfig.enable_wandb`, for training and evaluation jobs respectively. This is needed in addition to removing the file mount to prevent an error.
- `~/.cache/huggingface/token`: This file contains your Huggingface credentials, which are needed to access gated datasets/models on HuggingFace Hub.
  - To fix, follow {ref}`these instructions <optional-set-up-huggingface>`

### Training Stability & NaN Loss

- Lower the initial learning rate
- Enable gradient clipping (or, apply further clipping if already enabled)
- Add learning rate warmup

```python
config = TrainingConfig(
    training=TrainingParams(
        max_grad_norm=0.5,
        optimizer="adamw_torch_fused",
        warmup_ratio=0.01,
        lr_scheduler_type="cosine",
        learning_rate=1e-5,
    ),
)
```

### Inference Issues

- Verify {doc}`model </resources/models/models>` and [tokenizer](/resources/models/models.md#tokenizer-integration) paths are correct
- Ensure [input data](/user_guides/infer/infer.md#input-data) is correctly formatted and preprocessed
- Validate that the {doc}`inference engine </user_guides/infer/inference_engines>` is compatible with your model type

### Quantization-Specific Issues

Decreased model performance:

- Increase `lora_r` and `lora_alpha` parameters in {py:obj}`oumi.core.configs.PeftParams`
