import os
from glob import glob

from collections import OrderedDict
from tqdm import tqdm

import torch


def _numel(shape):
    numel = 1
    for d in shape:
        numel *= d
    return numel


def _consolidate_param(state_dict_list, shard_metadata, name, prefix, suffix):
    p_shard_list = []
    for state_dict in state_dict_list:
        p_shard = state_dict[name]
        p_shard_list.append(p_shard)

        # if prefix not in shard_metadata["shard_info"]:
        #   prefix = prefix.replace('_fpw_module.', '')
        p_info = shard_metadata["shard_info"][prefix][suffix]
        orig_name = p_info["_orig_name"]
        orig_size = p_info["_orig_size"]

    full_param = torch.cat(p_shard_list, dim=0)
    if full_param.dim() == 1:
        # it's a flattened tensor as in the (usual) case with `shard_param_on_dim_0=False`
        full_param = full_param[:_numel(orig_size)].view(*orig_size)
    else:
        # handle those FSDP models trained with `shard_param_on_dim_0=True`
        full_param = full_param[:orig_size[0]]

    full_name = orig_name
    if prefix != "":
        full_name = prefix + "." + orig_name

    return full_param, full_name


def _unflatten_param(p, metadata, prefix):
    param_names, param_shapes, param_numels = metadata
    full_params = [
        t.view(s) for (t, s) in zip(p.split(param_numels), param_shapes)
    ]
    full_names = param_names
    if prefix != "":
        full_names = [prefix + "." + n for n in full_names]
    return full_params, full_names


def consolidate_sharded_state_dicts(state_dict_list, shard_metadata):
    """
    Consolidate the sharded FSDP model state dicts.

    Args:
        state_dict_list (OrderedDict):
            a list of ``model.state_dict()`` obtained from the FSDP model of
            each rank, **sorted in ascending order by their ranks**
        shard_metadata (dict):
            ``model.get_shard_metadata()`` from an FSDP model of any rank

    Returns:
        full_state_dict: the consolidated model state dict
    """
    assert len(state_dict_list) == shard_metadata["world_size"]
    full_state_dict = OrderedDict()
    buffer_info = shard_metadata.get("buffer_info", {})

    unsharded_flag = False

    # consolidate the sharded parameters
    for name, p in state_dict_list[0].items():
        if name in buffer_info:  # cast buffer back to its original dtype
            p = p.to(buffer_info[name]["_orig_dtype"])

        is_sharded = False
        name_splits = name.split(".")
        for idx, sep in enumerate(name_splits):
            if sep.startswith("_fsdp_shard"):
                is_sharded = True
                prefix = ".".join(name_splits[:idx])
                suffix = ".".join(name_splits[idx:])
                break

        if is_sharded:
            full_param, full_name = _consolidate_param(state_dict_list,
                                                       shard_metadata, name, prefix,
                                                       suffix)
        else:
            print(name)
            unsharded_flag = True
            # unsharded buffers (we'll just use rank 0's state dict for buffers)
            full_param, full_name = p, name
        full_state_dict[full_name] = full_param

    for state_dict in state_dict_list:
        if unsharded_flag:
            break
        for name, p in state_dict.items():
            if name in buffer_info:  # cast buffer back to its original dtype
                p = p.to(buffer_info[name]["_orig_dtype"])

            is_sharded = False
            name_splits = name.split(".")
            for idx, sep in enumerate(name_splits):
                if sep.startswith("_fsdp_shard"):
                    is_sharded = True
                    prefix = ".".join(name_splits[:idx])
                    suffix = ".".join(name_splits[idx:])
                    break
            if is_sharded:
                continue
            else:
                unsharded_flag = True
                # unsharded buffers (we'll just use rank 0's state dict for buffers)
                full_param, full_name = p, name
            full_state_dict[full_name] = full_param

    # unflatten the parameters
    flatten_info = shard_metadata["flatten_info"]
    for name in list(full_state_dict):
        if "_fsdp_wrapped_module.flat_param_" in name:
            p = full_state_dict.pop(name)
            metadata = flatten_info[name]
            prefix = ".".join(name.split(".")[:-1])
            full_params, full_names = _unflatten_param(p, metadata, prefix)
            for fp, fn in zip(full_params, full_names):
                full_state_dict[fn] = fp

    full_state_dict = OrderedDict(
        (k.replace("_fsdp_wrapped_module.", "").replace("_fpw_module.", ""), v)
        for k, v in full_state_dict.items())

    return full_state_dict


def consolidate_sharded_model_checkpoints(ckpt_prefix,
                                          ckpt_suffix="*.pth",
                                          save_path="",
                                          save_model=True):
    """
    Consolidate the sharded FSDP checkpoints into a single model checkpoint.

    Args:
        ckpt_prefix (str):
            prefix to FSDP checkpoint files from all ranks
        ckpt_suffix (str, Optional):
            suffix to FSDP checkpoint files from all ranks. Files matching the
            pattern ``ckpt_prefix + ckpt_suffix`` will be loaded. The each
            checkpoint file is assumed to be a dict with a "model" key
            containing the FSDP model's ``model.state_dict()`` and a
            "shard_metadata" key containing the FSDP model's
            ``model.get_shard_metadata()``.
        save_path (str, Optional):
            the save path to the consolidated model checkpoint file (if
            ``save_model`` is ``True``). The checkpoint file is a dict with a
            "model" key containing the consolidated model state dict.
        save_model (str, Optional):
            if ``True``, the consolidated model checkpoint will be saved to
            ``save_path`` (or ``ckpt_prefix + "_consolidated.pth"`` if
            ``save_path`` is empty).

    Returns:
        full_state_dict: the consolidated model state dict
        actual_save_path: the path to the consolidated model checkpoint file
            (``None`` if ``save_model`` is ``False``)
    """
    ckpt_path_pattern = ckpt_prefix + ckpt_suffix
    ckpt_paths = glob(ckpt_path_pattern)
    assert len(
        ckpt_paths) > 0, f"Cannot find any files matching {ckpt_path_pattern}."
    print(f"found {len(ckpt_paths)} checkpoint files in {ckpt_path_pattern}")
    checkpoints_and_paths = []
    for path in tqdm(ckpt_paths, desc="loading checkpoints"):
        ckpt = torch.load(path, map_location="cpu")
        checkpoints_and_paths.append((ckpt, path))
    checkpoints_and_paths.sort(key=lambda c: c[0]["shard_metadata"]["rank"])
    checkpoints = [c[0] for c in checkpoints_and_paths]
    for rank, (ckpt, path) in enumerate(checkpoints_and_paths):
        assert ckpt["shard_metadata"]["world_size"] == len(checkpoints), (
            f'Expecting {ckpt["shard_metadata"]["world_size"]} files '
            f"(based on metadata in {path}) but got {len(checkpoints)} files. "
            f"Please check if you have missing or unexpected files in {ckpt_path_pattern}."
        )
        assert ckpt["shard_metadata"]["rank"] == rank, (
            f'Expecting rank {ckpt["shard_metadata"]["rank"]} for {path} but it is '
            f"ranked {rank} (out of {len(checkpoints)} files). "
            f"Please check if you have missing or unexpected files in {ckpt_path_pattern}."
        )

    state_dict_list = [ckpt["model"] for ckpt in checkpoints]
    shard_metadata = checkpoints[0]["shard_metadata"]
    full_state_dict = consolidate_sharded_state_dicts(state_dict_list,
                                                      shard_metadata)

    actual_save_path = None
    if save_model:
        actual_save_path = save_path if save_path else ckpt_prefix + "_consolidated.pth"
        torch.save({"model": full_state_dict}, actual_save_path)
        print(f"saved consolidated model to {actual_save_path}")

    return full_state_dict, actual_save_path


def main(ckpt_path, ckpt_prefix, ckpt_suffix, save_path, skip_existing=False):

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist")
    elif not os.path.isdir(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint path {ckpt_path} is not a directory")
    elif os.path.exists(save_path):
        if skip_existing:
            print(f"Save path {save_path} already exists. Passed `skip_existing=True`, skipping...")
            return
        print(f"Save path {save_path} already exists. Passed `skip_existing=False`, overwriting...")

    print(f"""Consolidating checkpoints:
    Path: {ckpt_path}
    Prefix: {ckpt_prefix}
    Suffix: {ckpt_suffix}
    Saving to: {save_path}
    """)

    ckpt_prefix = os.path.join(ckpt_path, ckpt_prefix)

    state_dict, _ = consolidate_sharded_model_checkpoints(
        ckpt_prefix=ckpt_prefix,
        ckpt_suffix=ckpt_suffix,
        save_model=False
    )

    print(f"Successfully consolidated checkpoints. Saving to {save_path}")
    torch.save(state_dict, save_path)

    print(f"Saved consolidated model to {save_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--ckpt_prefix', type=str, default='model_ckpt')
    parser.add_argument('--ckpt_suffix', type=str, default='_rank-*-of-*.pth')
    parser.add_argument('--save_filename', type=str, default='full_model.pth')
    parser.add_argument('--skip_existing', action='store_true', default=False, help='Skip if the save path already exists')
    args = parser.parse_args()

    save_path = os.path.join(args.ckpt_path, args.save_filename)

    main(args.ckpt_path, args.ckpt_prefix, args.ckpt_suffix, save_path, args.skip_existing)
