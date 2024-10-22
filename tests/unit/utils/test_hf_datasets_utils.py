import tempfile
from pathlib import Path

import datasets

from oumi.utils.hf_datasets_utils import is_cached_to_disk_hf_dataset


def test_is_saved_to_disk_hf_dataset():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        ds = datasets.Dataset.from_dict(
            {"pokemon": ["bulbasaur", "squirtle"], "type": ["grass", "water"]}
        )
        ds_dir = Path(output_temp_dir) / "toy_dataset"
        assert not is_cached_to_disk_hf_dataset(ds_dir)

        ds_dir.mkdir(parents=True, exist_ok=True)
        assert not is_cached_to_disk_hf_dataset(ds_dir)

        ds.save_to_disk(ds_dir, num_shards=2)
        assert is_cached_to_disk_hf_dataset(ds_dir)

        for filename in ("dataset_info.json", "state.json"):
            sub_path: Path = Path(ds_dir) / filename
            assert sub_path.exists() and sub_path.is_file()
            sub_path.unlink()
            assert not is_cached_to_disk_hf_dataset(ds_dir)
