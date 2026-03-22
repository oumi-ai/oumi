# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import safetensors
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
    tensor_map = {}
    for fname in tqdm(list(Path(".").glob("model-*-of-*.safetensors"))):
        with safetensors.safe_open(fname, framework="pt") as f:
            for k in f.keys():
                tensor_map.setdefault(k, dict(files=[], shape=None, dtype=None))
                tensor_map[k]["files"].append(fname)
                s = f.get_slice(k)
                tensor_map[k]["shape"] = s.get_shape()
                tensor_map[k]["dtype"] = s.get_dtype()
    assert {len(v["files"]) for v in tensor_map.values()} == {1}

    for k, v in tensor_map.items():
        v["file"] = Path(v["files"][0]).name
        del v["files"]

    Path("parameters_map.json").write_text(json.dumps(tensor_map))
