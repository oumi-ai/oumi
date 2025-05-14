# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Oumi (Open Universal Machine Intelligence)."""

from oumi.cli.main import run

if __name__ == "__main__":
    import sys

    # Per https://docs.python.org/3/library/sys_path_init.html , the first entry in
    # sys.path is the directory containing the input script.
    # This means `python ./src/oumi` will result in `import datasets` resolving to
    # `oumi.datasets` instead of the installed `datasets` package.
    # Moving the first entry of sys.path to the end will ensure that the installed
    # packages are found first.
    if len(sys.path) > 1:
        sys.path = sys.path[1:] + sys.path[:1]
    run()
