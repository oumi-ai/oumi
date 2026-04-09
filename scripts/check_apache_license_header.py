#!/usr/bin/env python3
# Copyright 2026 - Oumi
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

"""Pre-commit hook to check that Python source files have an Apache license header."""

import sys

# Year-agnostic: we just check for the Apache License text, not a specific copyright year
_APACHE_LICENSE_SNIPPET = "Licensed under the Apache License, Version 2.0"


def main() -> int:
    failed = []
    for filepath in sys.argv[1:]:
        try:
            with open(filepath) as f:
                # Read just the first 1024 bytes — the header should be near the top
                head = f.read(1024)
        except (OSError, UnicodeDecodeError):
            continue

        if _APACHE_LICENSE_SNIPPET not in head:
            failed.append(filepath)

    if failed:
        print("The following files are missing an Apache 2.0 license header:")
        for f in failed:
            print(f"  {f}")
        print(
            "\nPlease add the following header to the top of each file:\n"
            "\n"
            "# Copyright <YEAR> - Oumi\n"
            "#\n"
            '# Licensed under the Apache License, Version 2.0 (the "License");\n'
            "# you may not use this file except in compliance with the License.\n"
            "# You may obtain a copy of the License at\n"
            "#\n"
            "#     http://www.apache.org/licenses/LICENSE-2.0\n"
            "#\n"
            "# Unless required by applicable law or agreed to in writing, software\n"
            '# distributed under the License is distributed on an "AS IS" BASIS,\n'
            "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
            "# See the License for the specific language governing permissions and\n"
            "# limitations under the License.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
