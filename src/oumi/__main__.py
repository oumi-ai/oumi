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
