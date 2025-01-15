"""Oumi (Open Universal Machine Intelligence)."""

from oumi.cli.main import run

if __name__ == "__main__":
    import sys

    # move the current working directory to the end position
    sys.path = sys.path[1:] + sys.path[:1]
    run()
