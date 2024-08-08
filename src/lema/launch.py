import argparse

import lema.launcher as launcher
from lema.utils.logging import logger

_START_TIME = -1.0


def parse_cli():
    """Parses command line arguments and returns the configuration filename."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default=None, help="Path to the configuration file"
    )

    parser.add_argument(
        "--cluster", default=None, help="The cluster name to use for the job"
    )

    args, unknown = parser.parse_known_args()
    return args.config, args.cluster, unknown


def main() -> None:
    """Main entry point for launching jobs on LeMa.

    Arguments are fetched from the following sources, ordered by
    decreasing priority:
    1. [Optional] Arguments provided as CLI arguments, in dotfile format
    2. [Optional] Arguments provided in a yaml config file
    3. Default arguments values defined in the data class
    """
    # Load configuration
    config_path, _cluster, arg_list = parse_cli()

    config: launcher.JobConfig = launcher.JobConfig.from_yaml_and_arg_list(
        config_path, arg_list, logger=logger
    )
    config.validate()

    # Start the job
    running_cluster, job_status = launcher.up(config, _cluster)


if __name__ == "__main__":
    main()
