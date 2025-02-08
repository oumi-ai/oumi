# A tool to save Oumi Conversation-s from SFT datasets to a file.

import argparse

from oumi.utils.logging import update_logger_level


def main(args):
    """The script's entry point."""
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saves SFT conversations ")
    parser.add_argument(
        "-c", "--config", type=str, help="Path to the Oumi training configuration file."
    )
    parser.add_argument(
        "-d",
        "--dummy",
        action="store_true",
        help="Use a dummy dataset instead of an Oumi dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataloader_benchmark_results",
        help="Output directory for benchmark result plots",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Log level.",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=100,
        help="Maximum number of conversations to save.",
    )
    args = parser.parse_args()

    if not args.config and not args.dummy:
        raise ValueError("Either --config or --dummy must be provided")

    update_logger_level("oumi", level=args.log_level)

    main(args)
