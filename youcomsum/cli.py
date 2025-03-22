"""Module for command line interface."""

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import NoReturn, Optional

from .core import YouComSum
from .info import __issues__, __summary__, __version__

LOG_LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
logger = logging.getLogger(__name__)


class HelpArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        """Handle error from argparse.ArgumentParser."""
        self.print_help(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def get_parser() -> argparse.ArgumentParser:
    """Prepare ArgumentParser."""
    parser = HelpArgumentParser(
        prog="youcomsum",
        description=__summary__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s, version {__version__}",
    )
    # Verbose
    parser.add_argument(
        "-v",
        "--verbose",
        help="verbose mode, enable INFO and DEBUG messages.",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-l",
        "--lang",
        help="language for output text.",
        required=False,
    )
    parser.add_argument("url")
    return parser


def setup_logging(verbose: Optional[bool] = None) -> None:
    """Do setup logging."""
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    )
    if verbose:
        logging.getLogger("youcomsum").setLevel(logging.DEBUG)


def entrypoint(argv: Optional[Sequence[str]] = None) -> None:
    """Entrypoint for command line interface."""
    try:
        parser = get_parser()
        args = parser.parse_args(argv)
        setup_logging(args.verbose)
        youcomsum = YouComSum()
        result = youcomsum.summarize(
            video=args.url,
            lang=args.lang,
        )
        print(result)  # noqa: T201
    except Exception as err:  # NoQA: BLE001
        logger.critical("Unexpected error", exc_info=err)
        logger.critical("Please, report this error to %s.", __issues__)
        sys.exit(1)
