import logging

from oumi.utils.logging import get_logger, logger

TEST_LOGGER = get_logger("oumi.logging_test")


def _flush_logger(logger):
    for handler in logger.handlers:
        handler.flush()


def test_stdout_stderr_logging(caplog):
    with caplog.at_level(logging.WARNING):
        logger.info("apple")
        print("caplog.")
        logger.propagate = True
        _flush_logger(logger)
        assert caplog.text == "zzz"
    # print("hello")
    # assert out == "apple", f"err: {err}"
    # assert err == ""


def test_stdout_stderr_logging_capsys(capsys):
    logger.info("pear")
    # print("hello")
    out, err = capsys.readouterr()
    assert out == "pear", f"err: {err}"
    assert err == ""
