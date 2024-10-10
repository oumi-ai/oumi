# import pytest

# from oumi.utils.logging import configure_dependency_warnings, get_logger


# @pytest.fixture(scope="session", autouse=True)
# def setup_logging():
#     """Fixture to set up logging for all tests.

#     This fixture is automatically used for all tests due to autouse=True.
#     It configures the main logger and dependency warnings.
#     """
#     logger = get_logger("oumi", level="debug")
#     logger.propagate = True
#     configure_dependency_warnings(level="debug")
#     return logger
