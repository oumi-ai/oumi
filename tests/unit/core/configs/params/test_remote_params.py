from oumi.core.configs.params.remote_params import RemoteParams


def test_remote_params_allows_empty():
    params = RemoteParams()
    params.finalize_and_validate()
    # No exception should be raised
