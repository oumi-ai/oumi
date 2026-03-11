import pytest

import oumi.cli.deploy as deploy


def test_get_deployment_client_normalizes_provider(monkeypatch):
    class DummyClient:
        pass

    monkeypatch.setattr(deploy, "_DEPLOYMENT_CLIENTS", {"modal": DummyClient})

    client = deploy._get_deployment_client("  MODAL  ")

    assert isinstance(client, DummyClient)


def test_get_deployment_client_unsupported_lists_supported_providers(monkeypatch):
    class DummyClient:
        pass

    monkeypatch.setattr(deploy, "_DEPLOYMENT_CLIENTS", {"fireworks": DummyClient})

    with pytest.raises(ValueError, match="Unsupported provider: modal"):
        deploy._get_deployment_client("modal")
