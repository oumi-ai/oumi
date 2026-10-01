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

"""``oumi orcarouter`` CLI: two explicit sign-in paths and catalog inspection.

Both sign-in paths produce the same credential. ``--api-key`` (or the
``ORCAROUTER_API_KEY`` environment variable) stores a key the user already
holds; ``--login`` runs the OAuth 2.0 + PKCE browser flow. Neither replaces the
other.
"""

import json
import os
from typing import Annotated

import typer

import oumi.cli.cli_utils as cli_utils
from oumi.orcarouter.catalog import OrcaRouterCapability
from oumi.orcarouter.credentials import (
    KEYS_CONSOLE_URL,
    ORCAROUTER_API_KEY_ENV_VAR,
    CredentialSource,
    OrcaRouterAuthError,
    OrcaRouterCredentialError,
    OrcaRouterSession,
    mask_api_key,
    resolve_api_base_url,
    resolve_auth_base_url,
)
from oumi.utils.logging import logger

orcarouter_app = typer.Typer(
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["--help", "-h"]},
)


def _session_with_env_key() -> OrcaRouterSession:
    """Returns a session that also considers the documented environment key.

    ``ORCAROUTER_API_KEY`` is the zero-configuration path: when it is set and a
    key is not already stored, it is adopted so ``login`` and ``models`` work
    without a separate sign-in step.
    """
    session = OrcaRouterSession()
    env_key = os.environ.get(ORCAROUTER_API_KEY_ENV_VAR)
    if env_key and session.current_credential() is None:
        try:
            session.connect_with_api_key(env_key)
        except OrcaRouterCredentialError as e:
            logger.debug(f"ORCAROUTER_API_KEY was not adopted: {e}")
    return session


@orcarouter_app.command(name="login")
def orcarouter_login(
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help=(
                "Sign in with an existing OrcaRouter API key (sk-orca-...). "
                f"Falls back to the {ORCAROUTER_API_KEY_ENV_VAR} environment "
                "variable."
            ),
        ),
    ] = None,
    oob: Annotated[
        bool,
        typer.Option(
            "--oob",
            help=(
                "Use the out-of-band code flow. The consent screen shows a code "
                "that you paste back. Use this on hosts that cannot accept a "
                "loopback redirect."
            ),
        ),
    ] = False,
    app_name: Annotated[
        str,
        typer.Option(
            "--app-name",
            help="Label shown on the OrcaRouter consent screen.",
        ),
    ] = "Oumi",
    no_browser: Annotated[
        bool,
        typer.Option(
            "--no-browser",
            help="Print the authorization URL instead of opening a browser.",
        ),
    ] = False,
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help="Seconds to wait for the browser redirect.",
        ),
    ] = 300.0,
) -> None:
    """Connects Oumi to OrcaRouter and stores the credential.

    Two independent paths are offered: paste an existing API key, or authorize
    in a browser (OAuth 2.0 + PKCE S256). Both end up as the same kind of
    OrcaRouter API key; the key belongs to your account and can be revoked at
    any time.
    """
    session = OrcaRouterSession()
    cli_utils.section_header("OrcaRouter sign-in")
    cli_utils.CONSOLE.print(f"  Authentication origin: {resolve_auth_base_url()}")
    cli_utils.CONSOLE.print(f"  Inference origin:      {resolve_api_base_url()}/v1")

    try:
        if api_key is not None:
            credential = session.connect_with_api_key(api_key)
        elif os.environ.get(ORCAROUTER_API_KEY_ENV_VAR):
            credential = session.connect_with_api_key(None)
        else:
            credential = session.connect_with_oauth(
                flow="oob" if oob else "redirect",
                open_browser=not no_browser,
                app_name=app_name,
                timeout_seconds=timeout,
            )
    except OrcaRouterCredentialError as e:
        logger.error(str(e))
        raise typer.Exit(code=1) from None

    cli_utils.CONSOLE.print(
        f"\nStored a {credential.source.value} credential "
        f"({mask_api_key(credential.key)}) in {session.store.path}."
    )
    if credential.source is CredentialSource.OAUTH_PKCE:
        cli_utils.CONSOLE.print(
            "This key is durable: Oumi reuses it until you revoke it. It is not "
            "a refresh token, so there is nothing to refresh."
        )
    if credential.scope != "api":
        cli_utils.CONSOLE.print(
            f"Warning: OrcaRouter granted scope '{credential.scope}', not 'api'."
        )
    cli_utils.CONSOLE.print(
        f"Manage or revoke keys at [link={KEYS_CONSOLE_URL}]{KEYS_CONSOLE_URL}[/link]."
    )


@orcarouter_app.command(name="status")
def orcarouter_status() -> None:
    """Shows the stored OrcaRouter credential state, with the key masked."""
    session = _session_with_env_key()
    credential = session.current_credential()
    cli_utils.section_header("OrcaRouter credential")
    if credential is None:
        cli_utils.CONSOLE.print(
            "No credential is stored. Run [bold]oumi orcarouter login[/bold] "
            f"with {ORCAROUTER_API_KEY_ENV_VAR} set, or without it to authorize "
            "in a browser."
        )
        return

    cli_utils.CONSOLE.print(f"  Key:          {mask_api_key(credential.key)}")
    cli_utils.CONSOLE.print(f"  Source:       {credential.source.value}")
    cli_utils.CONSOLE.print(f"  Scope:        {credential.scope}")
    cli_utils.CONSOLE.print(f"  Account:      {credential.account_id or '<unknown>'}")
    cli_utils.CONSOLE.print(f"  Generation:   {credential.generation}")
    cli_utils.CONSOLE.print(f"  Needs reauth: {credential.needs_reauth}")
    cli_utils.CONSOLE.print(f"  Stored at:    {session.store.path}")


@orcarouter_app.command(name="logout")
def orcarouter_logout() -> None:
    """Removes the stored OrcaRouter credential from this machine."""
    session = OrcaRouterSession()
    if session.clear():
        cli_utils.CONSOLE.print("Removed the stored OrcaRouter credential.")
    else:
        cli_utils.CONSOLE.print("No stored OrcaRouter credential to remove.")


@orcarouter_app.command(name="models")
def orcarouter_models(
    capability: Annotated[
        str,
        typer.Option(
            "--capability",
            help=(
                "Capability to filter on: chat, embedding, image, video, rerank, "
                "or multimodal:<image|audio|video>."
            ),
        ),
    ] = OrcaRouterCapability.CHAT.value,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the catalog as JSON."),
    ] = False,
    current_model: Annotated[
        str | None,
        typer.Option(
            "--current-model",
            help=(
                "A model ID to validate against the filtered options. A value "
                "that is not compatible with the requested capability is "
                "cleared, exactly as a selector does when the capability changes."
            ),
        ),
    ] = None,
) -> None:
    """Lists the models OrcaRouter advertises for the stored credential.

    The list comes from the live ``/v1/models`` catalog. When the catalog is
    unreachable, a small verified fallback catalog is shown and clearly marked
    as degraded -- never as a free-text field and never as a hand-written
    example list pretending to be live.
    """
    try:
        OrcaRouterCapability.parse(capability)
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=2) from None

    session = _session_with_env_key()
    credential = session.active_credential()
    if credential is None and session.current_credential() is not None:
        logger.error(
            "The stored OrcaRouter credential needs reauthorization. Run "
            "`oumi orcarouter login`."
        )
        raise typer.Exit(code=1)

    from oumi.orcarouter.catalog import resolve_model_selector

    try:
        selector = resolve_model_selector(
            resolve_api_base_url(),
            credential.key if credential else None,
            capability=capability,
            current_model=current_model,
        )
    except OrcaRouterAuthError as e:
        # Terminal: mark exactly the credential that made the rejected request.
        if credential is not None:
            session.mark_needs_reauth(credential)
        logger.error(str(e))
        raise typer.Exit(code=1) from None
    except OrcaRouterCredentialError as e:
        logger.error(str(e))
        raise typer.Exit(code=1) from None
    result = selector.catalog

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "capability": capability,
                    "source": selector.source,
                    "degraded": selector.degraded,
                    "detail": selector.catalog.detail,
                    "selected": selector.selected,
                    "models": [model.to_dict() for model in selector.options],
                },
                indent=2,
            )
        )
        return

    cli_utils.section_header(f"OrcaRouter models ({capability})")
    if result.degraded:
        cli_utils.CONSOLE.print(
            f"[yellow]Degraded: live catalog unavailable ({result.detail}). "
            "Showing the verified fallback catalog.[/yellow]\n"
        )
    else:
        cli_utils.CONSOLE.print(f"  Live catalog: {result.catalog_source}\n")

    if not result.models:
        cli_utils.CONSOLE.print("  No models support this capability.")
        return
    for model in result.models:
        modalities = ",".join(model.input_modalities) or "?"
        cli_utils.CONSOLE.print(f"  {model.id}  [dim](inputs: {modalities})[/dim]")
