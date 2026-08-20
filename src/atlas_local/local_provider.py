from __future__ import annotations

import ipaddress
from typing import Any
from urllib import error, request
from urllib.parse import SplitResult, urlsplit, urlunsplit


MAX_PROVIDER_URL_LENGTH = 2048


def normalize_local_provider_base_url(
    value: str,
    *,
    allow_empty: bool = False,
) -> str:
    """Return a canonical loopback-only provider base URL.

    ``localhost`` is intentionally converted to a numeric loopback address so
    provider traffic cannot be redirected by DNS, NSS, or hosts-file changes
    after configuration validation.
    """

    return _normalize_local_provider_url(
        value,
        allow_empty=allow_empty,
        allow_query=False,
        strip_trailing_slash=True,
    )


def normalize_local_provider_request_url(value: str) -> str:
    """Return a canonical loopback-only request or redirect URL."""

    return _normalize_local_provider_url(
        value,
        allow_empty=False,
        allow_query=True,
        strip_trailing_slash=False,
    )


def provider_urlopen(
    target: str | request.Request,
    *,
    timeout: float,
) -> Any:
    """Open a provider request without proxies or cross-origin redirects."""

    if isinstance(target, request.Request):
        normalized_url = normalize_local_provider_request_url(target.full_url)
        resolved_target: str | request.Request = (
            target
            if normalized_url == target.full_url
            else _clone_request_with_url(target, normalized_url)
        )
    else:
        resolved_target = normalize_local_provider_request_url(str(target))
    return _PROVIDER_OPENER.open(resolved_target, timeout=timeout)


def _normalize_local_provider_url(
    value: str,
    *,
    allow_empty: bool,
    allow_query: bool,
    strip_trailing_slash: bool,
) -> str:
    resolved = str(value or "").strip()
    if not resolved:
        if allow_empty:
            return ""
        raise ValueError("Provider URL is required.")
    if len(resolved) > MAX_PROVIDER_URL_LENGTH:
        raise ValueError("Provider URL is too long.")
    if "\\" in resolved or any(
        character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F
        for character in resolved
    ):
        raise ValueError(
            "Provider URL cannot contain whitespace, control characters, or backslashes."
        )

    try:
        parsed = urlsplit(resolved)
        hostname = parsed.hostname
        parsed_port = parsed.port
    except (UnicodeError, ValueError) as exc:
        raise ValueError("Provider URL is malformed or contains an invalid port.") from exc

    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        raise ValueError("Provider URL must be an HTTP(S) loopback URL.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Provider URL must not contain credentials.")
    if parsed.fragment or parsed.query and not allow_query:
        raise ValueError("Provider base URL cannot contain a query or fragment.")
    if parsed_port is not None and not 1 <= parsed_port <= 65535:
        raise ValueError("Provider URL contains an invalid port.")

    canonical_host = _canonical_loopback_host(hostname)
    authority = (
        f"[{canonical_host}]"
        if ":" in canonical_host
        else canonical_host
    )
    if parsed_port is not None:
        authority = f"{authority}:{parsed_port}"

    path = parsed.path.rstrip("/") if strip_trailing_slash else parsed.path
    normalized = SplitResult(
        scheme=parsed.scheme.lower(),
        netloc=authority,
        path=path,
        query=parsed.query if allow_query else "",
        fragment="",
    )
    return urlunsplit(normalized)


def _canonical_loopback_host(hostname: str) -> str:
    normalized = hostname.strip().lower()
    if normalized == "localhost":
        return "127.0.0.1"
    if "%" in normalized:
        raise ValueError("Provider URL cannot contain an IPv6 zone identifier.")
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError as exc:
        raise ValueError(
            "Provider URL must use localhost or a numeric loopback address."
        ) from exc
    if not address.is_loopback:
        raise ValueError(
            "Provider URL must use localhost or a numeric loopback address."
        )
    return address.compressed


def _provider_origin(value: str) -> tuple[str, str, int]:
    parsed = urlsplit(normalize_local_provider_request_url(value))
    default_port = 443 if parsed.scheme == "https" else 80
    return parsed.scheme, str(parsed.hostname or ""), parsed.port or default_port


def _clone_request_with_url(
    source: request.Request,
    normalized_url: str,
) -> request.Request:
    cloned = request.Request(
        normalized_url,
        data=source.data,
        headers=dict(source.headers),
        origin_req_host=source.origin_req_host,
        unverifiable=source.unverifiable,
        method=source.get_method(),
    )
    for name, value in source.unredirected_hdrs.items():
        cloned.add_unredirected_header(name, value)
    return cloned


class _SameOriginProviderRedirectHandler(request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> request.Request | None:
        try:
            normalized_url = normalize_local_provider_request_url(newurl)
            same_origin = _provider_origin(req.full_url) == _provider_origin(
                normalized_url
            )
        except ValueError as exc:
            raise error.HTTPError(
                req.full_url,
                code,
                f"Atlas blocked an unsafe provider redirect: {exc}",
                headers,
                fp,
            ) from exc
        if not same_origin:
            raise error.HTTPError(
                req.full_url,
                code,
                "Atlas blocked a cross-origin provider redirect.",
                headers,
                fp,
            )
        return super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            normalized_url,
        )


# Supplying an empty ProxyHandler prevents build_opener from installing its
# default environment-backed proxy handler. OpenerDirector does not retain the
# empty handler because it exposes no protocol hooks.
_PROVIDER_OPENER = request.build_opener(
    request.ProxyHandler({}),
    _SameOriginProviderRedirectHandler(),
)
