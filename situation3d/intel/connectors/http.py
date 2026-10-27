"""Allow-listed HTTPS fetching for VGI connectors.

The connectors only talk to a small fixed set of public hosts.  This module is
deliberately small: environment proxies are ignored, no cookies, no redirects,
no credentials in URLs.  It is the Python-side equivalent of the GEV service
proxy rules.
"""

from __future__ import annotations

import json
import ipaddress
import socket
import urllib.error
import urllib.request
from typing import Any, Iterable
from urllib.parse import urlparse

DEFAULT_TIMEOUT_S = 20.0
DEFAULT_MAX_BYTES = 8 * 1024 * 1024
USER_AGENT = "shenji-vgi/1.0"


class FetchError(RuntimeError):
    """A source fetch failed or was rejected by the allow-list."""


class MissingCredential(FetchError):
    """A required API key is not configured."""


def validate_url(url: str, allowed_hosts: Iterable[str]) -> str:
    """Return ``url`` if it is https on an allowed default-port host."""
    text = str(url or "").strip()
    if not text:
        raise FetchError("empty URL")
    if any(ord(ch) < 32 or ch.isspace() for ch in text):
        raise FetchError("URL contains whitespace or control characters")
    parsed = urlparse(text)
    if parsed.scheme != "https":
        raise FetchError("only https is allowed")
    if parsed.username or parsed.password:
        raise FetchError("credentials in URL are not allowed")
    host = (parsed.hostname or "").lower().rstrip(".")
    allowed = {str(item).lower().rstrip(".") for item in allowed_hosts}
    if host not in allowed:
        raise FetchError("host is not allow-listed: %s" % (host or "?"))
    try:
        port = parsed.port
    except ValueError:
        raise FetchError("invalid port") from None
    if port not in (None, 443):
        raise FetchError("only the default HTTPS port is allowed")
    return text


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, "redirects are not allowed", headers, fp)


def ensure_public_host(url: str) -> None:
    """Reject a URL whose DNS resolution reaches private/reserved space."""
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme != "https":
        raise FetchError("custom feeds must use https")
    if parsed.username or parsed.password:
        raise FetchError("credentials in custom feed URL are not allowed")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise FetchError("URL has no host")
    try:
        if parsed.port not in (None, 443):
            raise FetchError("custom feeds only allow the default HTTPS port")
    except ValueError:
        raise FetchError("invalid port in custom feed URL") from None
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise FetchError("cannot resolve host %s: %s" % (host, exc)) from None
    for entry in addresses:
        address = entry[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            raise FetchError("invalid resolved address for %s" % (host,)) from None
        if (not ip.is_global or ip.is_private or ip.is_loopback
                or ip.is_link_local or ip.is_multicast or ip.is_reserved
                or ip.is_unspecified
                or ip in ipaddress.ip_network("64:ff9b::/96")):
            raise FetchError("host %s resolves to a non-public address" % (host,))


def fetch_bytes(url: str, allowed_hosts: Iterable[str],
                timeout_s: float = DEFAULT_TIMEOUT_S,
                max_bytes: int = DEFAULT_MAX_BYTES,
                user_agent: str = USER_AGENT) -> bytes:
    """Fetch one HTTPS resource with a host allow-list and a size cap."""
    validate_url(url, allowed_hosts)
    host = urlparse(url).hostname
    request = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent, "Accept": "*/*"},
    )
    # An explicit empty ProxyHandler disables HTTP(S)_PROXY; the FIRMS key is
    # in the URL path and must not be exposed to an environment proxy.
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=float(timeout_s)) as response:
            payload = response.read(int(max_bytes) + 1)
    except urllib.error.HTTPError as exc:
        raise FetchError("http %s from %s" % (exc.code, host)) from None
    except urllib.error.URLError as exc:
        raise FetchError("network error from %s: %s"
                         % (host, exc.reason)) from None
    except OSError as exc:
        raise FetchError("network error from %s: %s" % (host, exc)) from None
    if len(payload) > int(max_bytes):
        raise FetchError("response from %s exceeds %d bytes"
                         % (host, int(max_bytes)))
    return payload


def fetch_json(url: str, allowed_hosts: Iterable[str], **kwargs: Any):
    """Fetch and decode a JSON document."""
    raw = fetch_bytes(url, allowed_hosts, **kwargs)
    try:
        return json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise FetchError("invalid JSON from source: %s" % (exc,)) from None


def fetch_text(url: str, allowed_hosts: Iterable[str], **kwargs: Any) -> str:
    """Fetch a text document (CSV, etc.)."""
    raw = fetch_bytes(url, allowed_hosts, **kwargs)
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise FetchError("invalid text from source: %s" % (exc,)) from None
