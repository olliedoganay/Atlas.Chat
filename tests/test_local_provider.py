from __future__ import annotations

import unittest
from unittest.mock import patch
from urllib import error, request

from atlas_local import local_provider
from atlas_local.local_provider import (
    normalize_local_provider_base_url,
    provider_urlopen,
)


class LocalProviderUrlTests(unittest.TestCase):
    def test_normalizes_supported_loopback_urls_and_preserves_base_paths(self) -> None:
        cases = {
            "http://localhost:1234/v1/": "http://127.0.0.1:1234/v1",
            "http://127.42.0.7:8000/api/v1": "http://127.42.0.7:8000/api/v1",
            "https://[::1]:4443/v1/": "https://[::1]:4443/v1",
        }

        for raw_url, expected in cases.items():
            with self.subTest(raw_url=raw_url):
                self.assertEqual(
                    normalize_local_provider_base_url(raw_url),
                    expected,
                )

    def test_rejects_non_loopback_or_ambiguous_provider_urls(self) -> None:
        invalid_urls = (
            "https://example.com/v1",
            "http://192.168.1.8:8000/v1",
            "http://169.254.169.254/latest/meta-data",
            "http://2130706433:8000/v1",
            "http://user:secret@127.0.0.1:8000/v1",
            "http://127.0.0.1:8000/v1?key=value",
            "http://127.0.0.1:8000/v1#fragment",
            "http://127.0.0.1:0/v1",
            "http://[::1%25lo]:8000/v1",
            "http://127.0.0.1:8000\\@example.com/v1",
            "http://127.0.0.1:8000/v1\nInjected: value",
        )

        for raw_url in invalid_urls:
            with self.subTest(raw_url=raw_url):
                with self.assertRaises(ValueError):
                    normalize_local_provider_base_url(raw_url)

    def test_empty_url_is_only_allowed_when_explicitly_requested(self) -> None:
        with self.assertRaises(ValueError):
            normalize_local_provider_base_url("")

        self.assertEqual(
            normalize_local_provider_base_url("", allow_empty=True),
            "",
        )

    def test_provider_transport_disables_environment_proxies(self) -> None:
        proxy_handlers = [
            handler
            for handler in local_provider._PROVIDER_OPENER.handlers
            if isinstance(handler, request.ProxyHandler)
        ]

        # Passing ProxyHandler({}) suppresses urllib's default environment
        # proxy handler; an empty handler has no protocol hooks and is not
        # retained by OpenerDirector.
        self.assertEqual(proxy_handlers, [])

    def test_provider_transport_canonicalizes_localhost_before_opening(self) -> None:
        source = request.Request(
            "http://localhost:8000/v1/models",
            headers={"Authorization": "Bearer local-secret"},
        )
        sentinel = object()

        with patch.object(
            local_provider._PROVIDER_OPENER,
            "open",
            return_value=sentinel,
        ) as open_mock:
            result = provider_urlopen(source, timeout=2.0)

        self.assertIs(result, sentinel)
        opened_request = open_mock.call_args.args[0]
        self.assertEqual(
            opened_request.full_url,
            "http://127.0.0.1:8000/v1/models",
        )
        self.assertEqual(
            opened_request.get_header("Authorization"),
            "Bearer local-secret",
        )
        self.assertEqual(open_mock.call_args.kwargs["timeout"], 2.0)

    def test_same_origin_redirect_is_allowed_and_retains_authorization(self) -> None:
        source = request.Request(
            "http://127.0.0.1:8000/v1/models",
            headers={"Authorization": "Bearer local-secret"},
        )
        handler = local_provider._SameOriginProviderRedirectHandler()

        redirected = handler.redirect_request(
            source,
            None,
            302,
            "Found",
            {},
            "http://localhost:8000/v1/models/",
        )

        self.assertIsNotNone(redirected)
        self.assertEqual(
            redirected.full_url,
            "http://127.0.0.1:8000/v1/models/",
        )
        self.assertEqual(
            redirected.get_header("Authorization"),
            "Bearer local-secret",
        )

    def test_remote_and_other_local_origin_redirects_are_blocked(self) -> None:
        source = request.Request(
            "http://127.0.0.1:8000/v1/models",
            headers={"Authorization": "Bearer local-secret"},
        )
        handler = local_provider._SameOriginProviderRedirectHandler()

        for destination in (
            "https://example.com/collect",
            "http://127.0.0.1:8765/private",
            "https://127.0.0.1:8000/v1/models",
        ):
            with self.subTest(destination=destination):
                with self.assertRaises(error.HTTPError) as raised:
                    handler.redirect_request(
                        source,
                        None,
                        302,
                        "Found",
                        {},
                        destination,
                    )
                raised.exception.close()


if __name__ == "__main__":
    unittest.main()
