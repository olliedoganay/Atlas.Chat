import base64
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from atlas_local.api_service import (
    MAX_PDF_PAGES,
    AtlasBackendService,
    _extract_pdf_text,
    _history_attachment_list,
    _message_content_to_history_parts,
    _validated_attachments,
)


def _data_url(media_type: str, content: bytes) -> str:
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


class AttachmentValidationTests(unittest.TestCase):
    def test_valid_image_is_normalized_with_actual_size(self) -> None:
        image_bytes = b"\x89PNG\r\n\x1a\nminimal"

        attachments = _validated_attachments(
            [
                {
                    "kind": "image",
                    "name": "sample.png",
                    "media_type": "image/png",
                    "data_url": _data_url("image/png", image_bytes),
                    "byte_size": len(image_bytes),
                }
            ]
        )

        self.assertEqual(
            attachments,
            [
                {
                    "kind": "image",
                    "name": "sample.png",
                    "media_type": "image/png",
                    "data_url": _data_url("image/png", image_bytes),
                    "byte_size": len(image_bytes),
                }
            ],
        )

    def test_image_rejects_missing_base64_marker_mime_mismatch_and_svg(self) -> None:
        png_bytes = b"\x89PNG\r\n\x1a\nminimal"

        invalid_payloads = [
            {
                "kind": "image",
                "name": "not-base64.png",
                "media_type": "image/png",
                "data_url": "data:image/png,iVBORw0KGgo=",
            },
            {
                "kind": "image",
                "name": "mismatch.jpg",
                "media_type": "image/jpeg",
                "data_url": _data_url("image/png", png_bytes),
            },
            {
                "kind": "image",
                "name": "active.svg",
                "media_type": "image/svg+xml",
                "data_url": _data_url("image/svg+xml", b"<svg></svg>"),
            },
        ]

        for payload in invalid_payloads:
            with self.subTest(name=payload["name"]):
                with self.assertRaises(RuntimeError):
                    _validated_attachments([payload])

    def test_image_rejects_content_that_does_not_match_common_media_type(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "content does not match"):
            _validated_attachments(
                [
                    {
                        "kind": "image",
                        "name": "fake.png",
                        "media_type": "image/png",
                        "data_url": _data_url("image/png", b"not a PNG"),
                    }
                ]
            )

    def test_image_rejects_unknown_image_subtype_even_with_valid_base64(self) -> None:
        heic_bytes = b"\x00\x00\x00\x18ftypheicnot-really-an-image"

        with self.assertRaisesRegex(RuntimeError, "unsupported image type"):
            _validated_attachments(
                [
                    {
                        "kind": "image",
                        "name": "photo.heic",
                        "media_type": "image/heic",
                        "data_url": _data_url("image/heic", heic_bytes),
                        "byte_size": len(heic_bytes),
                    }
                ]
            )

    def test_attachment_rejects_declared_size_mismatch(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "reported 999 bytes but contains 5 bytes"):
            _validated_attachments(
                [
                    {
                        "kind": "file",
                        "name": "notes.txt",
                        "media_type": "text/plain",
                        "text_content": "hello",
                        "byte_size": 999,
                    }
                ]
            )

    def test_text_data_url_uses_embedded_type_when_declared_type_is_generic(self) -> None:
        raw_text = b'{"atlas": true}'

        attachments = _validated_attachments(
            [
                {
                    "kind": "file",
                    "name": "attachment",
                    "media_type": "application/octet-stream",
                    "data_url": _data_url("application/json", raw_text),
                    "byte_size": len(raw_text),
                }
            ]
        )

        self.assertEqual(attachments[0]["media_type"], "application/json")
        self.assertEqual(attachments[0]["text_content"], raw_text.decode("utf-8"))
        self.assertEqual(attachments[0]["byte_size"], len(raw_text))

    def test_unsupported_or_empty_file_is_rejected_instead_of_dropped(self) -> None:
        payloads = [
            {
                "kind": "file",
                "name": "archive.zip",
                "media_type": "application/zip",
                "data_url": _data_url("application/zip", b"PK\x03\x04"),
            },
            {
                "kind": "file",
                "name": "empty.txt",
                "media_type": "text/plain",
                "text_content": "  ",
                "byte_size": 2,
            },
        ]

        for payload in payloads:
            with self.subTest(name=payload["name"]):
                with self.assertRaises(RuntimeError):
                    _validated_attachments([payload])

    def test_pdf_is_normalized_to_extracted_text(self) -> None:
        pdf_bytes = b"%PDF-1.7\nminimal"
        with patch(
            "atlas_local.api_service._extract_pdf_text",
            return_value="Extracted PDF text",
        ):
            attachments = _validated_attachments(
                [
                    {
                        "kind": "file",
                        "name": "brief.pdf",
                        "media_type": "application/pdf",
                        "data_url": _data_url("application/pdf", pdf_bytes),
                        "byte_size": len(pdf_bytes),
                    }
                ]
            )

        self.assertEqual(
            attachments,
            [
                {
                    "kind": "file",
                    "name": "brief.pdf",
                    "media_type": "application/pdf",
                    "text_content": "Extracted PDF text",
                    "byte_size": len(pdf_bytes),
                }
            ],
        )

    def test_pdf_extraction_examines_at_most_the_page_limit(self) -> None:
        extracted_pages: list[int] = []

        class _Page:
            def __init__(self, index: int) -> None:
                self.index = index

            def extract_text(self) -> str:
                extracted_pages.append(self.index)
                return ""

        reader = SimpleNamespace(
            pages=[_Page(index) for index in range(MAX_PDF_PAGES + 20)]
        )
        with patch("atlas_local.api_service.PdfReader", return_value=reader):
            result = _extract_pdf_text(
                _data_url("application/pdf", b"%PDF-1.7\nminimal")
            )

        self.assertEqual(result, "")
        self.assertEqual(len(extracted_pages), MAX_PDF_PAGES)

    def test_start_chat_rejects_invalid_attachment_before_queueing(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        queued: list[dict[str, object]] = []
        service._ensure_user_unlocked = lambda _user_id: None  # type: ignore[method-assign]
        service._start_run = lambda **kwargs: queued.append(kwargs) or {}  # type: ignore[method-assign]

        with self.assertRaisesRegex(RuntimeError, "not a supported text or PDF"):
            service.start_chat(
                prompt="",
                user_id="research_user",
                thread_id="main",
                attachments=[
                    {
                        "kind": "file",
                        "name": "archive.zip",
                        "media_type": "application/zip",
                        "data_url": _data_url("application/zip", b"PK\x03\x04"),
                    }
                ],
            )

        self.assertEqual(queued, [])

    def test_start_chat_rejects_empty_direct_service_request(self) -> None:
        service = AtlasBackendService.__new__(AtlasBackendService)
        service._ensure_user_unlocked = lambda _user_id: None  # type: ignore[method-assign]
        service._start_run = lambda **_kwargs: self.fail("run should not be queued")  # type: ignore[method-assign]

        with self.assertRaisesRegex(RuntimeError, "Prompt or a supported attachment"):
            service.start_chat(
                prompt="  ",
                user_id="research_user",
                thread_id="main",
            )


class AttachmentHistoryTests(unittest.TestCase):
    def test_history_fallback_does_not_expose_remote_or_svg_image_sources(self) -> None:
        content = [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": "https://tracker.example/image.png"},
            {
                "type": "image_url",
                "image_url": _data_url("image/svg+xml", b"<svg></svg>"),
            },
        ]

        text, attachments = _message_content_to_history_parts(content)

        self.assertEqual(text, "hello")
        self.assertEqual(attachments, [])

    def test_stored_history_filters_mismatched_image_metadata(self) -> None:
        attachments = _history_attachment_list(
            [
                {
                    "kind": "image",
                    "name": "mismatch.jpg",
                    "media_type": "image/jpeg",
                    "data_url": _data_url(
                        "image/png",
                        b"\x89PNG\r\n\x1a\nminimal",
                    ),
                }
            ]
        )

        self.assertEqual(attachments, [])

    def test_stored_history_infers_missing_legacy_image_media_type(self) -> None:
        data_url = _data_url("image/jpeg", b"\xff\xd8\xffminimal")

        attachments = _history_attachment_list(
            [
                {
                    "kind": "image",
                    "name": "legacy.jpg",
                    "data_url": data_url,
                }
            ]
        )

        self.assertEqual(attachments[0]["media_type"], "image/jpeg")
        self.assertEqual(attachments[0]["data_url"], data_url)


if __name__ == "__main__":
    unittest.main()
