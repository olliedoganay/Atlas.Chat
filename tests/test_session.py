import unittest

from atlas_local.session import (
    legacy_scoped_thread_id,
    parse_scoped_thread_id,
    scoped_thread_id,
)


class ScopedThreadIdTests(unittest.TestCase):
    def test_v2_ids_are_unambiguous_for_legacy_collision_inputs(self) -> None:
        first = scoped_thread_id("a", "b::c")
        second = scoped_thread_id("a::b", "c")

        self.assertNotEqual(first, second)
        self.assertEqual(parse_scoped_thread_id(first), ("a", "b::c"))
        self.assertEqual(parse_scoped_thread_id(second), ("a::b", "c"))

    def test_v2_ids_round_trip_unicode_and_delimiters(self) -> None:
        value = scoped_thread_id("profile::ışık", "thread/::/研究")

        self.assertTrue(value.startswith("atlas-thread-v2:"))
        self.assertEqual(
            parse_scoped_thread_id(value),
            ("profile::ışık", "thread/::/研究"),
        )

    def test_parser_rejects_legacy_and_malformed_values(self) -> None:
        self.assertEqual(legacy_scoped_thread_id("a", "b"), "a::b")
        self.assertIsNone(parse_scoped_thread_id("a::b"))
        self.assertIsNone(parse_scoped_thread_id("atlas-thread-v2:not-base64"))


if __name__ == "__main__":
    unittest.main()
