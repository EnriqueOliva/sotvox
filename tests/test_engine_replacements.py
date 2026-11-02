"""Unit tests for engine.parse_replacements and engine.apply_replacements."""

import pytest

import engine

COMPANY = {"dasbank": "Dasbanq", "das bank": "Dasbanq", "dasban": "Dasbanq"}


class TestParseReplacements:
    def test_keeps_text_to_text_pairs_with_lowercase_variants(self):
        assert engine.parse_replacements({"DasBank": "Dasbanq"}) == {"dasbank": "Dasbanq"}

    def test_collapses_the_spaces_inside_a_variant(self):
        assert engine.parse_replacements({"  das   bank ": "Dasbanq"}) == {"das bank": "Dasbanq"}

    @pytest.mark.parametrize("raw_replacements", [None, [], "dasbank=Dasbanq", 3])
    def test_anything_but_a_mapping_means_no_replacements(self, raw_replacements):
        assert engine.parse_replacements(raw_replacements) == {}

    @pytest.mark.parametrize("variant, replacement", [("", "Dasbanq"), ("   ", "Dasbanq"), ("dasbank", None), ("dasbank", 7)])
    def test_skips_an_empty_variant_or_a_replacement_that_is_not_text(self, variant, replacement):
        assert engine.parse_replacements({variant: replacement, "dasban": "Dasbanq"}) == {"dasban": "Dasbanq"}


class TestApplyReplacements:
    def test_replaces_a_variant_whatever_its_case(self):
        assert engine.apply_replacements("Entré a DASBANK y después a dasbank", COMPANY) == (
            "Entré a Dasbanq y después a Dasbanq")

    def test_replaces_a_two_word_variant_across_any_spacing(self):
        assert engine.apply_replacements("la cuenta de das  bank quedó", COMPANY) == "la cuenta de Dasbanq quedó"

    def test_only_replaces_whole_words(self):
        assert engine.apply_replacements("dasbankdev y redasban siguen igual", COMPANY) == (
            "dasbankdev y redasban siguen igual")

    def test_prefers_the_longest_variant_when_one_starts_another(self):
        assert engine.apply_replacements("el panel de das bank", {"das": "X", "das bank": "Dasbanq"}) == (
            "el panel de Dasbanq")

    def test_leaves_punctuation_around_the_variant_in_place(self):
        assert engine.apply_replacements("(dasbank), dasban.", COMPANY) == "(Dasbanq), Dasbanq."

    def test_replaces_on_every_line(self):
        assert engine.apply_replacements("[00:03] Otros: dasbank\n[00:06] Enrique: das bank", COMPANY) == (
            "[00:03] Otros: Dasbanq\n[00:06] Enrique: Dasbanq")

    def test_treats_regex_characters_in_a_variant_literally(self):
        assert engine.apply_replacements("usamos c.i y cxi", {"c.i": "CI"}) == "usamos CI y cxi"

    def test_returns_the_text_unchanged_without_replacements(self):
        assert engine.apply_replacements("dasbank", {}) == "dasbank"
