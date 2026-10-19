"""Filesystem behaviour of engine.save_transcript against a real tmp_path."""

import os

import pytest

import engine

SPANISH_TEXT = "Añádelo aquí: ¿cómo estás? ¡Sí! El ñandú comió piñones."
CHINESE_JAPANESE_KOREAN_TEXT = "会議の記録です。中文字幕在这里。한국어 자막입니다."
EMOJI_TEXT = "Transcript 🎙️ finished ✅ 👍🏽 team 👨‍👩‍👧‍👦 done."
CYRILLIC_TEXT = "Привет, мир. Это расшифровка."
ARABIC_TEXT = "مرحبا بالعالم، هذا نص."
TYPOGRAPHIC_TEXT = "café — naïve … «citation» → 30 °C ≈ 86 °F"

ALL_SUPPORTED_EXTENSIONS = [
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".wma", ".aac",
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".ts", ".flv",
]

UTF8_BYTE_ORDER_MARK = b"\xef\xbb\xbf"


def written_bytes(directory, name):
    return (directory / name).read_bytes()


def written_text(directory, name):
    return (directory / name).read_text(encoding="utf-8")


class TestOutputNaming:
    @pytest.mark.parametrize("extension", ALL_SUPPORTED_EXTENSIONS)
    def test_replaces_the_source_extension_with_txt(self, tmp_path, extension):
        returned_name = engine.save_transcript(f"interview{extension}", "body", str(tmp_path))

        assert returned_name == "interview.txt"

    @pytest.mark.parametrize("extension", [".MP4", ".WAV", ".Mp3", ".FlAc"])
    def test_uppercase_source_extensions_are_replaced_too(self, tmp_path, extension):
        returned_name = engine.save_transcript(f"clip{extension}", "body", str(tmp_path))

        assert returned_name == "clip.txt"

    def test_only_the_last_extension_is_removed_when_the_name_contains_dots(self, tmp_path):
        returned_name = engine.save_transcript("lecture.part2.mp4", "body", str(tmp_path))

        assert returned_name == "lecture.part2.txt"

    def test_every_inner_dot_is_kept_for_deeply_dotted_names(self, tmp_path):
        returned_name = engine.save_transcript("2024.05.01.session.03.mkv", "body", str(tmp_path))

        assert returned_name == "2024.05.01.session.03.txt"

    def test_source_without_an_extension_still_gets_a_txt_suffix(self, tmp_path):
        returned_name = engine.save_transcript("README", "body", str(tmp_path))

        assert returned_name == "README.txt"
        assert (tmp_path / "README.txt").is_file()

    def test_source_already_named_txt_keeps_exactly_one_txt_suffix(self, tmp_path):
        returned_name = engine.save_transcript("notes.txt", "body", str(tmp_path))

        assert returned_name == "notes.txt"

    def test_leading_dot_name_is_treated_as_a_bare_name_and_gains_txt(self, tmp_path):
        returned_name = engine.save_transcript(".gitignore", "body", str(tmp_path))

        assert returned_name == ".gitignore.txt"

    def test_trailing_dot_is_dropped_from_the_output_name(self, tmp_path):
        returned_name = engine.save_transcript("audio.", "body", str(tmp_path))

        assert returned_name == "audio.txt"

    def test_empty_filepath_produces_a_bare_txt_file(self, tmp_path):
        returned_name = engine.save_transcript("", "body", str(tmp_path))

        assert returned_name == ".txt"
        assert written_text(tmp_path, ".txt") == "body"

    def test_filepath_that_is_only_a_directory_produces_a_bare_txt_file(self, tmp_path):
        returned_name = engine.save_transcript("C:\\videos\\", "body", str(tmp_path))

        assert returned_name == ".txt"

    def test_spaces_in_the_source_name_are_preserved(self, tmp_path):
        returned_name = engine.save_transcript("board meeting 03 - final.mp4", "body", str(tmp_path))

        assert returned_name == "board meeting 03 - final.txt"
        assert (tmp_path / returned_name).is_file()

    def test_punctuation_in_the_source_name_is_preserved(self, tmp_path):
        returned_name = engine.save_transcript("q1_report (draft) [v2]+final!.wav", "body", str(tmp_path))

        assert returned_name == "q1_report (draft) [v2]+final!.txt"

    @pytest.mark.parametrize(
        ("source_name", "expected_name"),
        [
            ("reunión.mp3", "reunión.txt"),
            ("会議録.mp4", "会議録.txt"),
            ("Привет.wav", "Привет.txt"),
            ("clip 🎙️.mkv", "clip 🎙️.txt"),
        ],
    )
    def test_non_ascii_source_names_survive_into_the_output_name(self, tmp_path, source_name, expected_name):
        returned_name = engine.save_transcript(source_name, "body", str(tmp_path))

        assert returned_name == expected_name
        assert written_text(tmp_path, expected_name) == "body"

    def test_dots_in_the_source_directory_do_not_affect_the_output_name(self, tmp_path):
        returned_name = engine.save_transcript("C:\\my.media.folder\\clip.mp4", "body", str(tmp_path))

        assert returned_name == "clip.txt"

    def test_dots_in_the_source_directory_do_not_leak_when_the_name_has_no_extension(self, tmp_path):
        returned_name = engine.save_transcript("C:\\my.media.folder\\clip", "body", str(tmp_path))

        assert returned_name == "clip.txt"

    @pytest.mark.parametrize(
        "filepath",
        [
            "clip.mp4",
            "C:\\videos\\clip.mp4",
            "C:/videos/clip.mp4",
            "relative/nested/clip.mp4",
            "relative\\nested\\clip.mp4",
            "\\\\server\\share\\clip.mp4",
            "D:\\a\\b\\c\\d\\e\\clip.mp4",
        ],
    )
    def test_output_name_is_independent_of_the_source_directory(self, tmp_path, filepath):
        returned_name = engine.save_transcript(filepath, "body", str(tmp_path))

        assert returned_name == "clip.txt"

    def test_the_output_name_always_ends_with_the_txt_suffix(self, tmp_path):
        names = [
            engine.save_transcript(source, "body", str(tmp_path))
            for source in ("a.mp3", "b", "c.tar.gz", ".hidden", "d.")
        ]

        assert all(name.endswith(".txt") for name in names)


class TestOutputLocation:
    def test_writes_into_the_output_directory_and_not_beside_the_source(self, tmp_path):
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        source_dir.mkdir()
        output_dir.mkdir()
        source_file = source_dir / "clip.mp4"
        source_file.write_bytes(b"not really a video")

        returned_name = engine.save_transcript(str(source_file), "body", str(output_dir))

        assert (output_dir / returned_name).is_file()
        assert not (source_dir / "clip.txt").exists()

    def test_the_source_directory_is_left_exactly_as_it_was(self, tmp_path):
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        source_dir.mkdir()
        output_dir.mkdir()
        source_file = source_dir / "clip.mp4"
        source_file.write_bytes(b"payload")
        before = sorted(entry.name for entry in source_dir.iterdir())

        engine.save_transcript(str(source_file), "body", str(output_dir))

        assert sorted(entry.name for entry in source_dir.iterdir()) == before
        assert source_file.read_bytes() == b"payload"

    def test_the_source_file_never_needs_to_exist(self, tmp_path):
        missing_source = tmp_path / "source" / "never-created.mp4"

        returned_name = engine.save_transcript(str(missing_source), "body", str(tmp_path))

        assert returned_name == "never-created.txt"
        assert not missing_source.exists()

    def test_output_directory_with_a_trailing_separator_is_accepted(self, tmp_path):
        returned_name = engine.save_transcript("clip.mp4", "body", str(tmp_path) + os.sep)

        assert returned_name == "clip.txt"
        assert written_text(tmp_path, "clip.txt") == "body"

    def test_a_nested_output_directory_receives_the_transcript(self, tmp_path):
        nested = tmp_path / "level one" / "level.two" / "level3"
        nested.mkdir(parents=True)

        returned_name = engine.save_transcript("clip.mp4", "body", str(nested))

        assert returned_name == "clip.txt"
        assert written_text(nested, "clip.txt") == "body"

    def test_an_output_directory_whose_name_has_a_dot_does_not_change_the_output_name(self, tmp_path):
        output_dir = tmp_path / "transcripts.v2"
        output_dir.mkdir()

        returned_name = engine.save_transcript("clip.mp4", "body", str(output_dir))

        assert returned_name == "clip.txt"
        assert (output_dir / "clip.txt").is_file()

    def test_a_relative_output_directory_resolves_against_the_working_directory(self, tmp_path, monkeypatch):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        monkeypatch.chdir(tmp_path)

        returned_name = engine.save_transcript("clip.mp4", "body", "out")

        assert returned_name == "clip.txt"
        assert written_text(output_dir, "clip.txt") == "body"

    def test_an_empty_output_directory_writes_into_the_working_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        returned_name = engine.save_transcript("clip.mp4", "body", "")

        assert returned_name == "clip.txt"
        assert written_text(tmp_path, "clip.txt") == "body"

    def test_a_dot_output_directory_writes_into_the_working_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        returned_name = engine.save_transcript("clip.mp4", "body", ".")

        assert returned_name == "clip.txt"
        assert written_text(tmp_path, "clip.txt") == "body"

    def test_only_one_file_is_created_per_save(self, tmp_path):
        engine.save_transcript("clip.mp4", "body", str(tmp_path))

        assert sorted(entry.name for entry in tmp_path.iterdir()) == ["clip.txt"]

    def test_two_different_sources_produce_two_separate_transcripts(self, tmp_path):
        engine.save_transcript("first.mp4", "one", str(tmp_path))
        engine.save_transcript("second.mp3", "two", str(tmp_path))

        assert sorted(entry.name for entry in tmp_path.iterdir()) == ["first.txt", "second.txt"]
        assert written_text(tmp_path, "first.txt") == "one"
        assert written_text(tmp_path, "second.txt") == "two"


class TestReturnValue:
    def test_returns_only_the_basename_and_not_the_full_path(self, tmp_path):
        returned_name = engine.save_transcript("C:\\videos\\clip.mp4", "body", str(tmp_path))

        assert returned_name == "clip.txt"
        assert str(tmp_path) not in returned_name

    def test_the_returned_name_contains_no_path_separator(self, tmp_path):
        deep_output = tmp_path / "a" / "b" / "c"
        deep_output.mkdir(parents=True)

        returned_name = engine.save_transcript("D:\\media\\clip.mov", "body", str(deep_output))

        assert os.sep not in returned_name
        assert "/" not in returned_name

    def test_the_returned_name_joined_with_the_output_directory_locates_the_file(self, tmp_path):
        returned_name = engine.save_transcript("C:\\videos\\clip.mp4", "the body", str(tmp_path))

        assert (tmp_path / returned_name).read_text(encoding="utf-8") == "the body"

    def test_the_returned_value_is_a_string(self, tmp_path):
        returned_name = engine.save_transcript("clip.mp4", "body", str(tmp_path))

        assert isinstance(returned_name, str)

    def test_the_returned_name_is_stable_across_repeated_saves(self, tmp_path):
        first = engine.save_transcript("clip.mp4", "one", str(tmp_path))
        second = engine.save_transcript("clip.mp4", "two", str(tmp_path))

        assert first == second

    def test_the_returned_name_does_not_depend_on_the_output_directory_depth(self, tmp_path):
        shallow = tmp_path / "shallow"
        deep = tmp_path / "one" / "two" / "three"
        shallow.mkdir()
        deep.mkdir(parents=True)

        assert engine.save_transcript("clip.mp4", "body", str(shallow)) == \
            engine.save_transcript("clip.mp4", "body", str(deep))


class TestContentFidelity:
    def test_ascii_text_is_written_verbatim(self, tmp_path):
        text = "Hello there, this is the transcript."

        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_text(tmp_path, returned_name) == text

    @pytest.mark.parametrize(
        "text",
        [
            SPANISH_TEXT,
            CHINESE_JAPANESE_KOREAN_TEXT,
            EMOJI_TEXT,
            CYRILLIC_TEXT,
            ARABIC_TEXT,
            TYPOGRAPHIC_TEXT,
        ],
    )
    def test_non_ascii_text_round_trips_unchanged(self, tmp_path, text):
        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_text(tmp_path, returned_name) == text

    @pytest.mark.parametrize(
        "text",
        [
            SPANISH_TEXT,
            CHINESE_JAPANESE_KOREAN_TEXT,
            EMOJI_TEXT,
            CYRILLIC_TEXT,
            ARABIC_TEXT,
            TYPOGRAPHIC_TEXT,
        ],
    )
    def test_the_bytes_on_disk_are_the_utf8_encoding_of_the_text(self, tmp_path, text):
        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_bytes(tmp_path, returned_name) == text.encode("utf-8")

    @pytest.mark.parametrize(
        "text",
        [SPANISH_TEXT, CHINESE_JAPANESE_KOREAN_TEXT, EMOJI_TEXT, CYRILLIC_TEXT],
    )
    def test_the_bytes_on_disk_decode_explicitly_as_utf8(self, tmp_path, text):
        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        raw_bytes = written_bytes(tmp_path, returned_name)

        assert raw_bytes.decode("utf-8") == text

    def test_non_ascii_text_is_not_written_in_the_windows_ansi_codepage(self, tmp_path):
        returned_name = engine.save_transcript("clip.mp4", SPANISH_TEXT, str(tmp_path))

        raw_bytes = written_bytes(tmp_path, returned_name)

        assert raw_bytes != SPANISH_TEXT.encode("cp1252")
        assert len(raw_bytes) > len(SPANISH_TEXT.encode("cp1252"))

    def test_non_ascii_bytes_cannot_be_decoded_as_ascii(self, tmp_path):
        returned_name = engine.save_transcript("clip.mp4", SPANISH_TEXT, str(tmp_path))

        raw_bytes = written_bytes(tmp_path, returned_name)

        with pytest.raises(UnicodeDecodeError):
            raw_bytes.decode("ascii")

    def test_pure_ascii_text_uses_one_byte_per_character(self, tmp_path):
        text = "plain ascii transcript"

        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert len(written_bytes(tmp_path, returned_name)) == len(text)

    @pytest.mark.parametrize(
        ("character", "expected_bytes"),
        [
            ("ñ", b"\xc3\xb1"),
            ("é", b"\xc3\xa9"),
            ("¿", b"\xc2\xbf"),
            ("会", b"\xe4\xbc\x9a"),
            ("🎙", b"\xf0\x9f\x8e\x99"),
        ],
    )
    def test_single_non_ascii_characters_use_their_utf8_sequences(self, tmp_path, character, expected_bytes):
        returned_name = engine.save_transcript("clip.mp4", character, str(tmp_path))

        assert written_bytes(tmp_path, returned_name) == expected_bytes

    def test_no_byte_order_mark_is_written_for_ascii_text(self, tmp_path):
        returned_name = engine.save_transcript("clip.mp4", "plain", str(tmp_path))

        assert not written_bytes(tmp_path, returned_name).startswith(UTF8_BYTE_ORDER_MARK)

    def test_no_byte_order_mark_is_written_for_non_ascii_text(self, tmp_path):
        returned_name = engine.save_transcript("clip.mp4", SPANISH_TEXT, str(tmp_path))

        raw_bytes = written_bytes(tmp_path, returned_name)

        assert not raw_bytes.startswith(UTF8_BYTE_ORDER_MARK)
        assert raw_bytes[0:1] == b"A"

    def test_no_byte_order_mark_is_written_for_empty_text(self, tmp_path):
        returned_name = engine.save_transcript("clip.mp4", "", str(tmp_path))

        assert written_bytes(tmp_path, returned_name) == b""

    def test_decoding_with_the_bom_aware_codec_yields_the_same_text(self, tmp_path):
        returned_name = engine.save_transcript("clip.mp4", SPANISH_TEXT, str(tmp_path))

        raw_bytes = written_bytes(tmp_path, returned_name)

        assert raw_bytes.decode("utf-8-sig") == raw_bytes.decode("utf-8")

    def test_an_explicit_zero_width_no_break_space_in_the_text_is_kept_as_content(self, tmp_path):
        text = "\ufeffleading marker"

        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_bytes(tmp_path, returned_name) == text.encode("utf-8")

    def test_precomposed_and_decomposed_accents_are_not_normalised(self, tmp_path):
        precomposed = "caf\u00e9"
        decomposed = "cafe\u0301"

        engine.save_transcript("one.mp4", precomposed, str(tmp_path))
        engine.save_transcript("two.mp4", decomposed, str(tmp_path))

        assert written_text(tmp_path, "one.txt") == precomposed
        assert written_text(tmp_path, "two.txt") == decomposed
        assert written_bytes(tmp_path, "one.txt") != written_bytes(tmp_path, "two.txt")

    def test_empty_text_writes_an_empty_file_that_exists(self, tmp_path):
        returned_name = engine.save_transcript("clip.mp4", "", str(tmp_path))

        transcript = tmp_path / returned_name

        assert transcript.is_file()
        assert transcript.stat().st_size == 0

    @pytest.mark.parametrize("text", [" ", "   ", "\t", " \t "])
    def test_whitespace_only_text_is_written_as_is(self, tmp_path, text):
        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_bytes(tmp_path, returned_name) == text.encode("utf-8")

    def test_leading_and_trailing_spaces_are_not_stripped(self, tmp_path):
        text = "   padded transcript   "

        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_text(tmp_path, returned_name) == text

    def test_a_large_transcript_is_written_completely(self, tmp_path):
        text = "sentence number one. " * 20000

        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert (tmp_path / returned_name).stat().st_size == len(text)
        assert written_text(tmp_path, returned_name) == text

    def test_null_characters_inside_the_text_are_written(self, tmp_path):
        text = "before\x00after"

        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_bytes(tmp_path, returned_name) == b"before\x00after"


class TestNewlineHandling:
    def test_line_feeds_round_trip_through_a_text_mode_read(self, tmp_path):
        text = "first line\nsecond line\nthird line"

        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_text(tmp_path, returned_name) == text

    def test_the_line_count_is_preserved(self, tmp_path):
        text = "one\ntwo\nthree\nfour"

        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_text(tmp_path, returned_name).split("\n") == ["one", "two", "three", "four"]

    def test_blank_lines_between_paragraphs_are_preserved(self, tmp_path):
        text = "paragraph one\n\nparagraph two\n\n\nparagraph three"

        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_text(tmp_path, returned_name) == text

    def test_a_trailing_newline_is_kept(self, tmp_path):
        text = "last line\n"

        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_text(tmp_path, returned_name) == text

    def test_a_leading_newline_is_kept(self, tmp_path):
        text = "\nfirst real line"

        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_text(tmp_path, returned_name) == text

    def test_text_that_is_only_newlines_is_kept(self, tmp_path):
        text = "\n\n\n"

        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_text(tmp_path, returned_name) == text

    def test_line_feeds_are_translated_to_the_platform_line_separator_on_disk(self, tmp_path):
        text = "first\nsecond"

        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_bytes(tmp_path, returned_name) == f"first{os.linesep}second".encode()

    def test_a_carriage_return_line_feed_in_the_text_becomes_a_doubled_carriage_return_on_disk(self, tmp_path):
        text = "first\r\nsecond"

        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_bytes(tmp_path, returned_name) == b"first\r\r\nsecond"

    def test_a_carriage_return_line_feed_reads_back_as_two_line_breaks(self, tmp_path):
        text = "first\r\nsecond"

        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_text(tmp_path, returned_name) == "first\n\nsecond"

    def test_a_lone_carriage_return_is_written_untranslated_but_reads_back_as_a_line_feed(self, tmp_path):
        text = "first\rsecond"

        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_bytes(tmp_path, returned_name) == b"first\rsecond"
        assert written_text(tmp_path, returned_name) == "first\nsecond"

    def test_newlines_mixed_with_non_ascii_still_round_trip(self, tmp_path):
        text = f"{SPANISH_TEXT}\n{CHINESE_JAPANESE_KOREAN_TEXT}\n{EMOJI_TEXT}"

        returned_name = engine.save_transcript("clip.mp4", text, str(tmp_path))

        assert written_text(tmp_path, returned_name) == text


class TestOverwriteBehaviour:
    def test_an_existing_transcript_is_replaced(self, tmp_path):
        (tmp_path / "clip.txt").write_text("previous content", encoding="utf-8")

        returned_name = engine.save_transcript("clip.mp4", "fresh content", str(tmp_path))

        assert written_text(tmp_path, returned_name) == "fresh content"

    def test_shorter_replacement_text_truncates_the_previous_content(self, tmp_path):
        (tmp_path / "clip.txt").write_text("a very long previous transcript body", encoding="utf-8")

        engine.save_transcript("clip.mp4", "short", str(tmp_path))

        assert written_text(tmp_path, "clip.txt") == "short"
        assert (tmp_path / "clip.txt").stat().st_size == len("short")

    def test_overwriting_with_empty_text_empties_the_file(self, tmp_path):
        (tmp_path / "clip.txt").write_text("something was here", encoding="utf-8")

        engine.save_transcript("clip.mp4", "", str(tmp_path))

        assert (tmp_path / "clip.txt").stat().st_size == 0

    def test_overwriting_a_file_written_by_another_tool_replaces_its_encoding(self, tmp_path):
        (tmp_path / "clip.txt").write_bytes(UTF8_BYTE_ORDER_MARK + "old".encode())

        engine.save_transcript("clip.mp4", SPANISH_TEXT, str(tmp_path))

        raw_bytes = written_bytes(tmp_path, "clip.txt")

        assert not raw_bytes.startswith(UTF8_BYTE_ORDER_MARK)
        assert raw_bytes == SPANISH_TEXT.encode("utf-8")

    def test_repeated_saves_leave_exactly_one_file(self, tmp_path):
        for index in range(5):
            engine.save_transcript("clip.mp4", f"pass {index}", str(tmp_path))

        assert sorted(entry.name for entry in tmp_path.iterdir()) == ["clip.txt"]
        assert written_text(tmp_path, "clip.txt") == "pass 4"

    def test_overwriting_one_transcript_leaves_its_siblings_untouched(self, tmp_path):
        engine.save_transcript("first.mp4", "first body", str(tmp_path))
        engine.save_transcript("second.mp4", "second body", str(tmp_path))

        engine.save_transcript("first.mp4", "rewritten", str(tmp_path))

        assert written_text(tmp_path, "first.txt") == "rewritten"
        assert written_text(tmp_path, "second.txt") == "second body"

    def test_sources_sharing_a_basename_across_extensions_collide_on_one_transcript(self, tmp_path):
        first_name = engine.save_transcript("clip.mp3", "audio version", str(tmp_path))
        second_name = engine.save_transcript("clip.mp4", "video version", str(tmp_path))

        assert first_name == second_name
        assert sorted(entry.name for entry in tmp_path.iterdir()) == ["clip.txt"]
        assert written_text(tmp_path, "clip.txt") == "video version"

    def test_sources_sharing_a_basename_across_directories_collide_on_one_transcript(self, tmp_path):
        engine.save_transcript("C:\\one\\clip.mp4", "from one", str(tmp_path))
        engine.save_transcript("C:\\two\\clip.mp4", "from two", str(tmp_path))

        assert sorted(entry.name for entry in tmp_path.iterdir()) == ["clip.txt"]
        assert written_text(tmp_path, "clip.txt") == "from two"

    def test_a_read_only_source_beside_the_output_directory_is_not_involved(self, tmp_path):
        source_file = tmp_path / "clip.mp4"
        source_file.write_bytes(b"binary payload")

        engine.save_transcript(str(source_file), "transcript body", str(tmp_path))

        assert source_file.read_bytes() == b"binary payload"
        assert written_text(tmp_path, "clip.txt") == "transcript body"


class TestErrorPaths:
    def test_a_missing_output_directory_raises_file_not_found(self, tmp_path):
        missing_directory = tmp_path / "does-not-exist"

        with pytest.raises(FileNotFoundError):
            engine.save_transcript("clip.mp4", "body", str(missing_directory))

        assert not missing_directory.exists()

    def test_a_missing_output_directory_is_not_created_on_the_way_out(self, tmp_path):
        missing_directory = tmp_path / "a" / "b"

        with pytest.raises(FileNotFoundError):
            engine.save_transcript("clip.mp4", "body", str(missing_directory))

        assert sorted(entry.name for entry in tmp_path.iterdir()) == []

    def test_an_output_directory_that_is_actually_a_file_raises_an_os_error(self, tmp_path):
        blocking_file = tmp_path / "not-a-directory"
        blocking_file.write_text("occupied", encoding="utf-8")

        with pytest.raises(OSError):
            engine.save_transcript("clip.mp4", "body", str(blocking_file))

        assert blocking_file.read_text(encoding="utf-8") == "occupied"

    def test_an_output_path_already_taken_by_a_directory_raises_an_os_error(self, tmp_path):
        (tmp_path / "clip.txt").mkdir()

        with pytest.raises(OSError):
            engine.save_transcript("clip.mp4", "body", str(tmp_path))

        assert (tmp_path / "clip.txt").is_dir()

    @pytest.mark.parametrize("bad_text", [None, 42, b"bytes", ["a", "b"]])
    def test_a_non_string_body_raises_type_error(self, tmp_path, bad_text):
        with pytest.raises(TypeError):
            engine.save_transcript("clip.mp4", bad_text, str(tmp_path))

    def test_a_non_string_body_still_truncates_an_existing_transcript(self, tmp_path):
        (tmp_path / "clip.txt").write_text("valuable earlier transcript", encoding="utf-8")

        with pytest.raises(TypeError):
            engine.save_transcript("clip.mp4", None, str(tmp_path))

        assert (tmp_path / "clip.txt").stat().st_size == 0

    def test_a_non_string_body_leaves_an_empty_file_behind(self, tmp_path):
        with pytest.raises(TypeError):
            engine.save_transcript("clip.mp4", None, str(tmp_path))

        assert (tmp_path / "clip.txt").is_file()
        assert (tmp_path / "clip.txt").stat().st_size == 0

    def test_a_non_string_filepath_raises_type_error(self, tmp_path):
        with pytest.raises(TypeError):
            engine.save_transcript(None, "body", str(tmp_path))

    def test_a_non_string_output_directory_raises_type_error(self, tmp_path):
        with pytest.raises(TypeError):
            engine.save_transcript("clip.mp4", "body", None)

    def test_errors_are_not_swallowed_and_no_sentinel_is_returned(self, tmp_path):
        missing_directory = tmp_path / "gone"

        try:
            engine.save_transcript("clip.mp4", "body", str(missing_directory))
        except FileNotFoundError as error:
            assert "clip.txt" in str(error)
        else:
            raise AssertionError("save_transcript swallowed a missing output directory")


class TestFileHandleLifetime:
    def test_the_transcript_can_be_deleted_immediately_which_proves_the_handle_closed(self, tmp_path):
        returned_name = engine.save_transcript("clip.mp4", "body", str(tmp_path))

        (tmp_path / returned_name).unlink()

        assert not (tmp_path / returned_name).exists()

    def test_the_transcript_can_be_renamed_immediately(self, tmp_path):
        returned_name = engine.save_transcript("clip.mp4", "body", str(tmp_path))

        (tmp_path / returned_name).rename(tmp_path / "renamed.txt")

        assert written_text(tmp_path, "renamed.txt") == "body"

    def test_the_transcript_can_be_rewritten_immediately_by_another_writer(self, tmp_path):
        returned_name = engine.save_transcript("clip.mp4", "body", str(tmp_path))

        (tmp_path / returned_name).write_text("replaced by the test", encoding="utf-8")

        assert written_text(tmp_path, returned_name) == "replaced by the test"
