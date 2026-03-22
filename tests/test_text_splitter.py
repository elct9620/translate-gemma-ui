from translate_gemma_ui.text_splitter import Window, create_windows, merge_translations, split_sentences


class TestSplitSentences:
    def test_empty_text(self):
        assert split_sentences("") == []

    def test_whitespace_only(self):
        assert split_sentences("   ") == []

    def test_single_sentence(self):
        result = split_sentences("Hello world.")
        assert result == ["Hello world."]

    def test_multiple_english_sentences(self):
        result = split_sentences("First sentence. Second sentence. Third sentence.")
        assert result == ["First sentence.", "Second sentence.", "Third sentence."]

    def test_question_marks(self):
        result = split_sentences("What? Why not? Sure.")
        assert result == ["What?", "Why not?", "Sure."]

    def test_exclamation_marks(self):
        result = split_sentences("Wow! Amazing! Done.")
        assert result == ["Wow!", "Amazing!", "Done."]

    def test_chinese_punctuation(self):
        result = split_sentences("第一句。第二句？第三句！")
        assert result == ["第一句。", "第二句？", "第三句！"]

    def test_mixed_punctuation(self):
        result = split_sentences("Hello. 你好。How are you?")
        assert result == ["Hello.", "你好。", "How are you?"]

    def test_no_punctuation(self):
        result = split_sentences("No punctuation here")
        assert result == ["No punctuation here"]


class TestCreateWindows:
    def _token_count(self, text: str) -> int:
        return len(text.split())

    def test_empty_sentences(self):
        assert create_windows([], 100, self._token_count) == []

    def test_single_sentence_fits(self):
        sentences = ["Hello world."]
        windows = create_windows(sentences, 100, self._token_count)
        assert len(windows) == 1
        assert windows[0].start_idx == 0
        assert windows[0].end_idx == 0
        assert windows[0].overlap_count == 0

    def test_all_sentences_fit_one_window(self):
        sentences = ["One.", "Two.", "Three."]
        windows = create_windows(sentences, 100, self._token_count)
        assert len(windows) == 1
        assert windows[0].start_idx == 0
        assert windows[0].end_idx == 2

    def test_sentences_split_into_multiple_windows(self):
        sentences = ["Word " * 5 + "end.", "Word " * 5 + "end.", "Word " * 5 + "end."]
        windows = create_windows(sentences, 8, self._token_count)
        assert len(windows) > 1

    def test_overlap_applied(self):
        sentences = ["A.", "B.", "C.", "D.", "E."]
        windows = create_windows(sentences, 3, self._token_count, overlap=2)
        if len(windows) > 1:
            assert windows[1].overlap_count > 0
            assert windows[1].start_idx < windows[0].end_idx + 1

    def test_overlap_capped_at_start(self):
        sentences = ["A.", "B.", "C.", "D."]
        windows = create_windows(sentences, 2, self._token_count, overlap=2)
        assert windows[0].overlap_count == 0

    def test_zero_overlap(self):
        sentences = ["A.", "B.", "C.", "D."]
        windows = create_windows(sentences, 2, self._token_count, overlap=0)
        for w in windows:
            assert w.overlap_count == 0


class TestMergeTranslations:
    def test_empty(self):
        assert merge_translations([], [], []) == ""

    def test_single_window(self):
        windows = [Window(text="Hello.", start_idx=0, end_idx=0, overlap_count=0)]
        translations = ["你好。"]
        original = ["Hello."]
        result = merge_translations(windows, translations, original)
        assert result == "你好。"

    def test_later_segment_wins_overlap(self):
        windows = [
            Window(text="A. B. C.", start_idx=0, end_idx=2, overlap_count=0),
            Window(text="B-v2. C-v2. D.", start_idx=1, end_idx=3, overlap_count=2),
        ]
        translations = ["甲。 乙。 丙。", "乙二。 丙二。 丁。"]
        original = ["A.", "B.", "C.", "D."]
        result = merge_translations(windows, translations, original)
        assert "甲。" in result
        assert "乙二。" in result
        assert "丙二。" in result
        assert "丁。" in result

    def test_fallback_to_original_on_empty_translation(self):
        windows = [Window(text="Hello.", start_idx=0, end_idx=0, overlap_count=0)]
        translations = [""]
        original = ["Hello."]
        result = merge_translations(windows, translations, original)
        assert result == "Hello."
