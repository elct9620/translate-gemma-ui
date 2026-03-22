import pytest

from translate_gemma_ui.glossary import apply_glossary_post, apply_glossary_pre, match_glossary, parse_glossary


class TestParseGlossary:
    def test_parses_valid_csv(self):
        content = "API,應用程式介面\ncloud,雲端"
        result = parse_glossary(content)
        assert result == [("API", "應用程式介面"), ("cloud", "雲端")]

    def test_empty_content_returns_empty_list(self):
        assert parse_glossary("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert parse_glossary("  \n  \n") == []

    def test_trims_whitespace_from_terms(self):
        result = parse_glossary("  API , 應用程式介面 ")
        assert result == [("API", "應用程式介面")]

    def test_skips_empty_lines(self):
        content = "API,應用程式介面\n\ncloud,雲端\n"
        result = parse_glossary(content)
        assert result == [("API", "應用程式介面"), ("cloud", "雲端")]

    def test_handles_utf8_bom(self):
        content = "\ufeffAPI,應用程式介面"
        result = parse_glossary(content)
        assert result == [("API", "應用程式介面")]

    def test_wrong_column_count_raises_error(self):
        with pytest.raises(ValueError, match="2"):
            parse_glossary("API")

    def test_three_columns_raises_error(self):
        with pytest.raises(ValueError, match="2"):
            parse_glossary("API,應用程式介面,extra")

    def test_empty_term_raises_error(self):
        with pytest.raises(ValueError, match="空白"):
            parse_glossary("API,")

    def test_cjk_terms(self):
        content = "翻訳,翻譯"
        result = parse_glossary(content)
        assert result == [("翻訳", "翻譯")]


class TestMatchGlossary:
    def test_case_insensitive_match(self):
        glossary = [("API", "應用程式介面")]
        result = match_glossary("The api is ready", glossary)
        assert result == [("API", "應用程式介面")]

    def test_whole_word_match_rejects_substring(self):
        glossary = [("cat", "貓")]
        result = match_glossary("The category is set", glossary)
        assert result == []

    def test_whole_word_match_accepts_exact(self):
        glossary = [("cat", "貓")]
        result = match_glossary("The cat is here", glossary)
        assert result == [("cat", "貓")]

    def test_cjk_uses_substring_match(self):
        glossary = [("翻訳", "翻譯")]
        result = match_glossary("翻訳が完了しました", glossary)
        assert result == [("翻訳", "翻譯")]

    def test_no_match_returns_empty(self):
        glossary = [("API", "應用程式介面")]
        result = match_glossary("Hello world", glossary)
        assert result == []

    def test_multiple_matches(self):
        glossary = [("API", "應用程式介面"), ("cloud", "雲端"), ("dog", "狗")]
        result = match_glossary("The API runs in the cloud", glossary)
        assert ("API", "應用程式介面") in result
        assert ("cloud", "雲端") in result
        assert ("dog", "狗") not in result

    def test_empty_glossary(self):
        assert match_glossary("Hello", []) == []


class TestApplyGlossaryPre:
    def test_replaces_source_with_target(self):
        glossary = [("API", "應用程式介面")]
        result = apply_glossary_pre("The API is ready", glossary)
        assert result == "The 應用程式介面 is ready"

    def test_case_insensitive_replacement(self):
        glossary = [("API", "應用程式介面")]
        result = apply_glossary_pre("The api is ready", glossary)
        assert result == "The 應用程式介面 is ready"

    def test_multiple_replacements(self):
        glossary = [("API", "應用程式介面"), ("cloud", "雲端")]
        result = apply_glossary_pre("The API runs in the cloud", glossary)
        assert "應用程式介面" in result
        assert "雲端" in result
        assert "API" not in result
        assert "cloud" not in result

    def test_no_match_returns_original(self):
        glossary = [("API", "應用程式介面")]
        result = apply_glossary_pre("Hello world", glossary)
        assert result == "Hello world"

    def test_none_glossary_returns_original(self):
        assert apply_glossary_pre("Hello", None) == "Hello"

    def test_empty_glossary_returns_original(self):
        assert apply_glossary_pre("Hello", []) == "Hello"

    def test_respects_word_boundary(self):
        glossary = [("cat", "貓")]
        result = apply_glossary_pre("The category is set", glossary)
        assert result == "The category is set"

    def test_cjk_substring_replacement(self):
        glossary = [("翻訳", "翻譯")]
        result = apply_glossary_pre("翻訳が完了しました", glossary)
        assert "翻譯" in result
        assert "翻訳" not in result


class TestApplyGlossaryPost:
    def test_replaces_remaining_source_terms(self):
        glossary = [("API", "應用程式介面")]
        result = apply_glossary_post("API已經準備好了", glossary)
        assert "應用程式介面" in result
        assert "API" not in result

    def test_no_match_returns_original(self):
        glossary = [("API", "應用程式介面")]
        result = apply_glossary_post("翻譯完成", glossary)
        assert result == "翻譯完成"

    def test_none_glossary_returns_original(self):
        assert apply_glossary_post("Hello", None) == "Hello"

    def test_empty_glossary_returns_original(self):
        assert apply_glossary_post("Hello", []) == "Hello"
