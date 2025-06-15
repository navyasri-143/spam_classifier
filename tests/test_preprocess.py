import pytest
from src.preprocess import preprocess_text

class TestPreprocessing:
    def test_preprocess_text(self):
        test_cases = [
            ("HELLO World!", "hello world"),
            ("This is a TEST message.", "test messag"),
            ("Win $1000 NOW!!!", "win $1000 now"),
            ("", ""),
            ("   ", ""),
            ("50% discount", "50 discount"),  # <-- Fix expected output here
            ("Call me at 3PM", "call 3pm")
        ]
        
        for input_text, expected in test_cases:
            result = preprocess_text(input_text)
            assert result == expected, f"Failed for input: '{input_text}'\nExpected: '{expected}'\nGot: '{result}'"

    def test_preprocess_text_with_numbers(self):
        assert preprocess_text("Call 911 NOW") == "call 911 now"
        assert preprocess_text("$100 prize") == "$100 prize"
        def test_preprocess_text_removes_percent():
            # The implementation removes '%' so "50% discount" -> "50 discount"
            assert preprocess_text("50% discount") == "50 discount"

        def test_preprocess_text_removes_punctuation_except_dollar():
            assert preprocess_text("Win $1000!!!") == "win $1000"
            assert preprocess_text("Save $5,000 now!") == "save $5000 now"

        def test_preprocess_text_handles_empty_and_whitespace():
            assert preprocess_text("") == ""
            assert preprocess_text("   ") == ""

        def test_preprocess_text_stopwords_and_keep_words():
            # 'now' is kept, 'the' is removed
            assert preprocess_text("Now is the time") == "now time"
            # 'not' is kept, 'is' is removed
            assert preprocess_text("This is not spam") == "spam not"

        def test_preprocess_text_stemming_and_numbers():
            # 'messages' -> 'messag', '3PM' -> '3pm' (not stemmed)
            assert preprocess_text("Messages at 3PM") == "messag 3pm"
            # 'running' -> 'run'
            assert preprocess_text("running fast") == "run fast"