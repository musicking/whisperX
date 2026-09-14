import pytest

from whisperx.api.audio.document import match_document
from whisperx.api.audio.schemas import AlignmentSegment
from whisperx.api.exceptions import UnsupportedOption


def segment(text, start=0, end=1):
    return AlignmentSegment(text=text, start=start, end=end)


def test_original_punctuation_and_paragraphs_are_preserved():
    original = "你好！\n今天市场上涨。"
    result = match_document(original, [segment("你好"), segment("今天市场上涨", 1, 2)])
    assert [item.text for item in result] == ["你好！\n", "今天市场上涨。"]
    assert [(item.start, item.end) for item in result] == [(0, 1), (1, 2)]
    assert "".join(item.text for item in result) == original


def test_case_and_fullwidth_characters_only_affect_matching():
    original = "The Ｓ＆Ｐ５００ rose."
    result = match_document(original, [segment("the sp500 rose")])
    assert result[0].text == original


def test_small_asr_error_does_not_replace_original_text():
    original = "今天纳斯达克指数上涨。"
    result = match_document(original, [segment("今天纳斯达刻指数上涨")])
    assert result[0].text == original


@pytest.mark.parametrize("original", ["", " \n ", "！？", "完全无关的旁白内容"])
def test_unusable_document_is_rejected(original):
    with pytest.raises(UnsupportedOption):
        match_document(original, [segment("你好")])


def test_unmatched_audio_segment_is_rejected():
    with pytest.raises(UnsupportedOption):
        match_document("今天市场上涨明天市场下跌", [segment("今天市场上涨"), segment("无关内容")])


def test_repeated_sentences_keep_their_order():
    result = match_document("你好！你好？", [segment("你好"), segment("你好", 1, 2)])
    assert [item.text for item in result] == ["你好！", "你好？"]


def test_large_unspoken_document_passage_is_rejected():
    with pytest.raises(UnsupportedOption):
        match_document("今天市场上涨。这里有一大段没有朗读的原稿。", [segment("今天市场上涨")])
