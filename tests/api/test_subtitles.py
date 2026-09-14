from whisperx.api.audio.engine import WhisperXEngine
from whisperx.api.audio.renderers import render_result
from whisperx.api.audio.schemas import AudioTask, ResponseFormat


def result_for(text):
    words = [{"word": char, "start": i * 0.1, "end": (i + 1) * 0.1} for i, char in enumerate(text)]
    return WhisperXEngine._normalize_result(
        {
            "language": "zh",
            "segments": [{"start": 0, "end": len(text) * 0.1, "text": text, "words": words}],
        },
        task=AudioTask.TRANSCRIBE,
        duration=len(text) * 0.1,
    )


def test_subtitles_split_at_punctuation_not_character_limit():
    text = (
        "大家好，我是懂险帝。两亿港元加上五千万港元，"
        "真正难的不是怎么投，而是别把架构、税务和保单功能混成一件事。"
    )
    result = result_for(text)
    content, _ = render_result(result, ResponseFormat.SRT, max_line_width=30, max_line_count=1)
    blocks = content.decode().strip().split("\n\n")
    lines = ["".join(block.splitlines()[2:]) for block in blocks]
    assert lines == [
        "大家好",
        "我是懂险帝",
        "两亿港元加上五千万港元",
        "真正难的不是怎么投",
        "而是别把架构、税务和保单功能混成一件事",
    ]
    assert "架构、税务" in lines[-1]
    assert result.segments[0].text == text


def test_subtitles_preserve_decimal_and_closing_quotes():
    text = "收益是3.14%，他说“可以。”下一句。"
    content, _ = render_result(
        result_for(text), ResponseFormat.VTT, max_line_width=8, max_line_count=1
    )
    lines = [block.splitlines()[-1] for block in content.decode().strip().split("\n\n")[1:]]
    assert lines == ["收益是3.14%", "他说“可以”", "下一句"]


def test_translation_joins_english_output_with_spaces():
    raw = {
        "language": "zh",
        "segments": [
            {"start": 0, "end": 1, "text": " One thing."},
            {"start": 1, "end": 2, "text": " A few days ago."},
        ],
    }
    result = WhisperXEngine._normalize_result(raw, task=AudioTask.TRANSLATE, duration=2)
    assert result.language == "zh"
    assert result.text == "One thing. A few days ago."
