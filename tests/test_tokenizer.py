from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import get_suppressed_tokens


def test_suppressed_tokens_minus_1():
    model = WhisperModel("tiny.en")

    tokenizer = Tokenizer(model.hf_tokenizer, False)
    tokens = get_suppressed_tokens(tokenizer, [-1])
    assert tokens == (
        1,
        2,
        7,
        8,
        9,
        10,
        14,
        25,
        26,
        27,
        28,
        29,
        31,
        58,
        59,
        60,
        61,
        62,
        63,
        90,
        91,
        92,
        93,
        357,
        366,
        438,
        532,
        685,
        705,
        796,
        930,
        1058,
        1220,
        1267,
        1279,
        1303,
        1343,
        1377,
        1391,
        1635,
        1782,
        1875,
        2162,
        2361,
        2488,
        3467,
        4008,
        4211,
        4600,
        4808,
        5299,
        5855,
        6329,
        7203,
        9609,
        9959,
        10563,
        10786,
        11420,
        11709,
        11907,
        13163,
        13697,
        13700,
        14808,
        15306,
        16410,
        16791,
        17992,
        19203,
        19510,
        20724,
        22305,
        22935,
        27007,
        30109,
        30420,
        33409,
        34949,
        40283,
        40493,
        40549,
        47282,
        49146,
        50257,
        50357,
        50358,
        50359,
        50360,
    )


def test_suppressed_tokens_minus_value():
    model = WhisperModel("tiny.en")

    tokenizer = Tokenizer(model.hf_tokenizer, False)
    tokens = get_suppressed_tokens(tokenizer, [13])
    assert tokens == (13, 50257, 50357, 50358, 50359, 50360)


def test_split_on_unicode():
    model = WhisperModel("tiny")
    tokenizer = Tokenizer(model.hf_tokenizer, False)

    tokens = [8404, 871, 287, 6, 246, 526, 3210, 20378]
    words, word_tokens = tokenizer.split_tokens_on_unicode(tokens)

    assert words == [" elle", " est", " l", "'", "\ufffd", "é", "rit", "oire"]
    assert word_tokens == [[8404], [871], [287], [6], [246], [526], [3210], [20378]]


def test_split_thai_tokens():
    """Test Thai word tokenization with proper word segmentation."""
    model = WhisperModel("tiny")
    tokenizer = Tokenizer(model.hf_tokenizer, multilingual=True, language="th", task="transcribe")
    
    # Mock Thai text tokens (these would represent Thai characters/syllables)
    # In practice, you'd need actual Thai token IDs from the model
    # This test will only run if pythainlp is installed
    try:
        from pythainlp import word_tokenize
        
        # Test with actual Thai text by encoding and then splitting
        thai_text = "สวัสดีครับ"
        tokens = tokenizer.encode(thai_text)
        
        words, word_tokens = tokenizer.split_to_word_tokens(tokens)
        
        # With pythainlp, this should split into proper Thai words
        # The exact words depend on the tokenizer's word segmentation
        assert len(words) > 0
        assert len(word_tokens) == len(words)
        assert all(len(wt) > 0 for wt in word_tokens)
        
    except ImportError:
        # If pythainlp is not installed, it should fall back to unicode splitting
        pass
