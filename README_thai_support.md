# Thai Language Support in faster-whisper

This document explains the enhanced Thai language support in faster-whisper, which provides proper word-level tokenization for Thai text.

## Background

By default, Whisper's tokenizer creates character-level or sub-word tokens for Thai text because Thai doesn't use spaces to separate words. This results in:
- Word timestamps being assigned to individual characters rather than complete words
- Unnatural word boundaries in transcriptions
- Less meaningful timestamp alignments

## Solution

We've integrated Thai word segmentation using PyThaiNLP's newmm engine, which provides linguistically accurate word boundaries.

## Installation

To enable Thai word segmentation, install faster-whisper with the Thai language support:

```bash
pip install faster-whisper[thai]
```

Or install PyThaiNLP separately:

```bash
pip install pythainlp
```

## How It Works

When transcribing Thai audio with `word_timestamps=True`:

1. **Without PyThaiNLP** (default behavior):
   - Text: "สวัสดีครับ" (hello)
   - Words: ["ส", "วัส", "ดี", "ค", "รับ"] (character groups)
   - Each character group gets its own timestamp

2. **With PyThaiNLP** (enhanced behavior):
   - Text: "สวัสดีครับ" (hello)  
   - Words: ["สวัสดี", "ครับ"] (proper words)
   - Timestamps align with actual word boundaries

## Usage Example

```python
from faster_whisper import WhisperModel

# Load model
model = WhisperModel("medium", device="cuda")

# Transcribe Thai audio with word timestamps
segments, info = model.transcribe(
    "thai_audio.wav",
    language="th",
    word_timestamps=True
)

# Print words with timestamps
for segment in segments:
    for word in segment.words:
        print(f"{word.start:.2f}s - {word.end:.2f}s: {word.word}")
```

## Technical Details

The implementation:
1. Detects when the language is Thai (`language_code == "th"`)
2. Uses PyThaiNLP to segment the decoded text into proper words
3. Maps Whisper's character-level tokens back to Thai words
4. Groups tokens belonging to the same word together
5. Maintains compatibility with the existing timestamp alignment system

## Fallback Behavior

If PyThaiNLP is not installed:
- A warning is displayed
- The system falls back to character-level splitting
- Word timestamps still work but represent character boundaries

## Benefits

- **Accurate Word Boundaries**: Timestamps align with linguistically correct Thai words
- **Better Readability**: Transcriptions show complete words instead of character fragments
- **Improved Applications**: Enables better subtitle generation, word highlighting, and text analysis
- **Backward Compatible**: Works with existing code, just provides better word segmentation

## Performance Considerations

- PyThaiNLP word segmentation adds minimal overhead
- The character-to-token mapping is computed efficiently
- Overall transcription speed is not significantly affected

## Limitations

- Timestamp accuracy depends on the quality of Whisper's attention alignment
- Complex compound words might still be segmented differently than expected
- Special tokens (timestamps, punctuation) are handled separately from Thai words