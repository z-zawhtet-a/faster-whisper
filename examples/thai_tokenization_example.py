#!/usr/bin/env python3
"""
Example demonstrating Thai word tokenization in faster-whisper.

This shows how Thai text is properly segmented into words rather than
characters when pythainlp is installed.
"""

from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer


def demonstrate_thai_tokenization():
    # Load a multilingual model
    print("Loading Whisper model...")
    model = WhisperModel("tiny", device="cpu")
    
    # Create a tokenizer for Thai language
    tokenizer = Tokenizer(
        model.hf_tokenizer,
        multilingual=True,
        task="transcribe",
        language="th"
    )
    
    # Example Thai text
    thai_text = "สวัสดีครับ วันนี้อากาศดีมาก"
    print(f"\nOriginal Thai text: {thai_text}")
    
    # Encode the text into tokens
    tokens = tokenizer.encode(thai_text)
    print(f"\nWhisper tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    
    # Decode tokens to see character-level representation
    decoded = tokenizer.decode(tokens)
    print(f"\nDecoded text: {decoded}")
    
    # Split tokens into words
    print("\n--- Word Tokenization ---")
    words, word_tokens = tokenizer.split_to_word_tokens(tokens)
    
    print(f"\nNumber of words: {len(words)}")
    print("\nWord boundaries:")
    for i, (word, tokens_for_word) in enumerate(zip(words, word_tokens)):
        print(f"  Word {i+1}: '{word}' (tokens: {tokens_for_word})")
    
    # Compare with character-level splitting
    print("\n--- Character-level splitting (fallback) ---")
    char_words, char_tokens = tokenizer.split_tokens_on_unicode(tokens)
    print(f"Number of character groups: {len(char_words)}")
    for i, (char_group, tokens_for_char) in enumerate(zip(char_words[:10], char_tokens[:10])):
        print(f"  Char group {i+1}: '{char_group}' (tokens: {tokens_for_char})")
    if len(char_words) > 10:
        print(f"  ... and {len(char_words) - 10} more character groups")


def check_pythainlp_installation():
    try:
        import pythainlp
        print("✓ pythainlp is installed")
        print(f"  Version: {pythainlp.__version__}")
        return True
    except ImportError:
        print("✗ pythainlp is not installed")
        print("  Thai text will be split at character level instead of word level")
        print("  Install with: pip install pythainlp")
        return False


if __name__ == "__main__":
    print("Thai Word Tokenization Example")
    print("=" * 40)
    
    # Check if pythainlp is available
    has_pythainlp = check_pythainlp_installation()
    
    # Run demonstration
    try:
        demonstrate_thai_tokenization()
        
        if has_pythainlp:
            print("\n✓ Thai words are properly segmented using pythainlp")
        else:
            print("\n⚠ Thai text was split at character level (install pythainlp for word-level segmentation)")
            
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have a multilingual Whisper model downloaded.")