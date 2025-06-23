import string

from functools import cached_property
from typing import List, Optional, Tuple

import tokenizers


class Tokenizer:
    """Simple wrapper around a tokenizers.Tokenizer."""

    def __init__(
        self,
        tokenizer: tokenizers.Tokenizer,
        multilingual: bool,
        task: Optional[str] = None,
        language: Optional[str] = None,
    ):
        self.tokenizer = tokenizer

        if multilingual:
            if task not in _TASKS:
                raise ValueError(
                    "'%s' is not a valid task (accepted tasks: %s)"
                    % (task, ", ".join(_TASKS))
                )

            if language not in _LANGUAGE_CODES:
                raise ValueError(
                    "'%s' is not a valid language code (accepted language codes: %s)"
                    % (language, ", ".join(_LANGUAGE_CODES))
                )

            self.task = self.tokenizer.token_to_id("<|%s|>" % task)
            self.language = self.tokenizer.token_to_id("<|%s|>" % language)
            self.language_code = language
        else:
            self.task = None
            self.language = None
            self.language_code = "en"

    @cached_property
    def transcribe(self) -> int:
        return self.tokenizer.token_to_id("<|transcribe|>")

    @cached_property
    def translate(self) -> int:
        return self.tokenizer.token_to_id("<|translate|>")

    @cached_property
    def sot(self) -> int:
        return self.tokenizer.token_to_id("<|startoftranscript|>")

    @cached_property
    def sot_lm(self) -> int:
        return self.tokenizer.token_to_id("<|startoflm|>")

    @cached_property
    def sot_prev(self) -> int:
        return self.tokenizer.token_to_id("<|startofprev|>")

    @cached_property
    def eot(self) -> int:
        return self.tokenizer.token_to_id("<|endoftext|>")

    @cached_property
    def no_timestamps(self) -> int:
        return self.tokenizer.token_to_id("<|notimestamps|>")

    @property
    def timestamp_begin(self) -> int:
        return self.no_timestamps + 1

    @property
    def sot_sequence(self) -> List[int]:
        sequence = [self.sot]

        if self.language is not None:
            sequence.append(self.language)

        if self.task is not None:
            sequence.append(self.task)

        return sequence

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens: List[int]) -> str:
        text_tokens = [token for token in tokens if token < self.eot]
        return self.tokenizer.decode(text_tokens)

    def decode_with_timestamps(self, tokens: List[int]) -> str:
        outputs = [[]]

        for token in tokens:
            if token >= self.timestamp_begin:
                timestamp = f"<|{(token - self.timestamp_begin) * 0.02:.2f}|>"
                outputs.append(timestamp)
                outputs.append([])
            else:
                outputs[-1].append(token)

        return "".join(
            [s if isinstance(s, str) else self.tokenizer.decode(s) for s in outputs]
        )

    @cached_property
    def non_speech_tokens(self) -> Tuple[int]:
        """
        Returns the list of tokens to suppress in order to avoid any speaker tags or non-speech
        annotations, to prevent sampling texts that are not actually spoken in the audio, e.g.

        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,

        keeping basic punctuations like commas, periods, question marks, exclamation points, etc.
        """
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += (
            "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
        )

        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        result = {self.encode(" -")[0], self.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [
                self.encode(symbol),
                self.encode(" " + symbol),
            ]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])

        return tuple(sorted(result))

    def split_to_word_tokens(
        self, tokens: List[int]
    ) -> Tuple[List[str], List[List[int]]]:
        if self.language_code == "th":
            # Use Thai-specific word segmentation for better word boundaries
            return self.split_to_word_tokens_thai(tokens)
        elif self.language_code in {"zh", "ja", "lo", "my", "yue"}:
            # These languages don't typically use spaces, so it is difficult to split words
            # without morpheme analysis. Here, we instead split words at any
            # position where the tokens are decoded as valid unicode points
            return self.split_tokens_on_unicode(tokens)

        return self.split_tokens_on_spaces(tokens)

    def split_tokens_on_unicode(
        self, tokens: List[int]
    ) -> Tuple[List[str], List[List[int]]]:
        decoded_full = self.decode_with_timestamps(tokens)
        replacement_char = "\ufffd"

        words = []
        word_tokens = []
        current_tokens = []
        unicode_offset = 0

        for token in tokens:
            current_tokens.append(token)
            decoded = self.decode_with_timestamps(current_tokens)

            try:
                replacement_char_index = decoded.index(replacement_char)
                replacement_char_index += unicode_offset
            except ValueError:
                replacement_char_index = None

            if replacement_char_index is None or (
                replacement_char_index < len(decoded_full)
                and decoded_full[replacement_char_index] == replacement_char
            ):
                words.append(decoded)
                word_tokens.append(current_tokens)
                current_tokens = []
                unicode_offset += len(decoded)

        return words, word_tokens

    def split_tokens_on_spaces(
        self, tokens: List[int]
    ) -> Tuple[List[str], List[List[int]]]:
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words = []
        word_tokens = []

        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            special = subword_tokens[0] >= self.eot
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or with_space or punctuation or len(words) == 0:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)

        return words, word_tokens

    def split_to_word_tokens_thai(
        self, tokens: List[int]
    ) -> Tuple[List[str], List[List[int]]]:
        """
        Split Thai tokens into proper words using PyThaiNLP while maintaining
        token-to-word mapping for timestamp alignment.
        
        Args:
            tokens: List of Whisper token IDs
            
        Returns:
            Tuple of (words, word_tokens) where:
            - words: List of Thai words as strings
            - word_tokens: List of token groups, each group corresponds to a word
        """
        try:
            from pythainlp import word_tokenize
        except ImportError:
            # Fall back to unicode splitting if pythainlp not available
            import warnings
            warnings.warn(
                "pythainlp not installed. Falling back to character-level splitting for Thai. "
                "Install with: pip install pythainlp"
            )
            return self.split_tokens_on_unicode(tokens)
        
        if not tokens:
            return [], []
        
        # Step 1: Build a character-level mapping from tokens
        # Skip special tokens like timestamps when building the text
        text_tokens = [t for t in tokens if t < self.timestamp_begin]
        special_tokens = [(i, t) for i, t in enumerate(tokens) if t >= self.timestamp_begin]
        
        if not text_tokens:
            # Only special tokens, return them as-is
            words = [self.decode_with_timestamps([t]) for _, t in special_tokens]
            word_tokens = [[t] for _, t in special_tokens]
            return words, word_tokens
        
        # Decode only text tokens to get the full text
        full_text = self.decode(text_tokens)
        
        # Build character to token index mapping
        char_to_token_idx = {}
        current_text = ""
        
        for i, token_id in enumerate(text_tokens):
            # Decode all tokens up to current one
            prev_len = len(current_text)
            current_text = self.decode(text_tokens[:i+1])
            
            # Map new characters to this token's index in the original tokens list
            original_idx = tokens.index(token_id)
            for char_idx in range(prev_len, len(current_text)):
                char_to_token_idx[char_idx] = original_idx
        
        # Step 2: Segment text into Thai words
        thai_words = word_tokenize(full_text, engine="newmm", keep_whitespace=False)
        
        # Step 3: Map Thai words back to tokens
        words = []
        word_tokens = []
        
        char_offset = 0
        for word in thai_words:
            if not word.strip():  # Skip empty words
                char_offset += len(word)
                continue
            
            word_len = len(word)
            
            # Find unique token indices for this word
            token_indices = set()
            for char_pos in range(char_offset, min(char_offset + word_len, len(full_text))):
                if char_pos in char_to_token_idx:
                    token_indices.add(char_to_token_idx[char_pos])
            
            if token_indices:
                # Sort indices and get corresponding tokens
                sorted_indices = sorted(token_indices)
                word_token_group = [tokens[i] for i in sorted_indices]
                
                words.append(word)
                word_tokens.append(word_token_group)
            
            char_offset += word_len
        
        # Step 4: Handle special tokens (timestamps, etc.)
        # Insert them back at appropriate positions
        if special_tokens:
            # For simplicity, append timestamp tokens at the end
            # In practice, you might want to insert them at their original positions
            for idx, token in special_tokens:
                words.append(self.decode_with_timestamps([token]))
                word_tokens.append([token])
        
        return words, word_tokens


_TASKS = (
    "transcribe",
    "translate",
)

_LANGUAGE_CODES = (
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "zh",
    "yue",
)
