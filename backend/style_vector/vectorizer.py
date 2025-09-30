# backend/style_vector/vectorizer.py

import numpy as np
from typing import List, Set

# This file provides a lightweight, dependency-free implementation for converting
# a text string into the 8-dimensional style vector used in the paper.
# It is designed for reproducibility of the external dataset simulations,
# approximating the core features from the full NLP pipeline without requiring
# heavy models. Note that some features (Sentiment, Empath) are set to zero
# to maintain this lightweight nature, matching the logic used for the paper's
# cross-corpus replications.

# The 8 features, in the canonical order required by the scaler.
STYLE_VECTOR_FEATURES: List[str] = [
    'informality_score_model',
    'sentiment_compound',
    'avg_sentence_length',
    'flesch_reading_ease',
    'empath_social',
    'empath_cognitive',
    'empath_affect',
    'function_word_ratio'
]

# A standard set of function words for calculating the ratio.
FUNCTION_WORDS: Set[str] = {
    "i","you","we","he","she","they","me","him","her","us","them", "a","an","the",
    "in","on","at","by","for","with","about","against","between","into","through",
    "during","before","after","above","below","to","from","up","down","over",
    "under","am","is","are","was","were","be","being","been","have","has","had",
    "having","do","does","did","will","would","shall","should","may","might",
    "must","can","could","and","but","or","so","yet","for","nor","while",
    "whereas","although","though","because","since","if","unless","until",
    "when","as","that","whether","no","not","never","none","n't"
}

# --- Internal Helper Functions ---

def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
    return [token for token in text.replace("\n", " ").split() if token]

def _sentences(text: str) -> List[str]:
    """Simple sentence splitter based on terminal punctuation."""
    out, current_sentence = [], []
    for char in text:
        current_sentence.append(char)
        if char in ".!?":
            out.append("".join(current_sentence).strip())
            current_sentence = []
    if current_sentence:
        out.append("".join(current_sentence).strip())
    return [s for s in out if s]

def _syllable_like_count(token: str) -> int:
    """Approximates syllable count based on vowel clusters."""
    token = token.lower()
    count = 0
    vowels = "aeiouy"
    is_prev_char_vowel = False
    for char in token:
        is_char_vowel = char in vowels
        if is_char_vowel and not is_prev_char_vowel:
            count += 1
        is_prev_char_vowel = is_char_vowel
    return max(1, count)

# --- Public Vectorization Function ---

def vectorize_text(text: str) -> np.ndarray:
    """
    Transforms a text string into its 8-dimensional style vector representation.

    This function is a lightweight, self-contained version of the full NLP pipeline,
    intended for easy reproducibility of the external dataset simulations.

    Args:
        text: The input text utterance.

    Returns:
        A NumPy array representing the 8D style vector. The features are
        in the canonical order defined by STYLE_VECTOR_FEATURES.
    """
    tokens = _tokenize(text)
    num_words = len(tokens)
    
    sentences = _sentences(text)
    num_sentences = max(1, len(sentences))

    # Feature 1: Informality (approximated)
    # A simple heuristic combining punctuation and pronoun usage.
    exclamations_and_questions = text.count("!") + text.count("?")
    first_second_person_pronouns = sum(t.lower() in {"i","you","u","im","i'm","you're"} for t in tokens)
    informality = (exclamations_and_questions / num_sentences) + (first_second_person_pronouns / max(1, num_words))

    # Feature 2: Sentiment (placeholder)
    # Set to a neutral 0.0 to avoid dependency on NLTK/VADER for reproducibility.
    sentiment = 0.0

    # Feature 3: Average Sentence Length
    avg_sentence_len = num_words / num_sentences

    # Feature 4: Readability (Flesch Reading Ease, approximated)
    # Calculated using a local syllable counter to avoid dependency on textstat.
    if num_words == 0:
        flesch = 0.0
    else:
        approx_syllables = sum(_syllable_like_count(t) for t in tokens)
        syllables_per_word = approx_syllables / num_words
        flesch = 206.835 - 1.015 * avg_sentence_len - 84.6 * syllables_per_word

    # Features 5, 6, 7: Empath Categories (placeholders)
    # Set to 0.0 to avoid dependency on the Empath lexicon.
    empath_social = 0.0
    empath_cognitive = 0.0
    empath_affect = 0.0

    # Feature 8: Function Word Ratio
    function_word_count = sum(1 for t in tokens if t.lower() in FUNCTION_WORDS)
    function_word_ratio = function_word_count / max(1, num_words)

    # Return the vector in the canonical order.
    return np.array([
        informality,
        sentiment,
        avg_sentence_len,
        flesch,
        empath_social,
        empath_cognitive,
        empath_affect,
        function_word_ratio
    ], dtype=float)