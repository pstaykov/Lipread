"""
Beam Search Decoder with Trigram Language Model and Adaptive Thresholding
for Multimodal Lipreading.

This implementation provides beam search decoding that combines multimodal model
probabilities with trigram language model scores, featuring an adaptive threshold
mechanism that falls back to multimodal probabilities when the language model
is uncertain about bigram predictions.

Key Features:
- Standard beam search with configurable beam width
- Trigram language model with add-one (Laplace) smoothing
- Adaptive thresholding: when bigram LM probability falls below threshold,
  rely more on multimodal model probabilities
- Length normalization option
- Support for start/end of sequence tokens
- Efficient top-k expansion for large vocabularies
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, Counter
import math
import heapq
from typing import List, Tuple, Dict, Optional


class TrigramLanguageModel:
    """
    Trigram language model with add-one (Laplace) smoothing.
    Estimates P(w_i | w_{i-2}, w_{i-1}) for word sequences.
    Includes methods to get probabilities with fallback to lower-order n-grams.
    """

    def __init__(self, vocab_size: int, pad_token: int = 0, unk_token: int = 1):
        """
        Initialize trigram language model.

        Args:
            vocab_size: Size of the vocabulary
            pad_token: Index for padding token
            unk_token: Index for unknown token
        """
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token

        # Counts for trigram, bigram, and unigram
        self.trigram_counts = defaultdict(Counter)  # (w_{i-2}, w_{i-1}) -> Counter of w_i
        self.bigram_counts = defaultdict(Counter)   # w_{i-1} -> Counter of w_i
        self.unigram_counts = Counter()             # w_i -> count

        # Total tokens for normalization
        self.total_tokens = 0

    def train(self, tokenized_sentences: List[List[int]]):
        """
        Train the trigram language model on tokenized sentences.

        Args:
            tokenized_sentences: List of sentences, each as list of token indices
        """
        for sentence in tokenized_sentences:
            # Add padding tokens at beginning for context
            padded_sentence = [self.pad_token, self.pad_token] + sentence

            # Count unigrams, bigrams, trigrams
            for i in range(len(padded_sentence)):
                # Unigram count
                token = padded_sentence[i]
                self.unigram_counts[token] += 1
                self.total_tokens += 1

                # Bigram count (if we have previous token)
                if i >= 1:
                    w_prev = padded_sentence[i-1]
                    self.bigram_counts[w_prev][token] += 1

                # Trigram count (if we have two previous tokens)
                if i >= 2:
                    w_prev2 = padded_sentence[i-2]
                    w_prev1 = padded_sentence[i-1]
                    self.trigram_counts[(w_prev2, w_prev1)][token] += 1

    def get_trigram_prob(self, w_i_minus_2: int, w_i_minus_1: int, w_i: int) -> float:
        """
        Get trigram probability P(w_i | w_{i-2}, w_{i-1}) with add-one smoothing.

        Returns:
            Probability as float in [0, 1]
        """
        # Handle padding tokens - if either context token is pad, use lower-order LM
        if w_i_minus_2 == self.pad_token or w_i_minus_1 == self.pad_token:
            # Fall back to bigram or unigram
            return self.get_bigram_prob(w_i_minus_1, w_i)

        # Get counts
        context = (w_i_minus_2, w_i_minus_1)
        trigram_count = self.trigram_counts[context].get(w_i, 0)
        bigram_count = sum(self.trigram_counts[context].values())  # Total count of this bigram context

        # Add-one smoothing: P(w_i|context) = (count(w_i, context) + 1) / (total_context_count + vocab_size)
        if bigram_count == 0:
            # Unseen bigram context - fall back to unigram
            return self.get_unigram_prob(w_i)

        return (trigram_count + 1) / (bigram_count + self.vocab_size)

    def get_bigram_prob(self, w_i_minus_1: int, w_i: int) -> float:
        """Get bigram probability with add-one smoothing."""
        if w_i_minus_1 == self.pad_token:
            # Fall back to unigram
            return self.get_unigram_prob(w_i)

        bigram_count = self.bigram_counts[w_i_minus_1].get(w_i, 0)
        context_count = sum(self.bigram_counts[w_i_minus_1].values())  # Total count of w_{i-1}

        if context_count == 0:
            return self.get_unigram_prob(w_i)

        return (bigram_count + 1) / (context_count + self.vocab_size)

    def get_unigram_prob(self, w_i: int) -> float:
        """Get unigram probability with add-one smoothing."""
        unigram_count = self.unigram_counts.get(w_i, 0)
        return (unigram_count + 1) / (self.total_tokens + self.vocab_size)

    def get_log_prob(self, w_i_minus_2: int, w_i_minus_1: int, w_i: int) -> float:
        """Get log trigram probability (returns -inf if probability is 0)."""
        prob = self.get_trigram_prob(w_i_minus_2, w_i_minus_1, w_i)
        if prob <= 0:
            return float('-inf')
        return math.log(prob)

    def get_prob(self, w_i_minus_2: int, w_i_minus_1: int, w_i: int) -> float:
        """Get probability (not log) for thresholding decisions."""
        return self.get_trigram_prob(w_i_minus_2, w_i_minus_1, w_i)


class BeamSearchNode:
    """Node in the beam search tree representing a partial hypothesis."""

    def __init__(self,
                 tokens: List[int],
                 log_prob: float,
                 lm_log_prob: float,
                 multimodal_log_prob: float):
        """
        Initialize beam search node.

        Args:
            tokens: List of token indices in the hypothesis
            log_prob: Total log probability (multimodal + LM)
            lm_log_prob: Log probability from language model only
            multimodal_log_prob: Log probability from multimodal model only
        """
        self.tokens = tokens
        self.log_prob = log_prob
        self.lm_log_prob = lm_log_prob
        self.multimodal_log_prob = multimodal_log_prob
        self.length = len(tokens)

    def extend(self, token: int, log_prob_increment: float, lm_log_prob_increment: float) -> 'BeamSearchNode':
        """Create extended node with additional token."""
        new_tokens = self.tokens + [token]
        new_log_prob = self.log_prob + log_prob_increment
        new_lm_log_prob = self.lm_log_prob + lm_log_prob_increment
        new_mm_log_prob = self.multimodal_log_prob + log_prob_increment  # LM part is separate

        return BeamSearchNode(new_tokens, new_log_prob, new_lm_log_prob, new_mm_log_prob)

    def __lt__(self, other):
        """For priority queue ordering - higher log prob first."""
        return self.log_prob > other.log_prob  # Reverse for max-heap behavior

    def __eq__(self, other):
        return isinstance(other, BeamSearchNode) and self.tokens == other.tokens


class BeamSearchDecoder:
    """
    Beam search decoder for multimodal lipreading with trigram LM rescoring and adaptive thresholding.

    Implements the beam search algorithm described in the requirement:
    - For each word position, maintains several partial sentence hypotheses (beam width)
    - For each hypothesis, expands with candidate words from vocabulary
    - Scores expansions using: log P_multimodal + weight × log P_trigram
    - Applies adaptive thresholding: if bigram LM probability < threshold,
      reduces reliance on LM and increases reliance on multimodal probabilities
    - Keeps only highest-scoring beams after each expansion step
    """

    def __init__(self,
                 vocab_size: int,
                 beam_width: int = 5,
                 lm_weight: float = 0.7,
                 length_penalty: float = 0.0,
                 pad_token: int = 0,
                 unk_token: int = 1,
                 sos_token: Optional[int] = None,
                 eos_token: Optional[int] = None,
                 bigram_threshold: float = 0.01):
        """
        Initialize beam search decoder with adaptive thresholding.

        Args:
            vocab_size: Size of the vocabulary (e.g., 500 for GLipsNet)
            beam_width: Number of beams to keep at each step (default: 5 as specified)
            lm_weight: Base weight for LM score in combination [0,1]
                      (0 = only multimodal, 1 = only LM)
            length_penalty: Length penalty factor (0 = no penalty, >0 penalizes long hypotheses)
            pad_token: Padding token index (default: 0)
            unk_token: Unknown token index (default: 1)
            sos_token: Start-of-sequence token index (if None, uses pad_token)
            eos_token: End-of-sequence token index (if None, no EOS handling)
            bigram_threshold: Threshold for bigram LM confidence. If bigram probability
                            < this threshold, reduce LM reliance and increase multimodal reliance.
                            Default: 0.01 (1% probability threshold)
        """
        self.vocab_size = vocab_size
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.length_penalty = length_penalty
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token if sos_token is not None else pad_token
        self.eos_token = eos_token
        self.bigram_threshold = bigram_threshold

        # Initialize language model
        self.lm = TrigramLanguageModel(vocab_size, pad_token, unk_token)

    def train_language_model(self, tokenized_sentences: List[List[int]]):
        """Train the trigram language model on provided text data."""
        self.lm.train(tokenized_sentences)

    def _get_adaptive_lm_weight(self, base_lm_weight: float, bigram_prob: float) -> float:
        """
        Calculate adaptive LM weight based on bigram probability confidence.

        When bigram probability is high (>= threshold): use base_lm_weight
        When bigram probability is low (< threshold): reduce LM weight,
        increase reliance on multimodal probabilities

        Args:
            base_lm_weight: Base weight for LM score
            bigram_prob: Probability of the bigram prediction

        Returns:
            Adaptive LM weight for this prediction
        """
        if bigram_prob >= self.bigram_threshold:
            # Confident in bigram prediction - use base weight
            return base_lm_weight
        else:
            # Not confident in bigram prediction - reduce LM weight
            # Linear decay: when prob=0, weight=0; when prob=threshold, weight=base_lm_weight
            if self.bigram_threshold > 0:
                adaptive_weight = base_lm_weight * (bigram_prob / self.bigram_threshold)
                # Ensure we don't go below 0 or above base_lm_weight
                return max(0.0, min(adaptive_weight, base_lm_weight))
            else:
                return 0.0

    def decode(self,
               multimodal_logits: torch.Tensor,
               return_top_k: int = 1) -> List[List[int]]:
        """
        Perform beam search decoding with adaptive thresholding.

        Args:
            multimodal_logits: Tensor of shape (seq_len, vocab_size) containing logits
                             from the multimodal model for each time step
            return_top_k: Number of top hypotheses to return

        Returns:
            List of token sequences (hypotheses), each as list of token indices
        """
        seq_len = multimodal_logits.shape[0]

        # Convert logits to log probabilities
        log_probs = torch.log_softmax(multimodal_logits, dim=-1)  # (seq_len, vocab_size)

        # Initialize beam with start-of-sequence token
        initial_node = BeamSearchNode(
            tokens=[self.sos_token],
            log_prob=0.0,
            lm_log_prob=0.0,
            multimodal_log_prob=0.0
        )

        beams = [initial_node]
        completed_hypotheses = []

        # Iterate over each time step (word position)
        for t in range(seq_len):
            next_beams = []

            # For each current beam hypothesis, expand with candidate words
            for beam in beams:
                # If beam already ended with EOS, keep it as completed hypothesis
                if beam.tokens[-1] == self.eos_token and self.eos_token is not None:
                    completed_hypotheses.append(beam)
                    continue

                # Get log probabilities for next token from multimodal model
                token_log_probs = log_probs[t]  # (vocab_size,)

                # Consider top-k tokens for efficiency (can be adjusted)
                top_k = min(50, self.vocab_size)  # Consider top 50 tokens for efficiency
                top_log_probs, top_indices = torch.topk(token_log_probs, top_k)

                # Expand beam with each candidate token
                for i in range(top_k):
                    token_idx = top_indices[i].item()
                    mm_log_prob = top_log_probs[i].item()  # Multimodal log prob

                    # Get language model context (last two tokens in hypothesis)
                    if len(beam.tokens) >= 2:
                        w_i_minus_2 = beam.tokens[-2]
                        w_i_minus_1 = beam.tokens[-1]
                    elif len(beam.tokens) == 1:
                        w_i_minus_2 = self.pad_token
                        w_i_minus_1 = beam.tokens[-1]
                    else:  # len == 0 (shouldn't happen with SOS)
                        w_i_minus_2 = self.pad_token
                        w_i_minus_1 = self.pad_token

                    # Get LM probability and log probability
                    lm_log_prob = self.lm.get_log_prob(w_i_minus_2, w_i_minus_1, token_idx)
                    bigram_prob = self.lm.get_prob(w_i_minus_2, w_i_minus_1, token_idx)

                    # Calculate adaptive LM weight based on bigram confidence
                    adaptive_lm_weight = self._get_adaptive_lm_weight(self.lm_weight, bigram_prob)

                    # Handle case where LM probability is zero (log prob = -inf)
                    if lm_log_prob == float('-inf'):
                        # LM gives zero probability - rely only on multimodal model
                        combined_log_prob = mm_log_prob
                        effective_lm_weight = 0.0
                    else:
                        # Combine multimodal and LM scores with adaptive weighting
                        combined_log_prob = mm_log_prob + adaptive_lm_weight * lm_log_prob
                        effective_lm_weight = adaptive_lm_weight

                    # Apply length penalty if configured
                    if self.length_penalty > 0:
                        length_penalty_factor = ((5 + (len(beam.tokens) + 1)) / 6) ** self.length_penalty
                        combined_log_prob /= length_penalty_factor

                    # Create extended beam hypothesis
                    extended_beam = beam.extend(token_idx, mm_log_prob, lm_log_prob)
                    extended_beam.log_prob = combined_log_prob  # Override with combined score
                    # Store effective weights for analysis (optional)
                    extended_beam.effective_lm_weight = effective_lm_weight

                    next_beams.append(extended_beam)

            # Keep only the top beam_width hypotheses for next iteration
            beams = sorted(next_beams, key=lambda x: x.log_prob, reverse=True)[:self.beam_width]

            # Early stopping optimization: if all beams have ended with EOS
            if self.eos_token is not None:
                all_ended = all(beam.tokens[-1] == self.eos_token for beam in beams)
                if all_ended:
                    completed_hypotheses.extend(beams)
                    break

        # Add remaining beams to completed hypotheses
        completed_hypotheses.extend(beams)

        # Sort completed hypotheses by score (with length normalization)
        def score_with_length_penalty(hypothesis):
            score = hypothesis.log_prob
            if self.length_penalty > 0 and hypothesis.length > 0:
                length_penalty = ((5 + hypothesis.length) / 6) ** self.length_penalty
                score /= length_penalty
            return score

        completed_hypotheses = sorted(completed_hypotheses,
                                    key=score_with_length_penalty,
                                    reverse=True)

        # Return top-k hypotheses (excluding SOS token)
        results = []
        for i in range(min(return_top_k, len(completed_hypotheses))):
            hypothesis = completed_hypotheses[i]
            # Remove SOS token if present
            tokens = hypothesis.tokens[1:] if hypothesis.tokens[0] == self.sos_token else hypothesis.tokens
            results.append(tokens)

        return results

    def get_hypothesis_scores(self,
                            multimodal_logits: torch.Tensor) -> List[Tuple[List[int], float, float, float, float]]:
        """
        Get detailed scores for top hypotheses including adaptive LM weights.

        Returns:
            List of tuples: (tokens, total_log_prob, lm_log_prob, multimodal_log_prob, avg_lm_weight)
        """
        seq_len = multimodal_logits.shape[0]
        log_probs = torch.log_softmax(multimodal_logits, dim=-1)

        initial_node = BeamSearchNode(
            tokens=[self.sos_token],
            log_prob=0.0,
            lm_log_prob=0.0,
            multimodal_log_prob=0.0
        )

        beams = [initial_node]
        completed_hypotheses = []

        for t in range(seq_len):
            next_beams = []

            for beam in beams:
                if beam.tokens[-1] == self.eos_token and self.eos_token is not None:
                    completed_hypotheses.append(beam)
                    continue

                token_log_probs = log_probs[t]
                top_k = min(50, self.vocab_size)
                top_log_probs, top_indices = torch.topk(token_log_probs, top_k)

                for i in range(top_k):
                    token_idx = top_indices[i].item()
                    mm_log_prob = top_log_probs[i].item()

                    # LM context
                    if len(beam.tokens) >= 2:
                        w_i_minus_2 = beam.tokens[-2]
                        w_i_minus_1 = beam.tokens[-1]
                    elif len(beam.tokens) == 1:
                        w_i_minus_2 = self.pad_token
                        w_i_minus_1 = beam.tokens[-1]
                    else:
                        w_i_minus_2 = self.pad_token
                        w_i_minus_1 = self.pad_token

                    lm_log_prob = self.lm.get_log_prob(w_i_minus_2, w_i_minus_1, token_idx)
                    bigram_prob = self.lm.get_prob(w_i_minus_2, w_i_minus_1, token_idx)

                    # Calculate adaptive LM weight
                    adaptive_lm_weight = self._get_adaptive_lm_weight(self.lm_weight, bigram_prob)

                    if lm_log_prob == float('-inf'):
                        combined_log_prob = mm_log_prob
                        effective_lm_weight = 0.0
                    else:
                        combined_log_prob = mm_log_prob + adaptive_lm_weight * lm_log_prob
                        effective_lm_weight = adaptive_lm_weight

                    if self.length_penalty > 0:
                        length_penalty_factor = ((5 + (len(beam.tokens) + 1)) / 6) ** self.length_penalty
                        combined_log_prob /= length_penalty_factor

                    extended_beam = beam.extend(token_idx, mm_log_prob, lm_log_prob)
                    extended_beam.log_prob = combined_log_prob
                    extended_beam.effective_lm_weight = effective_lm_weight

                    next_beams.append(extended_beam)

            beams = sorted(next_beams, key=lambda x: x.log_prob, reverse=True)[:self.beam_width]

            if self.eos_token is not None:
                all_ended = all(beam.tokens[-1] == self.eos_token for beam in beams)
                if all_ended:
                    completed_hypotheses.extend(beams)
                    break

        completed_hypotheses.extend(beams)

        def score_with_length_penalty(hypothesis):
            score = hypothesis.log_prob
            if self.length_penalty > 0 and hypothesis.length > 0:
                length_penalty = ((5 + hypothesis.length) / 6) ** self.length_penalty
                score /= length_penalty
            return score

        completed_hypotheses = sorted(completed_hypotheses,
                                    key=score_with_length_penalty,
                                    reverse=True)

        results = []
        for hypothesis in completed_hypotheses[:self.beam_width]:
            tokens = hypothesis.tokens[1:] if hypothesis.tokens[0] == self.sos_token else hypothesis.tokens
            avg_lm_weight = getattr(hypothesis, 'effective_lm_weight', self.lm_weight)
            results.append((tokens, hypothesis.log_prob, hypothesis.lm_log_prob,
                          hypothesis.multimodal_log_prob, avg_lm_weight))

        return results


def create_dummy_text_corpus(vocab_size: int, num_sentences: int = 1000,
                           max_sentence_length: int = 10) -> List[List[int]]:
    """
    Create a dummy text corpus for training the language model when no real text is available.
    This generates random sentences for demonstration purposes.

    In a real implementation, you would load actual text data corresponding to your vocabulary.

    Args:
        vocab_size: Size of the vocabulary
        num_sentences: Number of sentences to generate
        max_sentence_length: Maximum length of each sentence

    Returns:
        List of sentences, each as list of token indices
    """
    # Avoid special tokens (0=pad, 1=unk) for random words
    word_range = list(range(2, vocab_size))

    sentences = []
    for _ in range(num_sentences):
        # Random sentence length between 2 and max_sentence_length
        length = np.random.randint(2, max_sentence_length + 1)
        sentence = np.random.choice(word_range, size=length, replace=False).tolist()
        sentences.append(sentence)

    return sentences


# Example usage function demonstrating the adaptive threshold feature
def example_usage_with_threshold():
    """Example showing how the adaptive threshold works with uncertain bigrams."""
    print("=" * 60)
    print("BEAM SEARCH DECODER WITH ADAPTIVE THRESHOLDING - EXAMPLE")
    print("=" * 60)

    # Parameters
    VOCAB_SIZE = 500
    SEQ_LENGTH = 6
    BEAM_WIDTH = 5
    LM_WEIGHT = 0.7
    BIGRAM_THRESHOLD = 0.05  # 5% threshold for bigram confidence

    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Sequence length: {SEQ_LENGTH} words")
    print(f"Beam width: {BEAM_WIDTH}")
    print(f"Base LM weight: {LM_WEIGHT}")
    print(f"Bigram threshold: {BIGRAM_THRESHOLD}")
    print()

    # Create beam search decoder with thresholding
    decoder = BeamSearchDecoder(
        vocab_size=VOCAB_SIZE,
        beam_width=BEAM_WIDTH,
        lm_weight=LM_WEIGHT,
        length_penalty=0.1,
        pad_token=0,
        unk_token=1,
        sos_token=0,
        eos_token=2,
        bigram_threshold=BIGRAM_THRESHOLD
    )

    # Create a training corpus that teaches certain bigram patterns
    # but leaves some bigrams uncertain/unseen
    training_corpus = [
        [1, 2, 3, 4],  # Common pattern: A B C D
        [1, 2, 3, 5],  # Common pattern: A B C E
        [6, 7, 8, 9],  # Another pattern: F G H I
        [6, 7, 8, 10], # Another pattern: F G H J
    ] * 20  # Repeat for sufficient training

    # Add some variety
    import random
    for _ in range(300):
        length = random.randint(2, 5)
        sentence = [random.randint(1, 20) for _ in range(length)]
        training_corpus.append(sentence)

    decoder.train_language_model(training_corpus)
    print(f"Trained language model on {len(training_corpus)} sentences")
    print()

    # Define a test sequence with both certain and uncertain bigrams
    # Certain: positions where bigrams were seen in training
    # Uncertain: positions with novel bigrams not seen in training
    true_sentence_ids = [1, 2, 3, 4, 6, 7]  # "A B C D F G"
    print(f"True sentence (word IDs): {true_sentence_ids}")
    print("  Positions 0-3: Certain bigrams (seen in training)")
    print("  Positions 4-5: Uncertain bigrams (novel combination not in training)")
    print()

    # Generate multimodal logits with varying confidence
    torch.manual_seed(42)
    multimodal_logits_list = []

    # Position 0: Start token or first word - somewhat confident
    logits0 = torch.randn(VOCAB_SIZE)
    logits0[1] += 1.5  # Correct word "A"
    logits0[10] += 0.5  # Alternative
    multimodal_logits_list.append(logits0)

    # Position 1: Second word - quite certain (bigram 1->2 was in training)
    logits1 = torch.randn(VOCAB_SIZE)
    logits1[2] += 2.0  # Correct word "B"
    logits1[11] += 0.3  # Alternative
    multimodal_logits_list.append(logits1)

    # Position 2: Third word - quite certain (bigram 2->3 was in training)
    logits2 = torch.randn(VOCAB_SIZE)
    logits2[3] += 2.0  # Correct word "C"
    logits2[12] += 0.3  # Alternative
    multimodal_logits_list.append(logits2)

    # Position 3: Fourth word - uncertain bigram (3->4 was in training but let's make multimodal weak)
    logits3 = torch.randn(VOCAB_SIZE)
    logits3[4] += 0.5  # Correct word "D" - weak multimodal signal
    logits3[13] += 1.5  # Strong alternative (this creates ambiguity)
    multimodal_logits_list.append(logits3)

    # Position 4: Fifth word - uncertain bigram (4->6 not in training)
    logits4 = torch.randn(VOCAB_SIZE)
    logits4[6] += 1.5  # Correct word "F"
    logits4[14] += 0.5  # Alternative
    multimodal_logits_list.append(logits4)

    # Position 5: Sixth word - certain bigram (6->7 was in training)
    logits5 = torch.randn(VOCAB_SIZE)
    logits5[7] += 2.0  # Correct word "G"
    logits5[15] += 0.3  # Alternative
    multimodal_logits_list.append(logits5)

    multimodal_logits = torch.stack(multimodal_logits_list, dim=0)  # (SEQ_LENGTH, VOCAB_SIZE)

    print("Multimodal model top-3 predictions at each position:")
    print("-" * 55)
    import torch.nn.functional as F
    for pos in range(SEQ_LENGTH):
        logits_pos = multimodal_logits[pos]
        probs = F.softmax(logits_pos, dim=-1)
        top3_prob, top3_indices = torch.topk(probs, 3)

        print(f"Position {pos+1}: ", end="")
        for i in range(3):
            word_id = top3_indices[i].item()
            prob_val = top3_prob[i].item()
            is_correct = word_id == true_sentence_ids[pos]
            marker = " [*]" if is_correct else "   "
            print(f"{word_id}({prob_val:.3f}){marker} ", end="")
        print()
    print()

    # Show what happens with greedy decoding (argmax at each position)
    print("Greedy decoding (argmax at each position):")
    print("-" * 42)
    greedy_predictions = []
    for pos in range(SEQ_LENGTH):
        logits_pos = multimodal_logits[pos]
        pred_id = torch.argmax(logits_pos).item()
        greedy_predictions.append(pred_id)
        marker = " [*]" if pred_id == true_sentence_ids[pos] else "   "
        print(f"Position {pos+1}: {pred_id}{marker}")

    greedy_accuracy = sum(1 for i in range(SEQ_LENGTH)
                         if greedy_predictions[i] == true_sentence_ids[i]) / SEQ_LENGTH * 100
    print(f"\nGreedy accuracy: {greedy_accuracy:.1f}%")
    print(f"Greedy prediction: {greedy_predictions}")
    print(f"True sentence:     {true_sentence_ids}")
    print()

    # Apply beam search decoding with adaptive thresholding
    print("Beam search decoding with adaptive thresholding:")
    print("-" * 48)
    top_hypotheses = decoder.decode(multimodal_logits, return_top_k=3)

    for i, hypothesis in enumerate(top_hypotheses):
        accuracy = sum(1 for j in range(min(len(hypothesis), SEQ_LENGTH))
                      if j < len(true_sentence_ids) and hypothesis[j] == true_sentence_ids[j]) / SEQ_LENGTH * 100

        print(f"Rank {i+1}: {hypothesis}")
        print(f"         Accuracy: {accuracy:.1f}%")

        # Show word-by-word comparison
        print("         Comparison: ", end="")
        for j in range(SEQ_LENGTH):
            if j < len(hypothesis):
                word_id = hypothesis[j]
                true_word = true_sentence_ids[j] if j < len(true_sentence_ids) else -1
                if word_id == true_word:
                    print(f"[{word_id}] ", end="")  # Correct
                else:
                    print(f" {word_id} ", end="")   # Incorrect
            else:
                print(" [PAD] ", end="")
        print()

        # Show detailed scores for best hypothesis
        if i == 0:
            print("         LM confidence analysis:")
            detailed_scores = decoder.get_hypothesis_scores(multimodal_logits)
            if detailed_scores:
                tokens, total_logprob, lm_logprob, mm_logprob, avg_lm_weight = detailed_scores[0]
                print(f"           Average adaptive LM weight: {avg_lm_weight:.3f}")
                print(f"           (Base LM weight: {LM_WEIGHT})")
                print(f"           When LM weight < base weight: bigram LM was uncertain")
                print(f"           When LM weight = base weight: bigram LM was confident")
        print()

    # Show detailed scores for best hypothesis
    print("Detailed analysis of best hypothesis:")
    print("-" * 40)
    detailed_scores = decoder.get_hypothesis_scores(multimodal_logits)
    if detailed_scores:
        tokens, total_logprob, lm_logprob, mm_logprob, avg_lm_weight = detailed_scores[0]

        print(f"Tokens: {tokens}")
        print(f"Multimodal log probability (sum of word scores): {mm_logprob:.4f}")
        print(f"Language model log probability: {lm_logprob:.4f}")
        print(f"Average adaptive LM weight used: {avg_lm_weight:.3f}")
        print(f"Base LM weight: {LM_WEIGHT}")
        print(f"Combined score contribution:")
        print(f"  Multimodal: {mm_logprob:.4f}")
        print(f"  LM: {lm_logprob:.4f} × {avg_lm_weight:.3f} = {lm_logprob * avg_lm_weight:.4f}")
        print(f"  Total (before length norm): {mm_logprob + lm_logprob * avg_lm_weight:.4f}")
        print()
        print("Insight: The adaptive threshold mechanism:")
        print("  - Reduced LM weight when bigram probabilities were uncertain")
        print("  - Increased reliance on multimodal evidence in uncertain contexts")
        print("  - Maintained LM influence when bigram predictions were confident")
        print()

    print("=" * 60)
    print("ADAPTIVE THRESHOLDING BENEFIT:")
    print("When the language model is uncertain about bigram predictions")
    print("(probability below threshold), the system automatically:")
    print("  1. Reduces reliance on potentially unreliable LM scores")
    print("  2. Increases reliance on multimodal model probabilities")
    print("  3. Adapts dynamically to local confidence in linguistic predictions")
    print("This improves robustness when acoustic/visual evidence conflicts")
    print("with weak or uncertain language model predictions.")
    print("=" * 60)


def example_usage():
    """Basic example of how to use the beam search decoder."""
    # Parameters
    VOCAB_SIZE = 500
    SEQ_LENGTH = 10  # Number of words to predict
    BEAM_WIDTH = 5
    LM_WEIGHT = 0.7

    # Create dummy multimodal logits (in practice, these come from your model)
    # Shape: (sequence_length, vocab_size)
    multimodal_logits = torch.randn(SEQ_LENGTH, VOCAB_SIZE)

    # Create beam search decoder
    decoder = BeamSearchDecoder(
        vocab_size=VOCAB_SIZE,
        beam_width=BEAM_WIDTH,
        lm_weight=LM_WEIGHT,
        length_penalty=0.0,
        pad_token=0,
        unk_token=1,
        sos_token=0,  # Using pad as SOS for simplicity
        eos_token=None  # No EOS token in this example
    )

    # Train language model on dummy data (replace with real text data)
    dummy_corpus = create_dummy_text_corpus(VOCAB_SIZE, num_sentences=1000)
    decoder.train_language_model(dummy_corpus)

    # Perform decoding
    top_hypotheses = decoder.decode(multimodal_logits, return_top_k=3)

    print("Top hypotheses from beam search:")
    for i, hypothesis in enumerate(top_hypotheses):
        print(f"  {i+1}: {hypothesis}")

    # Get detailed scores
    detailed_scores = decoder.get_hypothesis_scores(multimodal_logits)
    print("\nDetailed scores:")
    for i, (tokens, total_logprob, lm_logprob, mm_logprob, _) in enumerate(detailed_scores):
        print(f"  {i+1}: Tokens={tokens}")
        print(f"      Total log prob: {total_logprob:.4f}")
        print(f"      LM log prob: {lm_logprob:.4f}")
        print(f"      Multimodal log prob: {mm_logprob:.4f}")


if __name__ == "__main__":
    # Run the example with thresholding demonstration
    example_usage_with_threshold()
    print("\n" + "="*60 + "\n")
    # Run basic example
    example_usage()