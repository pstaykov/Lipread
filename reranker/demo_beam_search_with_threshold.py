"""
Demonstration of the adaptive thresholding feature in the beam search decoder.

This script specifically showcases how the adaptive threshold mechanism
dynamically adjusts reliance on the language model based on bigram prediction confidence.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add paths to import model components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lipreading'))
sys.path.insert(0, os.path.dirname(__file__))

from beam_search_decoder import BeamSearchDecoder, create_dummy_text_corpus


def demo_adaptive_thresholding():
    """
    Demonstrate how adaptive thresholding works when bigram predictions
    are uncertain vs. confident.
    """
    print("=" * 70)
    print("ADAPTIVE THRESHOLDING IN BEAM SEARCH DECODER")
    print("=" * 70)
    print("Shows how LM weight is dynamically adjusted based on bigram confidence")
    print()

    # Configuration
    VOCAB_SIZE = 500
    SEQ_LENGTH = 6
    BEAM_WIDTH = 5
    BASE_LM_WEIGHT = 0.7
    BIGRAM_THRESHOLD = 0.02  # 2% threshold for demonstration

    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Sequence length: {SEQ_LENGTH} words")
    print(f"Beam width: {BEAM_WIDTH}")
    print(f"Base LM weight: {BASE_LM_WEIGHT}")
    print(f"Bigram threshold: {BIGRAM_THRESHOLD}")
    print()

    # Create beam search decoder with adaptive thresholding
    decoder = BeamSearchDecoder(
        vocab_size=VOCAB_SIZE,
        beam_width=BEAM_WIDTH,
        lm_weight=BASE_LM_WEIGHT,
        length_penalty=0.1,
        pad_token=0,
        unk_token=1,
        sos_token=0,
        eos_token=2,
        bigram_threshold=BIGRAM_THRESHOLD
    )

    # Create a training corpus with clear bigram patterns
    training_corpus = [
        [1, 2, 3, 4],  # A B C D - very common
        [1, 2, 3, 5],  # A B C E - very common
        [6, 7, 8, 9],  # F G H I - very common
        [6, 7, 8, 10], # F G H J - very common
    ] * 30  # Repeat for strong training signals

    # Add some variety but keep the strong patterns dominant
    import random
    for _ in range(200):
        length = random.randint(2, 4)
        sentence = [random.randint(1, 15) for _ in range(length)]
        training_corpus.append(sentence)

    decoder.train_language_model(training_corpus)
    print(f"Trained language model on {len(training_corpus)} sentences")
    print()

    # Demonstrate the adaptive weighting function directly
    print("ADAPTIVE WEIGHT FUNCTION BEHAVIOR:")
    print("-" * 40)
    test_probs = [0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.5, 1.0]
    print(f"{'Bigram Prob':<12} {'Adaptive Weight':<18} {'Ratio to Base':<15} {'LM Reliance'}")
    print("-" * 55)
    for prob in test_probs:
        adaptive_weight = decoder._get_adaptive_lm_weight(BASE_LM_WEIGHT, prob)
        ratio = adaptive_weight / BASE_LM_WEIGHT if BASE_LM_WEIGHT > 0 else 0
        reliance = "High" if ratio > 0.7 else "Medium" if ratio > 0.3 else "Low"
        print(f"{prob:<12.4f} {adaptive_weight:<18.4f} {ratio:<15.2f} {reliance}")
    print()

    # Define test sequences
    print("TEST SEQUENCES:")
    print("-" * 15)

    # Sequence 1: High confidence bigrams (should use full LM weight)
    high_confidence_ids = [1, 2, 3, 4, 6, 7]  # A B C D F G
    # Bigrams: 1-2 (trained), 2-3 (trained), 3-4 (trained), 4-6 (novel), 6-7 (trained)

    # Sequence 2: Mixed confidence
    mixed_confidence_ids = [1, 2, 9, 4, 6, 7]  # A B X D F G (where 2->9 is uncertain)
    # Bigrams: 1-2 (trained), 2-9 (uncertain), 9-4 (uncertain), 4-6 (novel), 6-7 (trained)

    print(f"High confidence sequence:  {high_confidence_ids}")
    print("  Bigrams: 1-2[t], 2-3[t], 3-4[t], 4-6[?], 6-7[t]")
    print(f"Mixed confidence sequence: {mixed_confidence_ids}")
    print("  Bigrams: 1-2[t], 2-9[?], 9-4[?], 4-6[?], 6-7[t]")
    print("[t] = trained/confident, [?] = uncertain/novel")
    print()

    # Generate multimodal logits
    # We'll make the correct words have moderate probability,
    # but add some alternatives to create realistic ambiguity
    torch.manual_seed(42)
    multimodal_logits_list = []

    def create_position_logits(correct_word, confidence_level="medium"):
        """Create logits for a position with specified correctness confidence."""
        logits = torch.randn(VOCAB_SIZE)

        if confidence_level == "high":
            logits[correct_word] += 2.5  # Strong signal
        elif confidence_level == "medium":
            logits[correct_word] += 1.5  # Moderate signal
        else:  # low
            logits[correct_word] += 0.5  # Weak signal

        # Add some confusion with alternatives
        num_alternatives = 3
        alt_indices = random.sample([i for i in range(VOCAB_SIZE) if i != correct_word],
                                  min(num_alternatives, VOCAB_SIZE-1))
        for alt in alt_indices:
            logits[alt] += random.uniform(0.2, 0.8)

        return logits

    # High confidence sequence logits
    high_conf_logits = []
    high_conf_logits.append(create_position_logits(1, "high"))   # Position 0: A (high)
    high_conf_logits.append(create_position_logits(2, "high"))   # Position 1: B (high)
    high_conf_logits.append(create_position_logits(3, "high"))   # Position 2: C (high)
    high_conf_logits.append(create_position_logits(4, "medium")) # Position 3: D (medium)
    high_conf_logits.append(create_position_logits(6, "high"))   # Position 4: F (high)
    high_conf_logits.append(create_position_logits(7, "high"))   # Position 5: G (high)

    # Mixed confidence sequence logits
    mixed_conf_logits = []
    mixed_conf_logits.append(create_position_logits(1, "high"))   # Position 0: A (high)
    mixed_conf_logits.append(create_position_logits(2, "high"))   # Position 1: B (high)
    mixed_conf_logits.append(create_position_logits(9, "low"))    # Position 2: X (low - uncertain)
    mixed_conf_logits.append(create_position_logits(4, "medium")) # Position 3: D (medium)
    mixed_conf_logits.append(create_position_logits(6, "high"))   # Position 4: F (high)
    mixed_conf_logits.append(create_position_logits(7, "high"))   # Position 5: G (high)

    high_conf_logits_tensor = torch.stack(high_conf_logits, dim=0)
    mixed_conf_logits_tensor = torch.stack(mixed_conf_logits, dim=0)

    print("MULTIMODAL MODEL TOP-3 PREDICTIONS:")
    print("-" * 40)

    def print_position_predictions(logits_tensor, sequence_name, true_ids):
        print(f"{sequence_name}:")
        for pos in range(SEQ_LENGTH):
            logits_pos = logits_tensor[pos]
            probs = F.softmax(logits_pos, dim=-1)
            top3_prob, top3_indices = torch.topk(probs, 3)

            print(f"  Pos {pos+1}: ", end="")
            for i in range(3):
                word_id = top3_indices[i].item()
                prob_val = top3_prob[i].item()
                is_correct = word_id == true_ids[pos]
                marker = " [*]" if is_correct else "   "
                print(f"{word_id}({prob_val:.3f}){marker} ", end="")
            print()
        print()

    print_position_predictions(high_conf_logits_tensor, "High Confidence Sequence", high_confidence_ids)
    print_position_predictions(mixed_conf_logits_tensor, "Mixed Confidence Sequence", mixed_confidence_ids)

    # Apply beam search decoding
    print("BEAM SEARCH DECODING RESULTS:")
    print("-" * 32)

    # Decode high confidence sequence
    print("1. High Confidence Sequence:")
    high_conf_hypotheses = decoder.decode(high_conf_logits_tensor, return_top_k=2)
    for i, hypothesis in enumerate(high_conf_hypotheses):
        accuracy = sum(1 for j in range(min(len(hypothesis), SEQ_LENGTH))
                      if j < len(high_confidence_ids) and hypothesis[j] == high_confidence_ids[j]) / SEQ_LENGTH * 100
        print(f"   Rank {i+1}: {hypothesis} (Accuracy: {accuracy:.1f}%)")

        # Show detailed scores for best hypothesis
        if i == 0:
            detailed_scores = decoder.get_hypothesis_scores(high_conf_logits_tensor)
            if detailed_scores:
                tokens, total_logprob, lm_logprob, mm_logprob, avg_lm_weight = detailed_scores[0]
                print(f"      Details: LM weight={avg_lm_weight:.3f} (base={BASE_LM_WEIGHT}), "
                      f"MM score={mm_logprob:.2f}, LM score={lm_logprob:.2f}")
    print()

    # Decode mixed confidence sequence
    print("2. Mixed Confidence Sequence:")
    mixed_conf_hypotheses = decoder.decode(mixed_conf_logits_tensor, return_top_k=2)
    for i, hypothesis in enumerate(mixed_conf_hypotheses):
        accuracy = sum(1 for j in range(min(len(hypothesis), SEQ_LENGTH))
                      if j < len(mixed_confidence_ids) and hypothesis[j] == mixed_confidence_ids[j]) / SEQ_LENGTH * 100
        print(f"   Rank {i+1}: {hypothesis} (Accuracy: {accuracy:.1f}%)")

        # Show detailed scores for best hypothesis
        if i == 0:
            detailed_scores = decoder.get_hypothesis_scores(mixed_conf_logits_tensor)
            if detailed_scores:
                tokens, total_logprob, lm_logprob, mm_logprob, avg_lm_weight = detailed_scores[0]
                print(f"      Details: LM weight={avg_lm_weight:.3f} (base={BASE_LM_WEIGHT}), "
                      f"MM score={mm_logprob:.2f}, LM score={lm_logprob:.2f}")
                print(f"      Interpretation: Lower LM weight indicates bigram uncertainty")
    print()

    # Show the adaptive thresholding in action with detailed analysis
    print("ADAPTIVE THRESHOLDING ANALYSIS:")
    print("-" * 33)
    print("For the mixed confidence sequence, examining position 2 (uncertain bigram 2->9):")

    # Get detailed scores to see adaptive weights
    detailed_scores = decoder.get_hypothesis_scores(mixed_conf_logits_tensor)
    if detailed_scores and len(detailed_scores) > 0:
        # We'll examine the first hypothesis in detail
        tokens, total_logprob, lm_logprob, mm_logprob, avg_lm_weight = detailed_scores[0]

        print(f"Sequence: {tokens}")
        print(f"Average adaptive LM weight: {avg_lm_weight:.3f}")
        print(f"Base LM weight: {BASE_LM_WEIGHT}")
        print(f"Weight ratio: {avg_lm_weight/BASE_LM_WEIGHT if BASE_LM_WEIGHT > 0 else 0:.2f}")

        if avg_lm_weight < BASE_LM_WEIGHT:
            reduction = (BASE_LM_WEIGHT - avg_lm_weight) / BASE_LM_WEIGHT * 100
            print(f"LM weight reduced by {reduction:.1f}% due to bigram uncertainty")
            print("→ System relied more on multimodal evidence at uncertain positions")
        else:
            print("LM weight maintained at base level (high confidence throughout)")

    print()
    print("KEY INSIGHTS ABOUT ADAPTIVE THRESHOLDING:")
    print("-" * 40)
    print("1. When BIGRAM PROBABILITY ≥ THRESHOLD:")
    print("   → Adaptive LM weight = Base LM weight")
    print("   → Full reliance on language model")
    print("   → Used for well-trained, confident bigram predictions")
    print()
    print("2. When BIGRAM PROBABILITY < THRESHOLD:")
    print("   → Adaptive LM weight = Base LM weight × (Probability / Threshold)")
    print("   → Reduced reliance on language model")
    print("   → Increased reliance on multimodal model probabilities")
    print("   → Used for novel, uncertain, or low-frequency bigrams")
    print()
    print("3. BENEFITS:")
    print("   • Prevents LM from overriding strong multimodal evidence")
    print("   • Gracefully handles out-of-vocabulary or rare word combinations")
    print("   • Adapts locally to confidence in linguistic predictions")
    print("   • Maintains performance on high-confidence LM predictions")
    print()

    print("=" * 70)
    print("ADAPTIVE THRESHOLDING SUMMARY:")
    print("The beam search decoder now dynamically balances multimodal and")
    print("linguistic evidence based on local confidence in bigram predictions,")
    print("providing robustness when linguistic context is uncertain or unreliable.")
    print("=" * 70)


def compare_with_and_without_thresholding():
    """
    Compare performance with and without adaptive thresholding
    in a scenario where LM is uncertain.
    """
    print("\n" + "=" * 70)
    print("COMPARISON: WITH vs WITHOUT ADAPTIVE THRESHOLDING")
    print("=" * 70)

    # Setup
    VOCAB_SIZE = 500
    SEQ_LENGTH = 5
    BEAM_WIDTH = 5
    BASE_LM_WEIGHT = 0.7

    # Create scenario: uncertain bigram in middle of sequence
    true_ids = [1, 2, 9, 4, 5]  # A B X D E where B->X is uncertain

    # Training data: strong A-B and D-E patterns, but no B-X or X-D
    training_corpus = [
        [1, 2, 3, 4],  # A B C D
        [1, 2, 3, 5],  # A B C E
        [6, 7, 8, 9],  # F G H I
        [4, 5, 6, 7],  # D E F G
    ] * 25

    import random
    for _ in range(150):
        length = random.randint(2, 4)
        sentence = [random.randint(1, 10) for _ in range(length)]
        training_corpus.append(sentence)

    # Generate multimodal logits where:
    # - Positions 0,1,3,4: moderate confidence in correct words
    # - Position 2: low confidence, ambiguous between correct (9) and alternative (15)
    torch.manual_seed(123)
    multimodal_logits_list = []

    multimodal_logits_list.append(create_position_logits(1, "medium"))   # Pos 0: A
    multimodal_logits_list.append(create_position_logits(2, "medium"))   # Pos 1: B
    multimodal_logits_list.append(create_position_logits(9, "low"))     # Pos 2: X (correct, but low conf)
    multimodal_logits_list[2][15] += 1.2  # Make alternative 15 somewhat attractive
    multimodal_logits_list.append(create_position_logits(4, "medium"))   # Pos 3: D
    multimodal_logits_list.append(create_position_logits(5, "medium"))   # Pos 4: E

    multimodal_logits = torch.stack(multimodal_logits_list, dim=0)

    print(f"True sequence: {true_ids}")
    print("Scenario: Position 2 has uncertain bigram (2->9) and weak multimodal signal")
    print()

    # Test WITHOUT thresholding (original behavior)
    print("1. WITHOUT ADAPTIVE THRESHOLDING (original beam search):")
    decoder_original = BeamSearchDecoder(
        vocab_size=VOCAB_SIZE,
        beam_width=BEAM_WIDTH,
        lm_weight=BASE_LM_WEIGHT,
        bigram_threshold=0.0,  # Disables adaptive thresholding
        pad_token=0,
        unk_token=1,
        sos_token=0,
        eos_token=2
    )
    decoder_original.train_language_model(training_corpus)

    orig_hypotheses = decoder_original.decode(multimodal_logits, return_top_k=2)
    for i, hypothesis in enumerate(orig_hypotheses):
        accuracy = sum(1 for j in range(min(len(hypothesis), SEQ_LENGTH))
                      if j < len(true_ids) and hypothesis[j] == true_ids[j]) / SEQ_LENGTH * 100
        print(f"   Rank {i+1}: {hypothesis} (Accuracy: {accuracy:.1f}%)")

        if i == 0:
            detailed_scores = decoder_original.get_hypothesis_scores(multimodal_logits)
            if detailed_scores:
                tokens, total_logprob, lm_logprob, mm_logprob, avg_lm_weight = detailed_scores[0]
                print(f"      LM weight used: {avg_lm_weight:.3f} (fixed)")
    print()

    # Test WITH thresholding (adaptive behavior)
    print("2. WITH ADAPTIVE THRESHOLDING (enhanced beam search):")
    decoder_adaptive = BeamSearchDecoder(
        vocab_size=VOCAB_SIZE,
        beam_width=BEAM_WIDTH,
        lm_weight=BASE_LM_WEIGHT,
        bigram_threshold=0.03,  # 3% threshold
        pad_token=0,
        unk_token=1,
        sos_token=0,
        eos_token=2
    )
    decoder_adaptive.train_language_model(training_corpus)

    adap_hypotheses = decoder_adaptive.decode(multimodal_logits, return_top_k=2)
    for i, hypothesis in enumerate(adap_hypotheses):
        accuracy = sum(1 for j in range(min(len(hypothesis), SEQ_LENGTH))
                      if j < len(true_ids) and hypothesis[j] == true_ids[j]) / SEQ_LENGTH * 100
        print(f"   Rank {i+1}: {hypothesis} (Accuracy: {accuracy:.1f}%)")

        if i == 0:
            detailed_scores = decoder_adaptive.get_hypothesis_scores(multimodal_logits)
            if detailed_scores:
                tokens, total_logprob, lm_logprob, mm_logprob, avg_lm_weight = detailed_scores[0]
                print(f"   Avg LM weight used: {avg_lm_weight:.3f} (adaptive)")
                if avg_lm_weight < BASE_LM_WEIGHT:
                    reduction = (BASE_LM_WEIGHT - avg_lm_weight) / BASE_LM_WEIGHT * 100
                    print(f"      LM weight reduced by {reduction:.1f}% due to uncertainty")
    print()

    # Show which approach worked better
    orig_best_acc = sum(1 for j in range(len(orig_hypotheses[0]))
                       if j < len(true_ids) and orig_hypotheses[0][j] == true_ids[j]) / len(true_ids) * 100
    adap_best_acc = sum(1 for j in range(len(adap_hypotheses[0]))
                       if j < len(true_ids) and adap_hypotheses[0][j] == true_ids[j]) / len(true_ids) * 100

    print("RESULTS COMPARISON:")
    print("-" * 18)
    print(f"Original approach accuracy: {orig_best_acc:.1f}%")
    print(f"Adaptive approach accuracy: {adap_best_acc:.1f}%")

    if adap_best_acc > orig_best_acc:
        improvement = adap_best_acc - orig_best_acc
        print(f"→ Adaptive thresholding IMPROVED accuracy by {improvement:.1f}%")
    elif orig_best_acc > adap_best_acc:
        degradation = orig_best_acc - adap_best_acc
        print(f"→ Original approach was better by {degradation:.1f}%")
    else:
        print("→ Both approaches performed equally")

    print()
    print("Interpretation:")
    print("When the language model is uncertain about a bigram (low probability),")
    print("adaptive thresholding reduces reliance on potentially misleading LM scores")
    print("and increases reliance on multimodal model evidence, which can lead to")
    print("better decisions when multimodal information is more reliable than LM.")
    print("=" * 70)


def create_position_logits(correct_word, confidence_level="medium"):
    """Helper function to create position logits with specified confidence."""
    logits = torch.randn(500)

    if confidence_level == "high":
        logits[correct_word] += 2.5
    elif confidence_level == "medium":
        logits[correct_word] += 1.5
    else:  # low
        logits[correct_word] += 0.5

    # Add some confusion
    import random
    num_alternatives = 2
    alt_indices = random.sample([i for i in range(500) if i != correct_word],
                              min(num_alternatives, 499))
    for alt in alt_indices[:num_alternatives]:
        logits[alt] += random.uniform(0.3, 0.9)

    return logits


if __name__ == "__main__":
    demo_adaptive_thresholding()
    compare_with_and_without_thresholding()