"""
Demonstration of beam search decoding with trigram language model
for multimodal lipreading prediction.

This script shows how the beam search decoder would be integrated
with the existing multimodal models to improve sequence-level prediction.
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


def demo_with_multimodal_model():
    """
    Demonstrate beam search decoding using a mock multimodal model.
    In practice, this would use the actual trained multimodal model
    (either late-fusion or cross-attn) to get word-level predictions.
    """
    print("=" * 60)
    print("BEAM SEARCH DECODER WITH TRIGRAM LM - DEMONSTRATION")
    print("=" * 60)

    # Configuration matching the existing models
    VOCAB_SIZE = 500  # GLipsNet vocabulary size
    SEQ_LENGTH = 8    # Number of words in the example sentence
    BEAM_WIDTH = 5    # Beam width as described in requirement
    LM_WEIGHT = 0.7   # Weight for combining multimodal and LM scores

    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Sequence length: {SEQ_LENGTH} words")
    print(f"Beam width: {BEAM_WIDTH}")
    print(f"LM weight: {LM_WEIGHT}")
    print()

    # Create beam search decoder
    decoder = BeamSearchDecoder(
        vocab_size=VOCAB_SIZE,
        beam_width=BEAM_WIDTH,
        lm_weight=LM_WEIGHT,
        length_penalty=0.1,  # Small length penalty
        pad_token=0,
        unk_token=1,
        sos_token=0,  # Start of sequence
        eos_token=2   # End of sequence (using token 2 as EOS)
    )

    # Train language model on dummy text data
    # In practice, you would train on actual text corpus matching your vocabulary
    print("Training trigram language model...")
    dummy_corpus = create_dummy_text_corpus(VOCAB_SIZE, num_sentences=5000, max_sentence_length=12)
    decoder.train_language_model(dummy_corpus)
    print(f"Trained on {len(dummy_corpus)} sentences")
    print()

    # Simulate multimodal model output for a sequence of words
    # In practice, this would come from feeding video clips through the multimodal model
    print("Simulating multimodal model predictions for each word position...")

    # Create a "true" sentence for demonstration (we'll see if beam search can recover it)
    true_sentence_ids = [3, 15, 27, 42, 8, 19, 31, 5]  # Example word IDs
    print(f"True sentence (word IDs): {true_sentence_ids}")
    print()

    # Generate mock multimodal logits for each position
    # Make the correct word have high probability, but add some uncertainty
    torch.manual_seed(42)  # For reproducibility
    multimodal_logits_list = []

    for pos in range(SEQ_LENGTH):
        # Start with random logits
        logits = torch.randn(VOCAB_SIZE)

        # Boost the logit for the true word at this position
        true_word_id = true_sentence_ids[pos]
        logits[true_word_id] += 2.0  # Make correct word more likely

        # Add some confusion by boosting a few other words slightly
        confusing_words = np.random.choice([i for i in range(VOCAB_SIZE) if i != true_word_id],
                                         size=3, replace=False)
        for w in confusing_words:
            logits[w] += 0.5

        multimodal_logits_list.append(logits)

    multimodal_logits = torch.stack(multimodal_logits_list, dim=0)  # (SEQ_LENGTH, VOCAB_SIZE)

    # Show top-5 predictions from multimodal model alone at each position
    print("Top-5 predictions from multimodal model at each position:")
    print("-" * 50)
    for pos in range(SEQ_LENGTH):
        logits_pos = multimodal_logits[pos]
        probs = F.softmax(logits_pos, dim=-1)
        top5_prob, top5_indices = torch.topk(probs, 5)

        print(f"Position {pos+1}: ", end="")
        for i in range(5):
            word_id = top5_indices[i].item()
            prob_val = top5_prob[i].item()
            marker = "*" if word_id == true_sentence_ids[pos] else " "
            print(f"{word_id}({prob_val:.3f}){marker} ", end="")
        print()
    print()

    # Apply beam search decoding with trigram LM
    print("Applying beam search decoding with trigram LM...")
    print("-" * 50)

    # Decode using beam search
    top_hypotheses = decoder.decode(multimodal_logits, return_top_k=3)

    print(f"Top {len(top_hypotheses)} hypotheses from beam search:")
    for i, hypothesis in enumerate(top_hypotheses):
        # Calculate accuracy vs true sentence
        min_len = min(len(hypothesis), len(true_sentence_ids))
        correct = sum(1 for j in range(min_len) if hypothesis[j] == true_sentence_ids[j])
        accuracy = correct / len(true_sentence_ids) * 100

        print(f"  {i+1}. {hypothesis}")
        print(f"      Accuracy vs true: {accuracy:.1f}% ({correct}/{len(true_sentence_ids)} correct)")

        # Show as word IDs for easy comparison
        print(f"      True:     {true_sentence_ids}")
        print(f"      Hypothesis: {hypothesis + [0]*(len(true_sentence_ids)-len(hypothesis)) if len(hypothesis) < len(true_sentence_ids) else hypothesis[:len(true_sentence_ids)]}")
        print()

        # Show detailed scores for best hypothesis
        if i == 0:
            detailed_scores = decoder.get_hypothesis_scores(multimodal_logits)
            if detailed_scores:
                tokens, total_logprob, lm_logprob, mm_logprob, _ = detailed_scores[0]  # Ignore adaptive weight for display
                print(f"      Detailed scores:")
                print(f"        Total log prob: {total_logprob:.4f}")
                print(f"        LM log prob: {lm_logprob:.4f}")
                print(f"        MM log prob: {mm_logprob:.4f}")
    print()

    # Get detailed scores for analysis
    print("Detailed scores for top hypothesis:")
    print("-" * 40)
    detailed_scores = decoder.get_hypothesis_scores(multimodal_logits)
    if detailed_scores:
        tokens, total_logprob, lm_logprob, mm_logprob, _ = detailed_scores[0]  # Ignore adaptive weight
        print(f"Tokens: {tokens}")
        print(f"Total log probability: {total_logprob:.4f}")
        print(f"Language model log probability: {lm_logprob:.4f}")
        print(f"Multimodal model log probability: {mm_logprob:.4f}")
        print(f"LM weight: {LM_WEIGHT}")
        print(f"Combined score: {mm_logprob:.4f} + {LM_WEIGHT} × {lm_logprob:.4f} = {mm_logprob + LM_WEIGHT * lm_logprob:.4f}")

    print()
    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)

    return top_hypotheses, true_sentence_ids


def show_integration_points():
    """
    Show where the beam search decoder would integrate with existing code.
    """
    print("\nINTEGRATION POINTS WITH EXISTING CODE:")
    print("=" * 40)
    print()
    print("1. In multimodal/late-fusion/test.py:")
    print("   - Instead of taking argmax(logits), get full probability distribution")
    print("   - Process sequence of video clips (not just single clips)")
    print("   - Apply beam_search_decoder.decode() to get final sentence")
    print()
    print("2. In multimodal/cross-attn/test.py:")
    print("   - Similar modifications for sequence processing")
    print()
    print("3. Required changes:")
    print("   - Modify data loading to return sequences of clips")
    print("   - Modify model to handle sequences (or process clips sequentially)")
    print("   - Integrate beam search decoder in testing/inference pipeline")
    print()
    print("4. Language model training:")
    print("   - Train TrigramLanguageModel on text corpus")
    print("   - Corpus should match the vocabulary domain (German words for GLips)")
    print()


if __name__ == "__main__":
    # Run the demonstration
    demo_with_multimodal_model()

    # Show integration information
    show_integration_points()