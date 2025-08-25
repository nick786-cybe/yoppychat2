# utils/intent_classifier.py
from transformers import pipeline
from functools import lru_cache
import logging

# Define the intents as described in the plan
INTENT_LABELS = [
    "Personal Questions",
    "Technical Questions",
    "Latest Updates",
    "Opinion Requests",
    "Storytelling",
    "Channel Info",
    "General Conversation"
]

# The plan mentioned DistilBERT, but for zero-shot, a model fine-tuned on NLI is better.
# facebook/bart-large-mnli is a popular and effective choice.
CLASSIFIER_MODEL = "facebook/bart-large-mnli"

@lru_cache(maxsize=1)
def get_classifier():
    """
    Loads and caches the zero-shot classification pipeline.
    This ensures the model is loaded only once.
    """
    logging.info(f"Loading zero-shot classification model: {CLASSIFIER_MODEL}...")
    try:
        # It's good practice to specify the device to avoid warnings, -1 for CPU, 0 for GPU
        classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL, device=-1)
        logging.info("Zero-shot classifier loaded successfully.")
        return classifier
    except Exception as e:
        logging.error(f"Failed to load zero-shot classifier: {e}", exc_info=True)
        return None

def classify_intent(query: str) -> str:
    """
    Classifies the user's query into one of the predefined intents.

    Returns:
        The name of the most likely intent (e.g., "Personal Questions").
    """
    classifier = get_classifier()
    if not classifier:
        logging.warning("Classifier not available. Falling back to 'General Conversation'.")
        return "General Conversation"

    # The hypothesis template makes the classification more accurate for this kind of task.
    result = classifier(query, INTENT_LABELS, hypothesis_template="This question is about {}.")

    # The result is a dict with 'labels' and 'scores' sorted by score.
    top_intent = result['labels'][0]
    top_score = result['scores'][0]

    logging.info(f"Classified intent for query '{query[:50]}...' as '{top_intent}' with score {top_score:.2f}")

    # Add a confidence threshold. If the top score is too low, it's likely a general chat.
    CONFIDENCE_THRESHOLD = 0.50
    if top_score < CONFIDENCE_THRESHOLD:
        logging.info(f"Confidence score ({top_score:.2f}) below threshold {CONFIDENCE_THRESHOLD}. Re-classifying as 'General Conversation'.")
        return "General Conversation"

    return top_intent
