"""
Zero-Shot Prompt Engineering for OOD Text-Anchor Scoring.
===============================================================================
Generates natural-language prompt pairs encoded by the VLM text encoder:
  - Text anchor embeddings for each known pathology
  - Positive prompts asserting presence of pathology
  - Negative prompts asserting absence of pathology

These anchors are used by OODDetector.score_text_anchor for zero-shot OOD detection.
"""


# ============================================================
# Prompt Generation Functions
# ============================================================

def generate_custom_prompts(
    pathologies: list[str],
    image_context: str = "a bone x-ray",
) -> dict[str, dict[str, str]]:
    """Generate positive / negative prompt pairs for text-anchor OOD scoring.

    Each pathology receives two prompts:
      - Positive: asserts the pathology is present in the image
      - Negative: asserts the pathology is absent from the image

    Template pattern: "this is an image of <context>; <assertion>"

    Args:
        pathologies: List of pathology or class names (e.g., ['Osteosarcoma', 'Normal']).
        image_context: Modality context descriptor prepended to prompts.

    Returns:
        Nested dictionary mapping each pathology name to a dict with
        'positive' and 'negative' prompt strings.

    Example:
        prompts = generate_custom_prompts(["Osteosarcoma", "Normal"])
        pos = prompts["Osteosarcoma"]["positive"]
    """
    # --- Generate positive and negative prompt templates ---
    return {
        path: {
            "positive": f"this is an image of {image_context}; {path.lower()} presented in image",
            "negative": f"this is an image of {image_context}; no {path.lower()} presented in image",
        }
        for path in pathologies
    }


