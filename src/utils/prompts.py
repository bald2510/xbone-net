"""Prompt templates used by zero-shot classification and OOD scoring.

Zero-shot classification follows the single-template protocol described in the
original CLIP paper.  The positive/negative prompt pairs are kept as a separate
utility for the optional text-anchor OOD detector.
"""


ORIGINAL_CLIP_PROMPT_TEMPLATE = "A photo of a {label}."


# ============================================================
# Prompt Generation Functions
# ============================================================

def generate_clip_class_prompts(
    class_names: list[str],
    template: str = ORIGINAL_CLIP_PROMPT_TEMPLATE,
) -> dict[str, str]:
    """Generate one canonical CLIP prompt for every class.

    The default is the single prompt reported by Radford et al.:
    ``A photo of a {label}.``.  Each class competes with all other classes
    through one class-wise softmax during zero-shot classification.

    Args:
        class_names: Ordered class names substituted into ``{label}``.
        template: Prompt template containing exactly the ``{label}`` field.

    Returns:
        An insertion-ordered mapping from class name to prompt.
    """
    if "{label}" not in template:
        raise ValueError("The CLIP prompt template must contain the '{label}' field.")
    return {
        class_name: template.format(label=class_name)
        for class_name in class_names
    }


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

