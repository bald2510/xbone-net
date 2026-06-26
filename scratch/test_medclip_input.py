import sys
import transformers
from transformers import CLIPImageProcessor
import transformers.processing_utils

if 'feature_extractor' in transformers.processing_utils.MODALITY_TO_BASE_CLASS_MAPPING:
    transformers.processing_utils.MODALITY_TO_BASE_CLASS_MAPPING['feature_extractor'] = (
        'FeatureExtractionMixin', 'ImageProcessingMixin'
    )

original_clip_init = CLIPImageProcessor.__init__
def wrapped_clip_init(self, *args, **kwargs):
    arg_names = [
        "do_resize", "size", "resample", "do_center_crop", 
        "crop_size", "do_normalize", "image_mean", "image_std", 
        "do_convert_rgb"
    ]
    new_kwargs = dict(kwargs)
    for name, val in zip(arg_names, args):
        new_kwargs[name] = val
    return original_clip_init(self, **new_kwargs)

CLIPImageProcessor.__init__ = wrapped_clip_init
sys.modules['transformers'].CLIPFeatureExtractor = CLIPImageProcessor
transformers.CLIPFeatureExtractor = CLIPImageProcessor

from medclip import MedCLIPModel, MedCLIPVisionModel, MedCLIPProcessor

processor = MedCLIPProcessor()
print("Processor keys:", dir(processor))

# Let's inspect the tokenizer and how it processes text
text = ["Patient: 48-year-old female. Palpable mass."]
encoded = processor(
    text=text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=256,
)
print("Encoded keys:", encoded.keys())
print("Encoded input_ids shape:", encoded["input_ids"].shape)
print("Encoded input_ids values:", encoded["input_ids"][0][:20])

# Let's check what bert model tokenizer is used by medclip
tokenizer = processor.tokenizer
print("Tokenizer class:", tokenizer.__class__.__name__)
print("Decoded text:", tokenizer.decode(encoded["input_ids"][0][:20]))
