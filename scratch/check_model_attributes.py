import sys
# Apply monkeypatches first
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

sys.path.append(".")
from models.backbone.biomedclip import BiomedCLIPFoundation
from models.backbone.medclip_foundation import MedCLIPFoundation

print("--- BiomedCLIP Structure ---")
biomedclip = BiomedCLIPFoundation(freeze_base=False)
print("biomedclip attributes:", dir(biomedclip))
print("biomedclip.model attributes:", dir(biomedclip.model))
print("Has biomedclip.model.text?", hasattr(biomedclip.model, 'text'))

print("\n--- MedCLIP Structure ---")
medclip = MedCLIPFoundation(freeze_base=False)
print("medclip attributes:", dir(medclip))
print("medclip.model attributes:", dir(medclip.model))
print("Has medclip.model.text?", hasattr(medclip.model, 'text'))
