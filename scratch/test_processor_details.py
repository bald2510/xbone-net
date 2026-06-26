import inspect
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

from medclip import MedCLIPProcessor

processor = MedCLIPProcessor()
print("Source code of MedCLIPProcessor.__call__:")
print(inspect.getsource(processor.__call__))
