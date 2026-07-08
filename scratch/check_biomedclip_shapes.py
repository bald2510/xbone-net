import torch
from src.models.backbone.biomedclip import BiomedCLIPFoundation

model = BiomedCLIPFoundation()
model.return_local = True

images = torch.randn(2, 3, 224, 224)
text_ids = torch.randint(0, 1000, (2, 77))

img_feats, txt_feats = model(images, text_ids)
print("Img shape:", img_feats.shape)
print("Txt shape:", txt_feats.shape)
