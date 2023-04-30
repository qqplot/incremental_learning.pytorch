import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import clip

__all__ = ['CLIP', 'myclip']


class CLIP(nn.Module):
    """
    
    github: https://github.com/openai/CLIP
    Available Models : https://github.com/mlfoundations/open_clip#pretrained-model-interface

    """
    def __init__(
        self,
        weight_name,
        **kwargs
    ):
        super(CLIP, self).__init__()

        model, preprocess = clip.load("ViT-B/32")

        self.preprocess = preprocess
        self.model = model
        self.encode_image = model.encode_image
        self.out_dim = self.model.text_projection.shape[1]
        
        print("Features dimension is {}.".format(self.out_dim))


    def forward(self, x):
        # x = self.preprocess(x)
        with torch.no_grad():
            image_features = self.model.encode_image(x)

        return {"raw_features": image_features, 
                "features": image_features, 
                "attention": None}


def myclip18(**kwargs):
    weight_name = 'resnet18'
    model = CLIP(weight_name=weight_name, **kwargs)
    return model
