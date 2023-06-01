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
        trainable=False,
        **kwargs
    ):
        super(CLIP, self).__init__()

        # model, preprocess = clip.load("ViT-B/32")
        model, preprocess = clip.load(weight_name)
        print(model)
        self.preprocess = preprocess    
        self.model = model
        self.encode_image = model.encode_image
        self.out_dim = self.model.text_projection.shape[1]
        
        for name, p in self.model.named_parameters():
            p.requires_grad = trainable

        print("Features dimension is {}.".format(self.out_dim))


    def forward(self, x):
        # x = self.preprocess(x)
        image_features = self.model.encode_image(x)

        return {"raw_features": image_features, 
                "features": image_features, 
                "attention": None}


def myclip(**kwargs):
    # weight_name = 'ViT-B/32' #  'RN50' # "ViT-B/32"
    model = CLIP(weight_name=kwargs['clip_weight_name'], trainable=kwargs['clip_trainable'], **kwargs)
    return model

# def resnet50(pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#     return model