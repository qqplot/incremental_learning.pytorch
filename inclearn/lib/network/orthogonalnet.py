import copy
import logging

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import clip

from .classifiers import (Classifier, CosineClassifier, DomainClassifier, MCCosineClassifier)
from .postprocessors import FactorScalar, HeatedUpScalar, InvertedFactorScalar
from .word import Word2vec

logger = logging.getLogger(__name__)


class OrthogonalNet(nn.Module):
    """
    
    github: https://github.com/openai/CLIP
    Available Models : https://github.com/mlfoundations/open_clip#pretrained-model-interface

    """
    def __init__(
        self,
        backbone_name,
        k_orth=1,
        postprocessor_kwargs={},
        init="kaiming",
        device=None,
        **kwargs
    ):
        super(OrthogonalNet, self).__init__()

        if postprocessor_kwargs.get("type") == "learned_scaling":
            self.post_processor = FactorScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs.get("type") == "inverted_learned_scaling":
            self.post_processor = InvertedFactorScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs.get("type") == "heatedup":
            self.post_processor = HeatedUpScalar(**postprocessor_kwargs)
        elif postprocessor_kwargs.get("type") is None:
            self.post_processor = None
        else:
            raise NotImplementedError(
                "Unknown postprocessor {}.".format(postprocessor_kwargs["type"])
            )
        logger.info("Post processor is: {}".format(self.post_processor))

        # Basic things
        self.device = device
        self.k_orth = k_orth

        # Load pretrained model
        model, preprocess = clip.load(backbone_name)
        self.preprocess = preprocess    
        self.model = model
        self.encode_image = model.encode_image
        self.out_dim = self.model.text_projection.shape[1]
        self.convnet = None
        
        for name, p in self.model.named_parameters():
            p.requires_grad = False

        print("Features dimension is {}.".format(self.out_dim))

        # Orthogonal Projection
        self._old_projection = None
        self._new_projection = None
        self.init_method = init
        self.use_proj = False
        self.use_dummy = False

        # Classifier
        self.dummy_classifier = Classifier(self.out_dim, device=device)
        self.classifier = Classifier(self.out_dim, device=device)
        self._last_n_classes = 0

    def forward(self, x):
        features = self.model.encode_image(x).to(torch.float32)

        if self.use_proj:
            new_proj = F.linear(features, self._new_projection)
            new_proj = F.linear(new_proj, self._new_projection.T)
            features = new_proj

            if self._old_projection is not None:
                old_proj = F.linear(features, self._old_projection)
                old_proj = F.linear(old_proj, self._old_projection.T)
                features = features + old_proj
            
        if self.use_dummy:
            clf_outputs = self.dummy_classifier(features)
        else:
            clf_outputs = self.classifier(features)

        outputs = {"raw_features": features, 
                    "features": features, 
                    "attention": None}
        outputs.update(clf_outputs)

        return outputs

    def add_classes(self, n_classes):
        self._last_n_classes = n_classes
        if self._new_projection is not None:
            if self._old_projection is not None:
                self._old_projection = nn.Parameter(torch.cat([self._old_projection, self._new_projection]))
            else:
                self._old_projection = self._new_projection
            self._old_projection.requires_grad = False
        self._new_projection = nn.Parameter(torch.randn(n_classes*self.k_orth, self.out_dim))

        self.classifier.add_classes(n_classes)
        # self.dummy_classifier = Classifier(self.out_dim, device=self.device)
        self.dummy_classifier.add_classes(n_classes)
        self.to(self.device)

    def add_imprinted_classes(self, class_indexes, inc_dataset, **kwargs):
        if hasattr(self.classifier, "add_imprinted_classes"):
            self.classifier.add_imprinted_classes(class_indexes, inc_dataset, self, **kwargs)

    def add_custom_weights(self, weights, **kwargs):
        self.classifier.add_custom_weights(weights, **kwargs)
    
    # ------------------------------------------------------------
    # OrthogonalNet original Methods
    def train_dummy(self):
        self.use_proj = False
        self.use_dummy = True
        for p in self.dummy_classifier.parameters():
            p.requires_grad = True
    
    def train_projection(self):
        self.use_proj = True
        self.use_dummy = True
        for p in self.dummy_classifier.parameters():
            p.requires_grad = False
    
    def train_classifier(self):
        self.use_proj = True
        self.use_dummy = False
        self._new_projection.requires_grad = False
    # ------------------------------------------------------------

    def post_process(self, x):
        if self.post_processor is None:
            return x
        return self.post_processor(x)
    
    # In orthogonal training, no memory is needed.
    # def get_memory(self):
    #     return None, None

    @property
    def features_dim(self):
        return self.out_dim

    @property
    def projection(self):
        if self._old_projection is not None:
            return torch.cat([self._old_projection, self._new_projection])
        return self._new_projection

    @staticmethod
    def _init(init_method, parameters):
        if isinstance(init_method, float) or isinstance(init_method, int):
            nn.init.constant_(parameters, init_method)
        elif init_method == "kaiming":
            nn.init.kaiming_normal_(parameters, nonlinearity="linear")
        else:
            raise NotImplementedError("Unknown initialization method: {}.".format(init_method))

    def extract(self, x):
        return self.model.encode_image(x)

    def freeze(self, trainable=False, model="all"):
        if model == "all":
            model = self
        elif model == "convnet":
            model = self.model
        elif model == "classifier":
            model = self.classifier
        else:
            assert False, model

        if not isinstance(model, nn.Module):
            return self

        for param in model.parameters():
            param.requires_grad = trainable

        if not trainable:
            model.eval()
        else:
            model.train()

        return self

    def on_task_end(self):
        if isinstance(self.classifier, nn.Module):
            self.classifier.on_task_end()
            self.dummy_classifier.on_task_end()
        if isinstance(self.post_processor, nn.Module):
            self.post_processor.on_task_end()

    def on_epoch_end(self):
        if isinstance(self.classifier, nn.Module):
            self.classifier.on_epoch_end()
            self.dummy_classifier.on_epoch_end()
        if isinstance(self.post_processor, nn.Module):
            self.post_processor.on_epoch_end()

    def copy(self):
        return copy.deepcopy(self)

    @property
    def n_classes(self):
        return self.classifier.n_classes
