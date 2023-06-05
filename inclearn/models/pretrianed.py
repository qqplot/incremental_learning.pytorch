import collections
import copy
import logging
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from inclearn.lib import data, factory, losses, network, utils
from inclearn.lib.network import hook
from inclearn.lib.data import samplers
from inclearn.models.icarl import ICarl

logger = logging.getLogger(__name__)


class Pretrained(ICarl):
    """
    Frozen Pretrained Network & Orthogonal projection.
    """

    def __init__(self, args):
        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        # Optimization:
        self._batch_size = args["batch_size"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]
        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        # Rehearsal Learning:
        self._memory_size = args["memory_size"]
        self._fixed_memory = args.get("fixed_memory", True)
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})
        self._n_classes = 0
        self._last_results = None
        self._validation_percent = args.get("validation")

        self._pod_flat_config = args.get("pod_flat", {})
        self._pod_spatial_config = args.get("pod_spatial", {})

        self._nca_config = args.get("nca", {})
        self._softmax_ce = args.get("softmax_ce", False)

        self._perceptual_features = args.get("perceptual_features")
        self._perceptual_style = args.get("perceptual_style")

        self._groupwise_factors = args.get("groupwise_factors", {})
        self._groupwise_factors_bis = args.get("groupwise_factors_bis", {})

        self._class_weights_config = args.get("class_weights_config", {})

        self._evaluation_type = args.get("eval_type", "icarl")
        self._evaluation_config = args.get("evaluation_config", {})

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        self._gradcam_distil = args.get("gradcam_distil", {})

        classifier_kwargs = args.get("classifier_config", {})
        self._network = network.OrthogonalNet(
            backbone_name=args["backbone_name"],
            k_orth=args["k_orth"],
            postprocessor_kwargs=args.get("postprocessor_config", {}),
            device=self._device
        )

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._finetuning_config = args.get("finetuning_config")
        self._finetuning_config["lr"] = args["ft_lr"]
        self._finetuning_config["epochs"] = args["ft_ep"]

        self._herding_indexes = []

        self._weight_generation = args.get("weight_generation")

        self._meta_transfer = args.get("meta_transfer", {})
        if self._meta_transfer:
            assert "mtl" in args["convnet"]

        self._post_processing_type = None
        self._data_memory, self._targets_memory = None, None

        self._args = args
        self._args["_logs"] = {}

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes

    def _before_task(self, train_loader, val_loader):
        self._n_classes += self._task_size
        self._network.add_classes(self._task_size)
        logger.info("Now {} examplars per class.".format(self._memory_per_class))

    def _train_task(self, train_loader, val_loader):
        # Base things
        if self._meta_transfer:
            logger.info("Setting task meta-transfer")
            self.set_meta_transfer()

        for p in self._network.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5., 5.))

        logger.debug("nb {}.".format(len(train_loader.dataset)))

        if self._meta_transfer.get("clip"):
            logger.info(f"Clipping MTL weights ({self._meta_transfer.get('clip')}).")
            clipper = BoundClipper(*self._meta_transfer.get("clip"))
        else:
            clipper = None
        
        # ----------------------------------------------------------------------
        # Train dummy_classifier
        logger.info("Dummy classifier tuning")

        self._network.train_dummy()

        parameters = self._network.dummy_classifier.parameters()

        self._optimizer = factory.get_optimizer(
            parameters, self._opt_name, self._finetuning_config["lr"], self.weight_decay
        )
        self._scheduler = None
        self._training_step(
            train_loader,
            val_loader,
            0,
            self._finetuning_config["epochs"],
            record_bn=False
        )

        # ----------------------------------------------------------------------
        # Train projection
        self._network.train_projection()
        parameters = self._network.model.parameters()
        self._optimizer = factory.get_optimizer(
            parameters, self._opt_name, self._lr, self._weight_decay
        )
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, self._scheduling, gamma=self._lr_decay
        )
        self._training_projection(
            train_loader, 
            val_loader, 
            self._finetuning_config["epochs"], 
            self._finetuning_config["epochs"] + self._n_epochs, 
            record_bn=True, 
            clipper=clipper
        )

        # ----------------------------------------------------------------------
        # Train classifier
        self._post_processing_type = None
        self._network.train_classifier()

        logger.info("classifier tuning")

        self._data_memory, self._targets_memory, _, _ = self.build_examplars(
            self.inc_dataset, self._herding_indexes
        )
        loader = self.inc_dataset.get_memory_loader(*self.get_memory())

        parameters = self._network.classifier.parameters()

        self._optimizer = factory.get_optimizer(
            parameters, self._opt_name, self._finetuning_config["lr"], self.weight_decay
        )
        self._scheduler = None
        self._training_step(
            loader,
            val_loader,
            self._finetuning_config["epochs"] + self._n_epochs,
            2 * self._finetuning_config["epochs"] + self._n_epochs,
            record_bn=False
        )

    @property
    def weight_decay(self):
        if isinstance(self._weight_decay, float):
            return self._weight_decay
        elif isinstance(self._weight_decay, dict):
            start, end = self._weight_decay["start"], self._weight_decay["end"]
            step = (max(start, end) - min(start, end)) / (self._n_tasks - 1)
            factor = -1 if start > end else 1

            return start + factor * self._task * step
        raise TypeError(
            "Invalid type {} for weight decay: {}.".format(
                type(self._weight_decay), self._weight_decay
            )
        )

    def _after_task(self, inc_dataset):
        if self._gradcam_distil:
            self._network.zero_grad()
            self._network.unset_gradcam_hook()
            self._old_model = self._network.copy().eval().to(self._device)
            self._network.on_task_end()

            self._network.set_gradcam_hook()
            self._old_model.set_gradcam_hook()
        else:
            super()._after_task(inc_dataset)

    def _eval_task(self, test_loader):
        if self._evaluation_type in ("icarl", "nme"):
            return super()._eval_task(test_loader)
        elif self._evaluation_type in ("softmax", "cnn"):
            ypred = []
            ytrue = []

            for input_dict in test_loader:
                ytrue.append(input_dict["targets"].numpy())

                inputs = input_dict["inputs"].to(self._device)
                logits = self._network(inputs)["logits"].detach()

                preds = F.softmax(logits, dim=-1)
                ypred.append(preds.cpu().numpy())

            ypred = np.concatenate(ypred)
            ytrue = np.concatenate(ytrue)

            self._last_results = (ypred, ytrue)

            return ypred, ytrue
        else:
            raise ValueError(self._evaluation_type)

    def _gen_weights(self):
        if self._weight_generation:
            utils.add_new_weights(
                self._network, self._weight_generation if self._task != 0 else "basic",
                self._n_classes, self._task_size, self.inc_dataset
            )

    def _before_task(self, train_loader, val_loader):
        # self._gen_weights()
        self._n_classes += self._task_size
        self._network.add_classes(self._task_size)
        logger.info("Now {} examplars per class.".format(self._memory_per_class))

        if self._groupwise_factors and isinstance(self._groupwise_factors, dict):
            if self._groupwise_factors_bis and self._task > 0:
                logger.info("Using second set of groupwise lr.")
                groupwise_factor = self._groupwise_factors_bis
            else:
                groupwise_factor = self._groupwise_factors

            params = []
            for group_name, group_params in self._network.get_group_parameters().items():
                if group_params is None or group_name == "last_block":
                    continue
                factor = groupwise_factor.get(group_name, 1.0)
                if factor == 0.:
                    continue
                params.append({"params": group_params, "lr": self._lr * factor})
                print(f"Group: {group_name}, lr: {self._lr * factor}.")
        elif self._groupwise_factors == "ucir":
            params = [
                {
                    "params": self._network.convnet.parameters(),
                    "lr": self._lr
                },
                {
                    "params": self._network.classifier.new_weights,
                    "lr": self._lr
                },
            ]
        else:
            params = self._network.parameters()

        self._optimizer_projection = factory.get_optimizer(params, "adamw", self._lr, self.weight_decay)

        self._scheduler_projection = factory.get_lr_scheduler(
            self._scheduling,
            self._optimizer_projection,
            nb_epochs=self._n_epochs,
            lr_decay=self._lr_decay,
            task=self._task
        )

        if self._class_weights_config:
            self._class_weights = torch.tensor(
                data.get_class_weights(train_loader.dataset, **self._class_weights_config)
            ).to(self._device)
        else:
            self._class_weights = None

    def _training_projection(
        self, train_loader, val_loader, initial_epoch, nb_epochs, record_bn=True, clipper=None
    ):
        best_epoch, best_acc = -1, -1.
        wait = 0

        grad, act = None, None
        if len(self._multiple_devices) > 1:
            logger.info("Duplicating model on {} gpus.".format(len(self._multiple_devices)))
            training_network = nn.DataParallel(self._network, self._multiple_devices)
            if self._network.gradcam_hook:
                grad, act, back_hook, for_hook = hook.get_gradcam_hook(training_network)
                training_network.module.convnet.last_conv.register_backward_hook(back_hook)
                training_network.module.convnet.last_conv.register_forward_hook(for_hook)
        else:
            training_network = self._network

        for epoch in range(initial_epoch, nb_epochs):
            self._metrics = collections.defaultdict(float)

            self._epoch_percent = epoch / (nb_epochs - initial_epoch)

            if epoch == nb_epochs - 1 and record_bn and len(self._multiple_devices) == 1 and \
               hasattr(training_network.convnet, "record_mode"):
                logger.info("Recording BN means & vars for MCBN...")
                training_network.convnet.clear_records()
                training_network.convnet.record_mode()

            prog_bar = tqdm(
                train_loader,
                disable=self._disable_progressbar,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            )
            for i, input_dict in enumerate(prog_bar, start=1):
                inputs, targets = input_dict["inputs"], input_dict["targets"]
                memory_flags = input_dict["memory_flags"]

                if grad is not None:
                    _clean_list(grad)
                    _clean_list(act)

                self._optimizer_projection.zero_grad()
                loss = self._forward_loss(
                    training_network,
                    inputs,
                    targets,
                    memory_flags,
                    gradcam_grad=grad,
                    gradcam_act=act
                )
                loss.backward()
                self._optimizer_projection.step()

                # Projection step
                Q,R = torch.linalg.qr(self._network.projection.T)
                orthogonalized_projection = (Q * torch.diag(R)).T
                self._network._new_projection = nn.Parameter(orthogonalized_projection[-1*self._network._last_n_classes:,:])

                if clipper:
                    training_network.apply(clipper)

                self._print_metrics(prog_bar, epoch, nb_epochs, i)

            if self._scheduler_projection:
                self._scheduler_projection.step(epoch)

            if self._eval_every_x_epochs and epoch != 0 and epoch % self._eval_every_x_epochs == 0:
                self._network.eval()
                self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
                    self.inc_dataset, self._herding_indexes
                )
                ytrue, ypred = self._eval_task(val_loader)
                acc = 100 * round((ypred == ytrue).sum() / len(ytrue), 3)
                logger.info("Val accuracy: {}".format(acc))
                self._network.train()

                if acc > best_acc:
                    best_epoch = epoch
                    best_acc = acc
                    wait = 0
                else:
                    wait += 1

                if self._early_stopping and self._early_stopping["patience"] > wait:
                    logger.warning("Early stopping!")
                    break
        
            # --------------------------------------------------------------------------------
            # This is added for debugging
            pretty_metrics = ", ".join(
                "{}: {}".format(metric_name, round(metric_value / (nb_epochs-initial_epoch), 3))
                for metric_name, metric_value in self._metrics.items()
            )
            logger.info(
                "T{}/{}, E{}/{} => {}".format(
                    self._task + 1, self._n_tasks, epoch + 1, nb_epochs, pretty_metrics
                )
            )
            # --------------------------------------------------------------------------------

        if self._eval_every_x_epochs:
            logger.info("Best accuracy reached at epoch {} with {}%.".format(best_epoch, best_acc))

        if len(self._multiple_devices) == 1 and hasattr(training_network.convnet, "record_mode"):
            training_network.convnet.normal_mode()

    def _compute_loss(self, inputs, outputs, targets, onehot_targets, memory_flags):
        features, logits, atts = outputs["raw_features"], outputs["logits"], outputs["attention"]

        if self._post_processing_type is None:
            scaled_logits = self._network.post_process(logits)
        else:
            scaled_logits = logits * self._post_processing_type
        
        # if self._nca_config:
        #     nca_config = copy.deepcopy(self._nca_config)
        #     if self._network.post_processor:
        #         nca_config["scale"] = self._network.post_processor.factor

        #     loss = losses.nca(
        #         logits,
        #         targets,
        #         memory_flags=memory_flags,
        #         class_weights=self._class_weights,
        #         **nca_config
        #     )
        #     self._metrics["nca"] += loss.item()
        # elif self._softmax_ce:
        #     loss = F.cross_entropy(scaled_logits, targets)
        #     self._metrics["cce"] += loss.item()
        
        loss = F.cross_entropy(scaled_logits, targets)
        self._metrics["cce"] += loss.item()

        return loss


class BoundClipper:

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, module):
        if hasattr(module, "mtl_weight"):
            module.mtl_weight.data.clamp_(min=self.lower_bound, max=self.upper_bound)
        if hasattr(module, "mtl_bias"):
            module.mtl_bias.data.clamp_(min=self.lower_bound, max=self.upper_bound)



def _clean_list(l):
    for i in range(len(l)):
        l[i] = None