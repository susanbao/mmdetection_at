# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import ConfigType, InstanceList, MultiConfig, OptInstanceList
from ..losses import smooth_l1_loss, SmoothL1Loss
from ..task_modules.samplers import PseudoSampler
from ..utils import multi_apply
from .anchor_head import AnchorHead

import json
import os
import numpy as np

def np_write(data, file):
    with open(file, "wb") as outfile:
        np.save(outfile, data)

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def write_json_results(json_data, path):
    with open(path, "w") as outfile:
        json.dump(json_data, outfile)

def transform_tensor_to_list(l):
    return l.cpu().tolist()

def transform_tensors_to_list(l):
    if torch.is_tensor(l):
        return transform_tensor_to_list(l)
    if isinstance(l, list):
        r = []
        for i in l:
            r.append(transform_tensors_to_list(i))
        return r
    if isinstance(l, dict):
        r = {}
        for k,v in l.items():
            r[k] = transform_tensors_to_list(v)
        return r
    return l

# TODO: add loss evaluator for SSD
@MODELS.register_module()
class SSDHead(AnchorHead):
    """Implementation of `SSD head <https://arxiv.org/abs/1512.02325>`_

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Sequence[int]): Number of channels in the input feature
            map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Defaults to 0.
        feat_channels (int): Number of hidden channels when stacked_convs
            > 0. Defaults to 256.
        use_depthwise (bool): Whether to use DepthwiseSeparableConv.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, Optional): Dictionary to construct
            and config conv layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, Optional): Dictionary to construct
            and config norm layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, Optional): Dictionary to construct
            and config activation layer. Defaults to None.
        anchor_generator (:obj:`ConfigDict` or dict): Config dict for anchor
            generator.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Defaults to False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        train_cfg (:obj:`ConfigDict` or dict, Optional): Training config of
            anchor head.
        test_cfg (:obj:`ConfigDict` or dict, Optional): Testing config of
            anchor head.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], Optional): Initialization config dict.
    """  # noqa: W605

    def __init__(
        self,
        num_classes: int = 80,
        in_channels: Sequence[int] = (512, 1024, 512, 256, 256, 256),
        stacked_convs: int = 0,
        feat_channels: int = 256,
        use_depthwise: bool = False,
        conv_cfg: Optional[ConfigType] = None,
        norm_cfg: Optional[ConfigType] = None,
        act_cfg: Optional[ConfigType] = None,
        anchor_generator: ConfigType = dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=300,
            strides=[8, 16, 32, 64, 100, 300],
            ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
            basesize_ratio_range=(0.1, 0.9)),
        bbox_coder: ConfigType = dict(
            type='DeltaXYWHBBoxCoder',
            clip_border=True,
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        reg_decoded_bbox: bool = False,
        train_cfg: Optional[ConfigType] = None,
        test_cfg: Optional[ConfigType] = None,
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform', bias=0)
    ) -> None:
        super(AnchorHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.use_depthwise = use_depthwise
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.cls_out_channels = num_classes + 1  # add background class
        self.prior_generator = TASK_UTILS.build(anchor_generator)

        # Usually the numbers of anchors for each level are the same
        # except SSD detectors. So it is an int in the most dense
        # heads but a list of int in SSDHead
        self.num_base_priors = self.prior_generator.num_base_priors

        self._init_layers()
        
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            if self.train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg['sampler'], default_args=dict(context=self))
            else:
                self.sampler = PseudoSampler(context=self)
        self._cnt = 0
        self._LLALFeatureCnt = 0
        self.active_SmoothL1Loss = SmoothL1Loss(beta = self.train_cfg['smoothl1_beta'], reduction = "none")
        
    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        # TODO: Use registry to choose ConvModule type
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule

        for channel, num_base_priors in zip(self.in_channels,
                                            self.num_base_priors):
            cls_layers = []
            reg_layers = []
            in_channel = channel
            # build stacked conv tower, not used in default ssd
            for i in range(self.stacked_convs):
                cls_layers.append(
                    conv(
                        in_channel,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_layers.append(
                    conv(
                        in_channel,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                in_channel = self.feat_channels
            # SSD-Lite head
            if self.use_depthwise:
                cls_layers.append(
                    ConvModule(
                        in_channel,
                        in_channel,
                        3,
                        padding=1,
                        groups=in_channel,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_layers.append(
                    ConvModule(
                        in_channel,
                        in_channel,
                        3,
                        padding=1,
                        groups=in_channel,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            cls_layers.append(
                nn.Conv2d(
                    in_channel,
                    num_base_priors * self.cls_out_channels,
                    kernel_size=1 if self.use_depthwise else 3,
                    padding=0 if self.use_depthwise else 1))
            reg_layers.append(
                nn.Conv2d(
                    in_channel,
                    num_base_priors * 4,
                    kernel_size=1 if self.use_depthwise else 3,
                    padding=0 if self.use_depthwise else 1))
            self.cls_convs.append(nn.Sequential(*cls_layers))
            self.reg_convs.append(nn.Sequential(*reg_layers))

    # def store_LLAL_feature(self, x, store_path = "SSD_COCO_LLAL", split = "val"):
    #     if len(x) <= 0:
    #         return
    #     path = "./pro_data/" + store_path
    #     path = os.path.join(path, split)
    #     create_folder_if_not_exists(path)
    #     feature_path = os.path.join(path, "LLALFeature/")
    #     create_folder_if_not_exists(feature_path)
    #     start = self._LLALFeatureCnt
    #     for i in range(len(x)):
    #         for j in range(x[i].shape[0]):
    #             if features[j] is None:
    #                 features[j] = [transform_tensors_to_list(x[i][j])]
    #             else:
    #                 features[j].append(transform_tensors_to_list(x[i][j]))
    #     for j in range(len(features)):
    #         write_json_results(features[j], feature_path + str(start+j) + ".json")
    
    def store_LLAL_feature(self, x, store_path = "SSD_COCO_LLAL", split = "val"):
        if len(x) <= 0:
            return
        path = "./pro_data/" + store_path
        path = os.path.join(path, split)
        create_folder_if_not_exists(path)
        feature_path = os.path.join(path, "LLALFeature")
        create_folder_if_not_exists(feature_path)
        start = self._LLALFeatureCnt
        features = [None]*x[0].shape[0]
        for i in range(len(x)):
            new_feature_path = os.path.join(feature_path, str(i))
            create_folder_if_not_exists(new_feature_path)
            for j in range(x[i].shape[0]):
                np_write(x[i][j].cpu().numpy(), os.path.join(new_feature_path, str(start+j) + ".npy"))
                
    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple[list[Tensor], list[Tensor]]: A tuple of cls_scores list and
            bbox_preds list.

            - cls_scores (list[Tensor]): Classification scores for all scale \
            levels, each is a 4D-tensor, the channels number is \
            num_anchors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale \
            levels, each is a 4D-tensor, the channels number is \
            num_anchors * 4.
        """
        cls_scores = []
        bbox_preds = []
        self.store_LLAL_feature(x, split="train")
        for feat, reg_conv, cls_conv in zip(x, self.reg_convs, self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds

    def get_topk_indexes(self, out_logits, k):
        k = min(k, out_logits.shape[0])
        prob = out_logits.softmax(dim=1)
        prob, _ = prob.max(dim=1)
        score, indexes = prob.topk(k)
        indexes,_ = torch.sort(indexes, dim=0)
        return indexes
    
    def store_results(self, cls_score, bbox_pred, labels, bbox_targets, store_path = "SSD_COCO_LLAL", split = "val"):
        patch_size = 196
        path = "./pro_data/" + store_path
        create_folder_if_not_exists(path)
        path = os.path.join(path, split)
        create_folder_if_not_exists(path)
        feature_path = os.path.join(path, "feature/")
        create_folder_if_not_exists(feature_path)
        annotation_path = os.path.join(path, "annotation/")
        create_folder_if_not_exists(annotation_path)
        index = torch.logical_and(labels != 80, labels >=0)
        length = index.sum()
        self._LLALFeatureCnt += 1
        if length == 0:
            return
        if length > patch_size:
            return
        loss = torch.zeros(length, device = cls_score.device)
        filtered_cls_pred = cls_score[index]
        filtered_bbox_pred = bbox_pred[index]
        filtered_labels = labels[index]
        filtered_bbox_targets = bbox_targets[index]
        loss = F.cross_entropy(filtered_cls_pred, filtered_labels, reduction='none')
        loss += self.active_SmoothL1Loss(filtered_bbox_pred, filtered_bbox_targets).sum(dim=1)
        filtered_index = loss <= 50
        length = filtered_index.sum()
        if length == 0:
            return
        filtered_cls_pred = filtered_cls_pred[filtered_index]
        filtered_bbox_pred = filtered_bbox_pred[filtered_index]
        loss = loss[filtered_index]
        true_indices = torch.nonzero(index)
        index[true_indices[torch.logical_not(filtered_index)]] = False
        feature_size = filtered_cls_pred.shape[1] + filtered_bbox_pred.shape[1]
        feature = torch.zeros(patch_size, feature_size, device = bbox_pred.device)
        feature[:length] = torch.cat((filtered_bbox_pred, filtered_cls_pred), 1)
        if length < patch_size:
            topk = patch_size - length
            index_neg = torch.logical_not(index)
            cls_pred_neg = cls_score[index_neg]
            bbox_pred_neg = bbox_pred[index_neg]
            index_neg = self.get_topk_indexes(cls_pred_neg, topk)
            feature[length:] = torch.cat((bbox_pred_neg[index_neg], cls_pred_neg[index_neg]), 1)

        json_data = {}
        json_data['loss'] = transform_tensors_to_list(loss)
        json_data['index'] = transform_tensors_to_list(torch.arange(length))
        json_data['LLALFeature'] = (self._LLALFeatureCnt - 1)
        write_json_results(json_data, annotation_path + str(self._cnt) + ".json")
        np_write(feature.cpu().numpy(), feature_path + str(self._cnt) + ".npy")
        self._cnt += 1
        
    
    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            anchor: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            bbox_weights: Tensor,
                            avg_factor: int) -> Tuple[Tensor, Tensor]:
        """Compute loss of a single image.

        Args:
            cls_score (Tensor): Box scores for eachimage
                Has shape (num_total_anchors, num_classes).
            bbox_pred (Tensor): Box energies / deltas for each image
                level with shape (num_total_anchors, 4).
            anchors (Tensor): Box reference for each scale level with shape
                (num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (num_total_anchors,).
            label_weights (Tensor): Label weights of each anchor with shape
                (num_total_anchors,)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (num_total_anchors, 4).
            avg_factor (int): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            Tuple[Tensor, Tensor]: A tuple of cls loss and bbox loss of one
            feature map.
        """

        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(
            as_tuple=False).reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero(
            as_tuple=False).view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg['neg_pos_ratio'] * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / avg_factor

        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.train_cfg['smoothl1_beta'],
            avg_factor=avg_factor)
        self.store_results(cls_score, bbox_pred, labels, bbox_targets, split = "train")
        return loss_cls[None], loss_bbox

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, List[Tensor]]:
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, list[Tensor]]: A dictionary of loss components. the dict
            has components below:

            - loss_cls (list[Tensor]): A list containing each feature map \
            classification loss.
            - loss_bbox (list[Tensor]): A list containing each feature map \
            regression loss.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            unmap_outputs=True)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor) = cls_reg_targets

        num_images = len(batch_img_metas)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))

        losses_cls, losses_bbox = multi_apply(
            self.loss_by_feat_single,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            avg_factor=avg_factor)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
