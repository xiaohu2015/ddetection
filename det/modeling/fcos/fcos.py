# -*- coding: UTF-8 -*-
# **********************************************************

import logging
import math
from typing import List
import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from fvcore.nn import sigmoid_focal_loss_jit

from detectron2.layers import ShapeSpec, batched_nms, cat, get_norm
from detectron2.structures import Boxes, ImageList, Instances, pairwise_point_box_distance
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from detectron2.modeling import META_ARCH_REGISTRY, BACKBONE_REGISTRY, build_backbone
from detectron2.config import configurable
from detectron2.modeling.box_regression import Box2BoxTransformLinear
from detectron2.modeling.backbone.fpn import FPN, build_resnet_backbone, LastLevelP6P7
from detectron2.config import CfgNode as CN

from det.utils import comm

logger = logging.getLogger(__name__)

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

def compute_ious(inputs, targets):
    """compute iou and giou"""
    inputs = torch.cat((-inputs[..., :2], inputs[..., 2:]), dim=-1)
    targets = torch.cat((-targets[..., :2], targets[..., 2:]), dim=-1)
    eps = torch.finfo(torch.float32).eps

    inputs_area = (inputs[..., 2] - inputs[..., 0]).clamp_(min=0) \
        * (inputs[..., 3] - inputs[..., 1]).clamp_(min=0)
    targets_area = (targets[..., 2] - targets[..., 0]).clamp_(min=0) \
        * (targets[..., 3] - targets[..., 1]).clamp_(min=0)

    w_intersect = (torch.min(inputs[..., 2], targets[..., 2])
                   - torch.max(inputs[..., 0], targets[..., 0])).clamp_(min=0)
    h_intersect = (torch.min(inputs[..., 3], targets[..., 3])
                   - torch.max(inputs[..., 1], targets[..., 1])).clamp_(min=0)

    area_intersect = w_intersect * h_intersect
    area_union = targets_area + inputs_area - area_intersect

    ious = area_intersect / area_union.clamp(min=eps)
    
    g_w_intersect = torch.max(inputs[..., 2], targets[..., 2]) \
            - torch.min(inputs[..., 0], targets[..., 0])
    g_h_intersect = torch.max(inputs[..., 3], targets[..., 3]) \
            - torch.min(inputs[..., 1], targets[..., 1])
    ac_uion = g_w_intersect * g_h_intersect
    gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)

    return ious, gious
    

def add_fcos_config(cfg):
    """
    Add config for FCOS
    """

    cfg.MODEL.FCOS = CN()
    cfg.MODEL.FCOS.NUM_CLASSES = 80
    cfg.MODEL.FCOS.IN_FEATURES = ['p3', 'p4', 'p5', 'p6', 'p7']
    cfg.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
    cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA = 0.25
    cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA = 2.0
    cfg.MODEL.FCOS.IOU_LOSS_TYPE = 'giou'
    cfg.MODEL.FCOS.BOX_QUALITY = 'iou'
    cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS = 1.5
    cfg.MODEL.FCOS.OBJECT_SIZE_OF_INTEREST = [
        [-1, 64],
        [64, 128],
        [128, 256],
        [256, 512],
        [512, float("inf")],
    ]
            
    cfg.MODEL.FCOS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST = 1000
    cfg.MODEL.FCOS.NMS_THRESH_TEST = 0.6

    cfg.MODEL.FCOS.NUM_CONVS = 4
    cfg.MODEL.FCOS.PRIOR_PROB = 0.01
    cfg.MODEL.FCOS.NORM = 'GN'

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
    cfg.MODEL.ANCHOR_GENERATOR.OFFSET = 0.5

    
@BACKBONE_REGISTRY.register()
def build_fcos_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = out_channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels, 'p5'),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@META_ARCH_REGISTRY.register()
class FCOS(nn.Module):
    """
    Implement FCOS (https://arxiv.org/abs/1708.02002).
    """

    @configurable
    def __init__(
        self,
        *,
        backbone,
        head,
        head_in_features,
        anchor_generator,
        box2box_transform,
        num_classes,
        fpn_strides,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        iou_loss_type="giou",
        box_quality="ctrness",
        center_sampling_radius=2.5,
        object_sizes_of_interest,
        test_score_thresh=0.05,
        test_topk_candidates=1000,
        test_nms_thresh=0.5,
        max_detections_per_image=100,
        pixel_mean,
        pixel_std,
        input_format="BGR",
    ):
        """
        NOTE: this interface is experimental.
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            head (nn.Module): a module that predicts logits, regression deltas, centerness
                for each level from a list of per-level features
            head_in_features (Tuple[str]): Names of the input feature maps to be used in head
            anchor_generator (nn.Module): a module that creates anchors from a
                list of features. Usually an instance of :class:`AnchorGenerator`. For FCOS,
                each grid only has one anchor, ther anchor size equals to grid size.
            box2box_transform (Box2BoxTransform): defines the transform from anchors boxes to
                instance boxes
            num_classes (int): number of classes. Used to label background proposals.
            fpn_strides (List[int]): stride of each level features
            # Loss parameters:
            focal_loss_alpha (float): focal_loss_alpha
            focal_loss_gamma (float): focal_loss_gamma
            iou_loss_type (str): box regression loss type, default "giou"
            box_quality (str): "ctrness" or 'iou', 'iou' is better
            center_sampling_radius (float): the radius for center sampling
            object_sizes_of_interest (List[List]): the object sizes of interest for each level
            # Inference parameters:
            test_score_thresh (float): Inference cls score threshold, only anchors with
                score > INFERENCE_TH are considered for inference (to improve speed)
            test_topk_candidates (int): Select topk candidates before NMS
            test_nms_thresh (float): Overlap threshold used for non-maximum suppression
                (suppress boxes with IoU >= this threshold)
            max_detections_per_image (int):
                Maximum number of detections to return per image during inference
                (100 is based on the limit established for the COCO dataset).
            # Input parameters
            pixel_mean (Tuple[float]):
                Values to be used for image normalization (BGR order).
                To train on images of different number of channels, set different mean & std.
                Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
            pixel_std (Tuple[float]):
                When using pre-trained models in Detectron1 or any MSRA models,
                std has been absorbed into its conv1 weights, so the std needs to be set 1.
                Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
            input_format (str): Whether the model needs RGB, YUV, HSV etc.
        """

        super().__init__()

        self.backbone = backbone
        self.head = head
        self.head_in_features = head_in_features
        self.anchor_generator = anchor_generator
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides

        # Matching and loss
        self.box2box_transform = box2box_transform
        self.center_sampling_radius = center_sampling_radius
        self.object_sizes_of_interest = object_sizes_of_interest
       
        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.iou_loss_type = iou_loss_type
        self.box_quality = box_quality
        
        # Inference parameters:
        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.max_detections_per_image = max_detections_per_image

        self.input_format = input_format

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.FCOS.IN_FEATURES]
        head = FCOSHead(cfg, feature_shapes)
        anchor_generator = build_anchor_generator(cfg, feature_shapes)
        return {
            "backbone": backbone,
            "head": head,
            "anchor_generator": anchor_generator,
            "box2box_transform": Box2BoxTransformLinear(normalize_by_size=True),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_classes": cfg.MODEL.FCOS.NUM_CLASSES,
            "head_in_features": cfg.MODEL.FCOS.IN_FEATURES,
            "fpn_strides": cfg.MODEL.FCOS.FPN_STRIDES,
            # Loss parameters:
            "focal_loss_alpha": cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA,
            "iou_loss_type": cfg.MODEL.FCOS.IOU_LOSS_TYPE,
            "box_quality": cfg.MODEL.FCOS.BOX_QUALITY,
            "center_sampling_radius": cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS,
            "object_sizes_of_interest": cfg.MODEL.FCOS.OBJECT_SIZE_OF_INTEREST,
            # Inference parameters:
            "test_score_thresh": cfg.MODEL.FCOS.SCORE_THRESH_TEST,
            "test_topk_candidates": cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST,
            "test_nms_thresh": cfg.MODEL.FCOS.NMS_THRESH_TEST,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # Vis parameters
            "input_format": cfg.INPUT.FORMAT,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        pred_logits, pred_box_deltas, pred_box_centerness = self.head(features)
        anchors = self.anchor_generator(features)

        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_box_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_box_deltas]
        pred_box_centerness = [permute_to_N_HWA_K(x, 1) for x in pred_box_centerness]

        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_classes, gt_shifts_reg_deltas, gt_centerness = self.get_ground_truth(
                anchors, gt_instances)
            return self.losses(gt_classes, gt_shifts_reg_deltas, gt_centerness,
                               pred_logits, pred_box_deltas, pred_box_centerness)
        else:
            results = self.inference(anchors, pred_logits, pred_box_deltas, pred_box_centerness,
                                     images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, gt_classes, gt_shifts_deltas, gt_centerness,
               pred_logits, pred_box_deltas, pred_centerness):
        """
        Args:
            gt_classes, gt_shifts_deltas, gt_centerness: see output of :meth:`get_ground_truth`.
                Their shapes are (N, R), (N, R, 4), (N, R, 1), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_box_deltas, pred_centerness: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4 or 1).
                Where K is the number of classes used in `pred_logits`.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_logits = cat(pred_logits, dim=1).view(-1, self.num_classes)
        pred_box_deltas = cat(pred_box_deltas, dim=1).view(-1, 4)
        pred_centerness = cat(pred_centerness, dim=1).view(-1, 1)

        gt_classes = gt_classes.flatten()
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4)
        gt_centerness = gt_centerness.view(-1, 1)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        num_foreground = comm.all_reduce(num_foreground) / float(comm.get_world_size())
        num_foreground_centerness = gt_centerness[foreground_idxs].sum()
        num_targets = comm.all_reduce(num_foreground_centerness)  / float(comm.get_world_size())

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1.0, num_foreground)

        # regression loss
        ious, gious = compute_ious(pred_box_deltas[foreground_idxs], 
            gt_shifts_deltas[foreground_idxs])

        if self.iou_loss_type == "iou":
            loss_box_reg = -ious.clamp(min=torch.finfo(torch.float32).eps).log()
        elif self.iou_loss_type == "linear_iou":
            loss_box_reg = 1 - ious
        elif self.iou_loss_type == "giou":
            loss_box_reg = 1 - gious
        else:
            raise NotImplementedError
        
        if self.box_quality == 'ctrness':
            loss_box_reg = loss_box_reg * gt_centerness[foreground_idxs].view(loss_box_reg.size())
            loss_box_reg = loss_box_reg.sum() / max(1.0, num_targets)

            # centerness loss
            loss_centerness = F.binary_cross_entropy_with_logits(
                pred_centerness[foreground_idxs],
                gt_centerness[foreground_idxs],
                reduction="sum",
            ) / max(1, num_foreground)

            return {
                "loss_cls": loss_cls,
                "loss_box_reg": loss_box_reg,
                "loss_box_centerness": loss_centerness
            }
        elif self.box_quality == 'iou':
            loss_box_reg = loss_box_reg.sum() / max(1.0, num_targets)
            loss_box_iou = F.binary_cross_entropy_with_logits(
                pred_centerness[foreground_idxs].view(-1), ious.detach(),
                reduction="sum"
            ) / max(1.0, num_targets)
            return {
                "loss_cls": loss_cls,
                "loss_box_reg": loss_box_reg,
                "loss_box_iou": loss_box_iou,              
            }
        else:
            raise NotImplementedError

    @torch.no_grad()
    def get_ground_truth(self, anchors, targets):
        """
        Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of points, i.e. the sum of Hi x Wi for all levels.
                Points in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Points in the background are assigned the label "K".
                Points in the ignore areas are assigned a label "-1", i.e. ignore.
            gt_shifts_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth point2box transform
                targets (dl, dt, dr, db) that map each shift to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                shift is labeled as foreground.
            gt_centerness (Tensor):
                An float tensor (0, 1) of shape (N, R) whose values in [0, 1]
                storing ground-truth centerness for each shift.
        """
        gt_classes = []
        gt_shifts_deltas = []
        gt_centerness = []

        # get anchor centers for each feature level
        points = [anchor_per_level.get_centers() for anchor_per_level in anchors]

        anchors_over_all_feature_maps = Boxes.cat(anchors)
        points_over_all_feature_maps = torch.cat(points, dim=0)
        num_anchors = len(anchors_over_all_feature_maps)

        for targets_per_image in targets:
            num_gts = len(targets_per_image)
            gt_boxes = targets_per_image.gt_boxes
            if num_gts == 0:
                gt_classes_i = targets_per_image.gt_classes.new_full((num_anchors,), self.num_classes)
                gt_shifts_reg_deltas_i = gt_boxes.tensor.new_zeros((num_anchors, 4))
                gt_centerness_i = gt_boxes.tensor.new_zeros((num_anchors,))
            else:
                object_sizes_of_interest = torch.cat([
                    points_i.new_tensor(size).unsqueeze(0).expand(
                        points_i.size(0), -1) for points_i, size in zip(
                            points, self.object_sizes_of_interest)
                ], dim=0)

                # [N, M, 4] # M: num_gts, N: num_points
                deltas = pairwise_point_box_distance(points_over_all_feature_maps, gt_boxes)

                if self.center_sampling_radius > 0:
                    gt_box_centers = gt_boxes.get_centers() # [M, 2]
                    is_in_boxes = []
                    for stride, points_i in zip(self.fpn_strides, points):
                        radius = stride * self.center_sampling_radius
                        center_boxes = Boxes(torch.cat((
                            torch.max(gt_box_centers - radius, gt_boxes.tensor[:, :2]),
                            torch.min(gt_box_centers + radius, gt_boxes.tensor[:, 2:]),
                        ), dim=-1))
                        center_deltas = pairwise_point_box_distance(
                            points_i, center_boxes)
                        is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                    is_in_boxes = torch.cat(is_in_boxes, dim=0)
                else:
                    # no center sampling, it will use all the locations within a ground-truth box
                    is_in_boxes = deltas.min(dim=-1).values > 0 # [N, M]

                max_deltas = deltas.max(dim=-1).values # [N, M]
                # limit the regression range for each location [N, M]
                is_cared_in_the_level = \
                    (max_deltas >= object_sizes_of_interest[:, None, 0]) & \
                    (max_deltas <= object_sizes_of_interest[:, None, 1])

                gt_positions_area = gt_boxes.area().unsqueeze(0).repeat(
                    points_over_all_feature_maps.size(0), 1) # [N, M]
                gt_positions_area[~is_in_boxes] = math.inf
                gt_positions_area[~is_cared_in_the_level] = math.inf

                # if there are still more than one objects for a position,
                # we choose the one with minimal area
                positions_min_area, gt_matched_idxs = gt_positions_area.min(dim=1)

                # ground truth box regression
                gt_shifts_reg_deltas_i = self.box2box_transform.get_deltas(
                    anchors_over_all_feature_maps.tensor, gt_boxes[gt_matched_idxs].tensor)

                # ground truth classes
                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Shifts with area inf are treated as background.
                gt_classes_i[positions_min_area == math.inf] = self.num_classes

                # ground truth centerness
                left_right = gt_shifts_reg_deltas_i[:, [0, 2]]
                top_bottom = gt_shifts_reg_deltas_i[:, [1, 3]]
                gt_centerness_i = torch.sqrt(
                    (left_right.min(dim=-1).values / left_right.max(dim=-1).values).clamp_(min=0)
                    * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values).clamp_(min=0)
                )

            gt_classes.append(gt_classes_i)
            gt_shifts_deltas.append(gt_shifts_reg_deltas_i)
            gt_centerness.append(gt_centerness_i)

        return torch.stack(gt_classes), torch.stack(
            gt_shifts_deltas), torch.stack(gt_centerness)

    def inference(self, anchors, pred_logits, pred_box_deltas, pred_box_centerness,
                    image_sizes):
        """
        Arguments:
            pred_logits, pred_box_deltas, pred_box_centerness: list[Tensor], one per level. 
                Each has shape (N, Hi * Wi * Ai, K or 4, or 1)
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
            image_sizes (List[(h, w)]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        
        results: List[Instances] = []
        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_box_deltas]
            centerness_per_image = [x[img_idx] for x in pred_box_centerness]
            results_per_image = self.inference_single_image(
                anchors, pred_logits_per_image, deltas_per_image, centerness_per_image, image_size
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, anchors, box_cls, box_delta, box_centerness,
                               image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).
        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            box_centerness (list[Tensor]): Same shape as 'box_cls' except that K becomes 1.
            image_size (tuple(H, W)): a tuple of the image height and width.
        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, box_ctr_i, anchors_i in zip(
                box_cls, box_delta, box_centerness, anchors):
            # (HxWxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.test_topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.test_score_thresh
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(
                box_reg_i, anchors_i.tensor)

            box_ctr_i = box_ctr_i.flatten().sigmoid_()[anchor_idxs]
            predicted_prob = torch.sqrt(predicted_prob * box_ctr_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]

        keep = batched_nms(
            boxes_all, scores_all, class_idxs_all,
            self.test_nms_thresh
        )
        keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class FCOSHead(nn.Module):
    """
    The head used in FCOS for object classification and box regression.
    It has two subnets for the three tasks, with a common structure but separate parameters.
    """

    @configurable
    def __init__(
        self,
        *,
        input_shape: List[ShapeSpec],
        num_classes,
        num_anchors,
        conv_dims: List[int],
        norm="",
        prior_prob=0.01,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape (List[ShapeSpec]): input shape
            num_classes (int): number of classes. Used to label background proposals.
            num_anchors (int): number of generated anchors
            conv_dims (List[int]): dimensions for each convolution layer
            norm (str or callable):
                    Normalization for conv layers except for the two output layers.
                    See :func:`detectron2.layers.get_norm` for supported types.
            prior_prob (float): Prior weight for computing bias
        """
        super().__init__()

        assert num_anchors == 1

        if norm == "BN" or norm == "SyncBN":
            logger.warning("Shared norm does not work well for BN, SyncBN, expect poor results")

        cls_subnet = []
        bbox_subnet = []
        for in_channels, out_channels in zip(
            [input_shape[0].channels] + list(conv_dims), conv_dims
        ):
            cls_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm:
                cls_subnet.append(get_norm(norm, out_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm:
                bbox_subnet.append(get_norm(norm, out_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            conv_dims[-1], num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(
            conv_dims[-1], num_anchors * 4, kernel_size=3, stride=1, padding=1
        )
        self.bbox_centerness = nn.Conv2d(
            conv_dims[-1], num_anchors * 1, kernel_size=3, stride=1, padding=1
        )

        #self.scales = nn.ModuleList(
        #    [Scale(init_value=1.0) for _ in range(len(input_shape))])

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score,
            self.bbox_pred, self.bbox_centerness]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        return {
            "input_shape": input_shape,
            "num_classes": cfg.MODEL.FCOS.NUM_CLASSES,
            "conv_dims": [input_shape[0].channels] * cfg.MODEL.FCOS.NUM_CONVS,
            "prior_prob": cfg.MODEL.FCOS.PRIOR_PROB,
            "norm": cfg.MODEL.FCOS.NORM,
            "num_anchors": num_anchors,
        }

    def forward(self, features: List[Tensor]):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.
        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
            bbox_centerness (list[Tensor]): #lvl tensors, each has shape (N, Ax1, Hi, Wi).
        """
        logits = []
        bbox_reg = []
        bbox_centerness = []
        for level, feature in enumerate(features):
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_output = self.bbox_subnet(feature)
            #bbox_reg.append(F.relu(self.scales[level](self.bbox_pred(bbox_output))))
            bbox_reg.append(F.relu(self.bbox_pred(bbox_output)))
            bbox_centerness.append(self.bbox_centerness(bbox_output))
        return logits, bbox_reg, bbox_centerness
