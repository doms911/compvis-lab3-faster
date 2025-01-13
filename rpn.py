from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, Conv2dNormActivation
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.rpn import concat_box_prediction_layers

from anchor_utils import AnchorGenerator
from utils import decode_boxes


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        conv_depth (int, optional): number of convolutions
    """
    def __init__(self, in_channels: int, num_anchors_per_location: int, conv_depth=1) -> None:
        super().__init__()
        convs = []
        for _ in range(conv_depth):
            convs.append(Conv2dNormActivation(in_channels, in_channels, kernel_size=3, norm_layer=None))
        self.conv = nn.Sequential(*convs)
        self.cls_logits = nn.Conv2d(
            in_channels,
            out_channels= num_anchors_per_location * 1,
            kernel_size=1,
            stride=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels,
            out_channels= num_anchors_per_location * 4,
            kernel_size=1,
            stride=1
        )

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        logits = []
        bbox_reg = []
        for feature in x:
            t = self.conv(feature)
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return concat_box_prediction_layers(logits, bbox_reg)


class RegionProposalNetwork(torch.nn.Module):
    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        head: nn.Module,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        score_thresh=0.0,
    ):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        # used during testing
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

    def _get_top_n_idx(self, objectness: Tensor, num_anchors_per_level: List[int]) -> Tensor:
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 0):
            num_anchors = ob.shape[0]
            pre_nms_top_n = det_utils._topk_min(ob, self.pre_nms_top_n, 0)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=0)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r)

    def filter_proposals(
        self,
        proposals,
        objectness,
        image_shape,
        num_anchors_per_level,
    ):
        objectness = objectness.reshape(-1)
        levels = [
            torch.full((n,), idx, dtype=torch.int64) for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels)
        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        objectness = objectness[top_n_idx]
        levels = levels[top_n_idx]
        proposals = proposals[0, top_n_idx]
        objectness_prob = torch.sigmoid(objectness)
        boxes = box_ops.clip_boxes_to_image(proposals, image_shape)
        # remove small boxes
        keep = box_ops.remove_small_boxes(boxes, self.min_size)

        boxes, scores, lvl = boxes[keep], objectness_prob[keep], levels[keep]

        # remove low scoring boxes
        # use >= for Backwards compatibility
        keep = torch.where(scores >= self.score_thresh)[0]
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
        # non-maximum suppression, independently done per level
        keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

        # keep only topk scoring predictions
        keep = keep[:self.post_nms_top_n]
        boxes, scores = boxes[keep], scores[keep]

        return boxes, scores

    def forward(self, image_shape, features):
        # RPN uses all feature maps that are available
        features = list(features.values())
        anchors, num_anchors_per_level = self.anchor_generator(image_shape, features)
        objectness, pred_bbox_deltas = self.head(features)
        proposals = decode_boxes(pred_bbox_deltas, anchors)
        proposals = proposals.view(1, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, image_shape, num_anchors_per_level)
        return boxes