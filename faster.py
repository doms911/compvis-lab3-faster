from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from rpn import RegionProposalNetwork, RPNHead
from roi_heads import RoIHeads
from anchor_utils import AnchorGenerator


def _default_anchorgen():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        """
        Arguments: x (Tensor): input tensor of shape (N, C, 7, 7)
        Returns: tensor of shape (N, representation_size)
        This function flattens the input and then applies
        two linear layers with relu activations
        after each layer.
        """
        # YOUR CODE HERE
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(
            in_channels,
            num_classes
        )
        self.bbox_pred = nn.Linear(
            in_channels,
            num_classes * 4
        )

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class FasterRCNNwithFPN(nn.Module):
    def __init__(self, backbone, fpn, num_classes=None, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_test=1000, rpn_nms_thresh=0.7, rpn_score_thresh=0.0,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100):
        super().__init__()
        out_channels = fpn.out_channels

        rpn_anchor_generator = _default_anchorgen()
        rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        box_roi_pool = MultiScaleRoIAlign(featmap_names=["fpn2", "fpn3", "fpn4", "fpn5"], output_size=7, sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

        representation_size = 1024
        box_predictor = FastRCNNPredictor(representation_size, num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        self.backbone = backbone
        self.fpn = fpn
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, img):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        h, w = img.shape[-2:]
        features = self.backbone(img)
        features = self.fpn(features)
        proposals = self.rpn((h, w), features)
        boxes, scores, labels = self.roi_heads(features, proposals, (h, w))
        return boxes, scores, labels
