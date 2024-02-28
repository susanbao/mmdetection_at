import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
import torch
from tools.utils_at import *
from mmdet.models.utils import transform_tensors_to_list
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from torchvision.ops.boxes import box_area
import torch.nn.functional as F

# score_threshold = 0.5 DETR_COCO 0.98, DFDETR_COCO 0.5

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)


    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)
    
def hungarian_matching(out_logits, out_boxes, tgt_ids, tgt_bbox, cost_class_weight = 1.0, cost_bbox_weight = 5.0, cost_giou_weight = 2.0, focal_alpha = 0.25):
    """ Performs the matching
    """
    
    # We flatten to compute the cost matrices in a batch
    num_queries = out_logits.shape[0]
    out_prob = out_logits.softmax(dim=1)  # [num_queries, num_classes]
    
    # Compute the classification cost.
    alpha = focal_alpha
    gamma = 2.0
    neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
    pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
    cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
    
    # Compute the L1 cost between boxes
    cost_bbox = torch.cdist(out_boxes, tgt_bbox, p=1)
    
    # Compute the giou cost betwen boxes            
    cost_giou = -generalized_box_iou(bbox_cxcywh_to_xyxy(out_boxes), bbox_cxcywh_to_xyxy(tgt_bbox))
    
    # Final cost matrix
    C = cost_bbox * cost_bbox_weight + cost_class * cost_class_weight + cost_giou * cost_giou_weight
    C = C.view(num_queries, -1)
    result = torch.argmin(C, axis=1)
    return result

def get_indexes(out_logits, score_threshold):
    prob = out_logits.softmax(dim=1)
    prob, _ = prob.max(dim=1)
    select_mask = prob > score_threshold
    if sum(select_mask) == 0:
        score, indexes = prob.topk(10)
        indexes,_ = torch.sort(indexes, dim=0)
        print(f"Cannot find detected objects with score larger than {score_threshold}")
    else:
        indexes = select_mask.nonzero().reshape(-1)
    return indexes

def get_topk_indexes(out_logits, k):
    k = min(k, out_logits.shape[0])
    prob = out_logits.softmax(dim=1)
    prob, _ = prob.max(dim=1)
    score, indexes = prob.topk(k)
    indexes,_ = torch.sort(indexes, dim=0)
    return indexes

def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss
    
def compute_loss(out_logits, out_boxes, tgt_ids, tgt_bbox, matched_target_indexes, cls_loss_coef = 1.0, bbox_loss_coef = 5.0, giou_loss_coef = 2.0):
    if matched_target_indexes == None:
        target_classes_onehot = torch.zeros([out_logits.shape[0], out_logits.shape[1]],dtype=out_logits.dtype, layout=out_logits.layout, 
                                        device=out_logits.device)
        loss_ce = sigmoid_focal_loss(out_logits, target_classes_onehot)
        loss_ce = loss_ce.sum(axis=1)
        loss = loss_ce * cls_loss_coef
        return loss, loss_ce, torch.zeros(loss.shape), torch.zeros(loss.shape)
    cls_loss_coef = 1.0
    bbox_loss_coef = 5.0
    giou_loss_coef = 2.0
    target_boxes = tgt_bbox[matched_target_indexes]
    loss_bbox = F.l1_loss(out_boxes, target_boxes, reduction='none') # [num_queries, 4]
    loss_bbox = loss_bbox.mean(axis=1) # [num_queries]
    loss_giou = 1 - torch.diag(generalized_box_iou(bbox_cxcywh_to_xyxy(out_boxes),
                bbox_cxcywh_to_xyxy(target_boxes))) # [num_queries]
    target_classes_onehot = torch.zeros([out_logits.shape[0], out_logits.shape[1]],dtype=out_logits.dtype, layout=out_logits.layout, 
                                        device=out_logits.device)
    target_labels = torch.tensor(tgt_ids)[matched_target_indexes]
    target_classes_onehot.scatter_(1, target_labels.unsqueeze(-1), 1)
    loss_ce = sigmoid_focal_loss(out_logits, target_classes_onehot)
    loss_ce = loss_ce.sum(axis=1)
    loss = loss_ce * cls_loss_coef + loss_bbox * bbox_loss_coef + loss_giou * giou_loss_coef
    return loss, loss_ce, loss_bbox, loss_giou

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_nums", type = int, default = 0,
                        help="Start No.")
    parser.add_argument("--split", type=str, default="val",
                        help="split type: val or train.")
    parser.add_argument("--end_nums", type = int, default = -1,
                        help="Start No.")
    parser.add_argument("--count", type = int, default = 0,
                        help="Start count")
    parser.add_argument("--query_nums", type = int, default = 100,
                        help="Number of query")
    parser.add_argument("--mode_data", type = str, default = "DETR_COCO",
                        help="mode and dataset type")
    parser.add_argument("--score_threshold", type = float, default = 0.98,
                        help="score threshold")
    args = parser.parse_args()
    split = args.split
    base_path = f"./pro_data/{args.mode_data}/"
    data_path = base_path + split + "/outputs/"
    feature_path = base_path + split + "/feature/"
    annotation_path = base_path + split + "/annotation/"
    create_folder_if_not_exists(feature_path)
    create_folder_if_not_exists(annotation_path)
    
    files_in_folder = os.listdir(data_path)
    if args.end_nums == -1:
        num_files = len(files_in_folder) - 1
    else:
        num_files = args.end_nums
    # count = len(os.listdir(feature_path))
    count = args.count
    for num in range(args.start_nums, num_files):
        file_path = data_path + str(num) + ".json"
        if not os.path.exists(file_path):
            print(f"{file_path} does not exits.")
            continue
        output_data = read_json_results(file_path)
        # feature = np.array(output_data['feature'][5][0])
        # out_logits = torch.FloatTensor(output_data['pred_logits'][5][0])
        # out_boxes = torch.FloatTensor(output_data['pred_boxes'][5][0])
        feature = np.array(output_data['feature'])
        out_logits = torch.FloatTensor(output_data['pred_logits'])
        out_boxes = torch.FloatTensor(output_data['pred_boxes'])
        out_logits = out_logits[:, 0:80] # for detr & coco
        target_labels = output_data['gt_labels']
        target_boxes = torch.FloatTensor(output_data['gt_boxes'])
        if len(target_boxes) == 0:
            continue
        target_boxes = bbox_xyxy_to_cxcywh(target_boxes)
        img_h, img_w = output_data['img_metas'][0]['img_shape']
        factors = output_data['img_metas'][0]['scale_factor']
        shapes = torch.FloatTensor([img_h, img_w, img_h, img_w])
        target_boxes = target_boxes / shapes

        selected_indexes = get_topk_indexes(out_logits, args.query_nums)
        feature = feature[selected_indexes]
        out_logits = out_logits[selected_indexes]
        out_boxes = out_boxes[selected_indexes]
        
        indexes = get_indexes(out_logits, args.score_threshold)
        out_logits = out_logits[indexes]
        out_boxes = out_boxes[indexes]
        gt_indexes = hungarian_matching(out_logits, out_boxes, target_labels, target_boxes)
        loss, loss_ce, loss_bbox, loss_giou = compute_loss(out_logits, out_boxes, target_labels, target_boxes, gt_indexes)
        np_write(feature, feature_path + str(count) + ".npy")
        json_data = {}
        json_data['loss'] = transform_tensors_to_list(loss)
        json_data['loss_ce'] = transform_tensors_to_list(loss_ce)
        json_data['loss_bbox'] = transform_tensors_to_list(loss_bbox)
        json_data['loss_giou'] = transform_tensors_to_list(loss_giou)
        json_data['index'] = transform_tensors_to_list(indexes)
        json_data['gt_indexes'] = transform_tensors_to_list(gt_indexes)
        json_data['output_file_num'] = num
        write_json_results(json_data, annotation_path + str(count) + ".json")
        count += 1
        if (num+1)%1000 == 0:
            print(num)
        
if __name__ == "__main__":
    main()