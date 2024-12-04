import os
import cv2
import numpy as np
import json
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

def pair_files(file_list):
    pairs = {}
    for filename in file_list:
        prefix = filename.split('_')[0]  # 提取前缀
        if prefix not in pairs:
            pairs[prefix] = []
        pairs[prefix].append(filename)
    return pairs

def calculate_miou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def process_paired_files(root, pairs):
    results = {}
    for prefix, files in pairs.items():
        gen_files = [f for f in files if 'gen' in f]
        ori_files = [f for f in files if 'ori' in f]

        if not ori_files:
            print(f"Escape: {prefix}")
            continue

        ori_file = ori_files[0]
        ori_path = os.path.join(root, ori_file)
        ori_img = cv2.imread(ori_path, cv2.IMREAD_GRAYSCALE)

        _, ori_mask = cv2.threshold(ori_img, 127, 1, cv2.THRESH_BINARY)

        for gen_file in gen_files:
            gen_path = os.path.join(root, gen_file)
            gen_img = cv2.imread(gen_path, cv2.IMREAD_GRAYSCALE)

            _, gen_mask = cv2.threshold(gen_img, 127, 1, cv2.THRESH_BINARY)

            miou = calculate_miou(gen_mask, ori_mask)
            print(f"处理配对：{prefix}, 文件={gen_file} 与 {ori_file}, mIoU={miou:.4f}")

            results[f"{prefix}_{gen_file}"] = miou

    return results

root = '.'
file_list = os.listdir(root)
paired_files = pair_files(file_list)
results = process_paired_files(root, paired_files)
mean = sum(np.array(list(results.values()))) / len(results.values())
# print(mean)

with open('miou_results.json', 'w') as json_file:
    json.dump(results, json_file)


def calculate_precision_recall(predictions, ground_truths, iou_threshold=0.5):
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x1_t, y1_t, x2_t, y2_t = box2
        xi1 = max(x1, x1_t)
        yi1 = max(y1, y1_t)
        xi2 = min(x2, x2_t)
        yi2 = min(y2, y2_t)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_t - x1_t) * (y2_t - y1_t)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0

    true_positive = []
    false_positive = []
    scores = []
    num_gt = len(ground_truths)

    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    for pred in predictions:
        pred_class, score, pred_box = pred
        matched = False
        for gt in ground_truths:
            gt_class, gt_box = gt
            if pred_class == gt_class:
                iou = calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    true_positive.append(1)
                    false_positive.append(0)
                    matched = True
                    break
        if not matched:
            true_positive.append(0)
            false_positive.append(1)
        scores.append(score)

    precision = np.cumsum(true_positive) / (np.cumsum(true_positive) + np.cumsum(false_positive))
    recall = np.cumsum(true_positive) / num_gt

    return precision, recall, scores


def calculate_ap(precision, recall):
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    ap = np.sum(np.diff(recall) * precision[1:])
    return ap


def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    ap_values = []
    for class_id in set([pred[0] for pred in predictions]):
        pred_class = [pred for pred in predictions if pred[0] == class_id]
        gt_class = [gt for gt in ground_truths if gt[0] == class_id]

        precision, recall, scores = calculate_precision_recall(pred_class, gt_class, iou_threshold)

        ap = calculate_ap(precision, recall)
        ap_values.append(ap)

    map_score = np.mean(ap_values)
    return map_score

map_score = calculate_map(predictions, ground_truths, iou_threshold=0.5)
print(f"mAP: {map_score:.4f}")