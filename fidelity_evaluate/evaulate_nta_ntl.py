from ultralytics import YOLO
import os
import json
import torch
import numpy as np
import os
from model import TwinLite as net
import cv2
from IOUEval import SegmentationMetric
from utils import AverageMeter
from evaluate_extract_lines import evaluate_extract_lines, evaluate_extract_lines_street_gaussian
from tqdm import tqdm
import json
import tensorflow as tf
from mmdet.apis import DetInferencer
import math
model_name = 'faster-rcnn_x101-64x4d_fpn_ms-3x_coco'
checkpoint = '/EXT_DISK/users/yangyuankun/driving_simulation/TwinLiteNet/mmdetection/checkpoints/faster_rcnn_x101_64x4d_fpn_2x_coco_20200512_161033-5961fa95.pth'
inferencer = DetInferencer(model_name, checkpoint, device=torch.cuda.current_device())


def calculate_centers(boxes):
    x1, y1, x2, y2 = boxes.T
    return torch.stack((x1 + x2 / 2, y1 + y2 / 2), dim=1)

def calculate_iou(box1, box2):
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = torch.min(box1[1] + box1[3], box2[1] + box2[3])

    inter_width = torch.clamp(x2 - x1, min=0)
    inter_height = torch.clamp(y2 - y1, min=0)
    inter_area = inter_width * inter_height

    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    iou = inter_area / (box1_area + box2_area - inter_area)
    
    return iou.item()

def find_closest_iou(gt_boxes, pred_boxes):
    gt_centers = calculate_centers(gt_boxes)
    pred_centers = calculate_centers(pred_boxes)
    
    distances = torch.cdist(gt_centers, pred_centers)
    closest_indices = torch.argmin(distances, dim=1)
    closest_preds = pred_boxes[closest_indices]
    
    ious = [calculate_iou(gt, pred) for gt, pred in zip(gt_boxes, closest_preds)]
    return ious

def calculate_precision_recall_curve(gt_boxes, pred_boxes, boxes_conf, iou_threshold=0.5):
    sorted_indices = torch.argsort(boxes_conf, descending=True)
    pred_boxes = pred_boxes[sorted_indices]
    boxes_conf = boxes_conf[sorted_indices]

    ious = find_closest_iou(gt_boxes, pred_boxes)
    
    precisions = []
    recalls = []
    
    true_positives = 0
    false_positives = 0
    num_gt_boxes = len(gt_boxes)
    
    for iou, conf in zip(ious, boxes_conf):
        if iou >= iou_threshold:
            true_positives += 1
        else:
            false_positives += 1

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (num_gt_boxes + 1e-10)
        
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls

def calculate_mAP(precisions, recalls):
    # Calculate mAP by integrating the area under the precision-recall curve
    mAP = 0
    for i in range(1, len(precisions)):
        mAP += (recalls[i] - recalls[i - 1]) * precisions[i]
    return mAP

def evaluate_nta(root, id_list, iou_threshold=[0.5]):
    model = YOLO("yolo11x.pt")
    ious_lists = []
    mAPs_lists = []
    for file in id_list:   
        ious_list = []
        mAPs_list = []
        root_img = os.path.join(root, file)
        image_list= sorted(os.listdir(root_img))
        with open(os.path.join(root, file, 'box_coordinates.json'), 'r') as json_file:
            gt_coord = json.load(json_file)
        for index in ['1', '2', '3']:
            ious = []
            mAPs = []
            for i, imgName in enumerate(image_list):
                if not imgName.endswith(f'{index}img.png'):
                    continue
                gt_boxes = gt_coord[imgName.split('.')[0]]
                if len(gt_boxes) == 0:
                    continue
                
                results = model(f"./lines/{file}/{imgName}")
                # Use the detector to do inference
                # result = inferencer(img, out_dir='./output')
                res = results[0]
                boxes = res.boxes.xyxy
                classes = res.boxes.cls
                class_mask = [classes[i].item() in {2,5,7} for i in range(len(classes))]
                boxes = boxes[class_mask]
                boxes_conf = res.boxes.conf[class_mask]
                if len(boxes) == 0:
                    ious += [0 for i in range(len(gt_boxes))]
                    mAPs.append(0)
                    continue
                gt_boxes = torch.tensor(gt_boxes).to(boxes.device)
                iou = find_closest_iou(gt_boxes, boxes)
                iou = [0 if math.isnan(x) else x for x in iou]
                ious += iou
                mAP = []
                for threshold in iou_threshold:
                    precisions, recalls = calculate_precision_recall_curve(gt_boxes, boxes, boxes_conf, iou_threshold=threshold)
                    output = calculate_mAP(precisions, recalls)
                    if math.isnan(output):
                        output = 0                    
                    mAP.append(output) 
                mAPs.append(np.mean(mAP))
            ious_list.append(np.mean(ious))
            mAPs_list.append(np.mean(mAPs))
        ious_lists.append(ious_list)
        mAPs_lists.append(mAPs_list)
    return ious_lists, mAPs_lists
            
def evaluate_ntl(root, id_list, debug, ntl=False):
    LL = SegmentationMetric(2)
    ious_lists = []

    model = net.TwinLiteNet()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load('pretrained/best.pth'))
    model.eval()

    for file in id_list:
        ious_list = []
        root_img = os.path.join(root, file)
        image_list = sorted(os.listdir(root_img))
        for index in ['1', '2', '3']:
            ll_mIoU_seg = AverageMeter()
            for _, imgName in enumerate(image_list):
                if not imgName.endswith(f'{index}img.png'):
                    continue
                img = cv2.imread(os.path.join(root,file,imgName))
                img = cv2.resize(img, (640, 360))
                img_rs = img.copy()

                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img)
                img = torch.unsqueeze(img, 0)  
                img = img.cuda().float() / 255.0
                with torch.no_grad():
                    img_out = model(img)        
                _ , out_ll = img_out   
                _, ll_predict = torch.max(out_ll, 1) # softmax prob
                LL_mask = ll_predict.byte().cpu().data.numpy()[0]*255
                img_rs[LL_mask>100]=[0,255,0]
                
                mask_gt = np.load(os.path.join(root,file,imgName[:-7]+'.npy'))
                mask_gt = cv2.resize(mask_gt, (640, 360))
                mask_gt = (mask_gt != [0,0,0]).all(axis=2) 
                
                ll_gt = torch.tensor(mask_gt).unsqueeze(0)
                LL.reset()
                LL.addBatch(ll_predict.cpu(), ll_gt.cpu())
                if ntl:
                    ll_mIoU = LL.meanIntersectionOverUnion()
                else:
                    ll_mIoU = LL.IntersectionOverUnion()    
                ll_mIoU_seg.update(ll_mIoU,ll_predict.shape[0])
                if debug:
                    mask_pred = (img_rs == [0,255,0]).all(axis=2) 
                    mask_pred_resized = cv2.resize(mask_pred.astype(np.uint8), (mask_gt.shape[1], mask_gt.shape[0]))
                    mask_pred_resized = mask_pred_resized.astype(bool)
                    intersection = np.logical_and(mask_pred_resized, mask_gt)
                    union = np.logical_or(mask_pred_resized, mask_gt)

                    lane = cv2.imread(os.path.join(root, file, imgName[:-7]+'imglane.png'))
                    resized_img = cv2.resize(img_rs, (lane.shape[1], lane.shape[0]))

                    intersection_3ch =  cv2.resize(cv2.cvtColor((intersection * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), (lane.shape[1], lane.shape[0]))
                    union_3ch = cv2.resize(cv2.cvtColor((union * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), (lane.shape[1], lane.shape[0]))
                    top_row = np.hstack((resized_img, intersection_3ch))
                    bottom_row = np.hstack((lane, union_3ch))
                    combined_image = np.vstack((top_row, bottom_row))
                    combined_image_path = os.path.join(root, file, imgName[:-7] + 'combined.png')
                    cv2.imwrite(combined_image_path, combined_image) 
            ious_list.append(ll_mIoU_seg.avg)
        ious_lists.append(ious_list)
    return ious_lists


if __name__ == '__main__':
    root = './lines/'
    name = 'comparison'
    id_list = ['a000','a001','a002','a003','a004','a005','a006','a007']   #
    result_dic = {}
    debug = True
    ntl = False
    for method in [ 'shift_emernerf' , 's3gaussianshifted']: #    ,  'pvg', 'street-gaussian', 'ours'
        with tf.device('/cpu'):
            evaluate_extract_lines_street_gaussian(root, id_list, name+'/'+method, debug, ntl=ntl)
        nta, mAP = evaluate_nta(root, id_list, iou_threshold=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        # ntl = evaluate_ntl(root, id_list, debug, ntl=True)
        iou = evaluate_ntl(root, id_list, debug, ntl=False)
        result_dic[f'{method}_mAP'] = mAP 
        result_dic[f'{method}_nta'] = nta 
        # result_dic[f'{method}_ntl'] = ntl 
        result_dic[f'{method}_iou'] = iou
    if os.path.exists(f'{name}.json'): 
        os.remove(f'{name}.json')
    with open(f'{name}.json', 'w') as json_file:
        json.dump(result_dic, json_file)

    # name = 'comparison'
    # id_list = ['a003','a006']   #
    # result_dic = {}
    # debug = False
    # ntl = False
    # for method in [ 'pvg', 'pvg_shifted_refine','street-gaussian', 'ours']:  
    #     with tf.device('/cpu'):
    #         evaluate_extract_lines_street_gaussian(root, id_list, name+'/'+method, debug, ntl=ntl)
    #     nta, mAP = evaluate_nta(root, id_list, iou_threshold=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    #     # ntl = evaluate_ntl(root, id_list, debug, ntl=True)
    #     iou = evaluate_ntl(root, id_list, debug, ntl=False)
    #     result_dic[f'{method}_mAP'] = mAP 
    #     result_dic[f'{method}_nta'] = nta 
    #     # result_dic[f'{method}_ntl'] = ntl 
    #     result_dic[f'{method}_iou'] = iou
    # if os.path.exists(f'{name}.json'): 
    #     os.remove(f'{name}.json')
    # with open(f'{name}_method.json', 'w') as json_file:
    #     json.dump(result_dic, json_file)


    # name = 'ablation'
    # id_list = ['a006', 'a003']
    # for method in os.listdir('./record/ablation'):
    #     if 'gt' in method:
    #         continue
    #     with tf.device('/cpu'):
    #         evaluate_extract_lines_street_gaussian(root, id_list, name+'/'+method, debug, ntl=ntl)
    #     nta, mAP = evaluate_nta(root, id_list, iou_threshold=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    #     # ntl = evaluate_ntl(root, id_list, debug, ntl=True)
    #     iou = evaluate_ntl(root, id_list, debug, ntl=False)
    #     result_dic[f'{method}_mAP'] = mAP 
    #     result_dic[f'{method}_nta'] = nta 
    #     # result_dic[f'{method}_ntl'] = ntl 
    #     result_dic[f'{method}_iou'] = iou
    # if os.path.exists(f'{name}.json'): 
    #     os.remove(f'{name}.json')
    # with open(f'{name}.json', 'w') as json_file:
    #     json.dump(result_dic, json_file)
