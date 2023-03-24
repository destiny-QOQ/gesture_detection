# -*- coding: utf-8 -*-
"""
@Time ： 2022/5/12 11:21
@Auth ： Zhang Di
@File ：detect.py
@IDE ：PyCharm
"""
import json
import os
import sys
import pdb
import tqdm
import numpy as np
import tensorflow as tf
import cv2
import math
from abc import ABCMeta, abstractmethod
import onnxruntime
sys.path.insert(0, '/media/tclwh2/facepro/lg/nanodet_9652')

import torch
from nanodet.model.arch import build_model
from nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    convert_old_model,
    load_config,
    load_model_weight,
    mkdir,
)

class HandDetectABC(metaclass=ABCMeta):

    def __init__(self, input_shape=(288, 512), prob_threshold=1.0, iou_threshold=1.0, use_sigmoid=False):
        self.classes = ["hand", "person", "face"]
        self.num_classes = 3
        self.use_sigmoid = use_sigmoid
        self.strides = (8, 16, 32, 64)
        self.input_shape = input_shape
        self.input_h = input_shape[0]
        self.input_w = input_shape[1]
        self.reg_max = 7
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.project = np.arange(self.reg_max + 1)
        # self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        # self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        self.mean = np.array([0, 0, 0], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([255, 255, 255], dtype=np.float32).reshape(1, 1, 3)
        self.keep_ratio = False
        self.mlvl_anchors = []
        for i in range(len(self.strides)):
            anchors = self._make_grid(
                (math.ceil(self.input_shape[0] / self.strides[i]), math.ceil(self.input_shape[1] / self.strides[i])),
                self.strides[i])
            self.mlvl_anchors.append(anchors)




    def _make_grid(self, featmap_size, stride):
        feat_h, feat_w = featmap_size
        shift_x = np.arange(0, feat_w) * stride
        shift_y = np.arange(0, feat_h) * stride
        xv, yv = np.meshgrid(shift_x, shift_y)
        xv = xv.flatten()
        yv = yv.flatten()
        """
        [[  0   0]
        [  8   0]
        [ 16   0]
        ....
        [496   0]
        [504   0]]
        """
        return np.stack((xv, yv), axis=-1)
        # cx = xv + 0.5 * (stride - 1)
        # cy = yv + 0.5 * (stride - 1)
        # return np.stack((cx, cy), axis=-1)

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def sigmoid(self, x, axis=0):
        x_exp = np.exp(-x)
        return 1. / (1 + x_exp)

    def _normalize(self, img):
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        return img

    def preprocess(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(self.input_shape[1] / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_shape[1] - neww - left, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.input_shape[0] * hw_scale), self.input_shape[1]
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_shape[0] - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        return img, newh, neww, top, left

    def detect(self, srcimg, draw_img):
        det_json = {}
        img, newh, neww, top, left = self.preprocess(srcimg, self.keep_ratio)

        preds = self.infer_img(img)

        det_bboxes, det_conf, det_classid = self.post_process(preds)
        imgh, imgw, imgc = draw_img.shape
        ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww

        for i in range(det_bboxes.shape[0]):
            xmin, ymin, xmax, ymax = max(int((det_bboxes[i, 0] - left) * ratiow), 0), max(
                int((det_bboxes[i, 1] - top) * ratioh), 0), min(
                int((det_bboxes[i, 2] - left) * ratiow), srcimg.shape[1]), min(int((det_bboxes[i, 3] - top) * ratioh),
                                                                               srcimg.shape[0])
            label = int(det_classid[i])
            score = float(det_conf[i])

            if label not in det_json.keys():
                det_json[label] = []
            det_json[label].append([xmin, ymin, xmax, ymax, score])

            self.drawPred(draw_img, det_classid[i], det_conf[i], xmin, ymin, xmax, ymax)

        return draw_img, det_json

    def post_process(self, preds, scale_factor=1, rescale=False):
        mlvl_bboxes = []
        mlvl_scores = []
        ind = 0
        for stride, anchors in zip(self.strides, self.mlvl_anchors):
            cls_score, bbox_pred = preds[ind:(ind + anchors.shape[0]), :self.num_classes], preds[
                                                                                           ind:(ind + anchors.shape[0]),
                                                                                           self.num_classes:]
            ind += anchors.shape[0]
            bbox_pred = self.softmax(bbox_pred.reshape(-1, self.reg_max + 1), axis=1)
            # bbox_pred = np.sum(bbox_pred * np.expand_dims(self.project, axis=0), axis=1).reshape((-1, 4))
            bbox_pred = np.dot(bbox_pred, self.project).reshape(-1, 4)
            bbox_pred *= stride

            # nms_pre = cfg.get('nms_pre', -1)
            nms_pre = 1000
            if nms_pre > 0 and cls_score.shape[0] > nms_pre:
                max_scores = cls_score.max(axis=1)
                topk_inds = max_scores.argsort()[::-1][0:nms_pre]
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                cls_score = cls_score[topk_inds, :]

            bboxes = self.distance2bbox(anchors, bbox_pred, max_shape=self.input_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(cls_score)

        mlvl_bboxes = np.concatenate(mlvl_bboxes, axis=0)
        if rescale:
            mlvl_bboxes /= scale_factor
        mlvl_scores = np.concatenate(mlvl_scores, axis=0)

        bboxes_wh = mlvl_bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]  ####xywh
        classIds = np.argmax(mlvl_scores, axis=1)
        confidences = np.max(mlvl_scores, axis=1)  ####max_class_confidence

        if self.use_sigmoid:
            confidences = self.sigmoid(confidences)

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.prob_threshold, self.iou_threshold)
        if len(indices) > 0:
            mlvl_bboxes = mlvl_bboxes[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            return mlvl_bboxes, confidences, classIds
        else:
            print('nothing detect')
            return np.array([]), np.array([]), np.array([])

    def process_result(self, det_boxes):
        detected_boxes = []
        for one_box in range(len(det_boxes)):
            ymin, xmin, ymax, xmax = det_boxes[one_box]
            detected_boxes.append((xmin, ymin, xmax - xmin, ymax - ymin))  # x1, y1, w, h
        return detected_boxes

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        if classId == 0:
            color = (255, 0, 0)
        if classId == 1:
            color = (0, 255, 0)
        if classId == 2:
            color = (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
        return frame

    @abstractmethod
    def infer_img(self, img):
        pass


class HandDetectTFLITE(HandDetectABC):
    def __init__(self, model_path, prob_threshold, iou_threshold, use_sigmoid, *args, **kwargs):
        super(HandDetectTFLITE, self).__init__(*args, **kwargs)
        print("Using TFLITE as inference backend")
        print(f"Using weight : {model_path}")
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.use_sigmoid = use_sigmoid
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def infer_img(self, img):
        img = img[None, ...]
        self.interpreter.set_tensor(self.input_details[0]["index"], img)
        self.interpreter.invoke()

        preds = []
        for i in range(len(self.output_details)):
            scale, zero = self.output_details[i]["quantization"]
            output_data = self.interpreter.get_tensor(self.output_details[i]['index']).astype(np.float32)
            output_data = (output_data - zero) * scale
            preds.append(output_data[0])

        return preds[0]


class HandDetectONNX(HandDetectABC):
    def __init__(self, model_path, prob_threshold, iou_threshold, use_sigmoid, *args, **kwargs):
        super(HandDetectONNX, self).__init__()
        print("Using ONNX as inference backend")
        print(f"Using weight : {model_path}")

        self.model_path = model_path
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.use_sigmoid = use_sigmoid
        self.sess = onnxruntime.InferenceSession(model_path)

    def infer_img(self, img):
        img = self._normalize(img)
        img = img[None, ...]
        img = np.transpose(img, (0, 3, 1, 2))
        onnx_out = self.sess.run(None, {'data': img.astype('float32')})
        # onnx_out[0].tofile("onnx_out.bin")
        # pdb.set_trace()
        ### onnx_out[0] : (1, 3064, 35)
        return onnx_out[0][0]


class HandDetectOPENCV(HandDetectABC):
    def __init__(self, model_path, prob_threshold, iou_threshold, use_sigmoid, *args, **kwargs):
        super(HandDetectOPENCV, self).__init__()
        print("Using OPENCV as inference backend")
        print(f"Using weight : {model_path}")

        self.model_path = model_path
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.use_sigmoid = use_sigmoid
        self.net = cv2.dnn.readNet(model_path)

    def infer_img(self, img):
        img = self._normalize(img)
        blob = cv2.dnn.blobFromImage(img)
        self.net.setInput(blob)

        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0].squeeze(axis=0)
        # outs.tofile("opencv_out.bin")
        # pdb.set_trace()
        return outs


class HandDetectTORCH(HandDetectABC):
    def __init__(self, model_path, prob_threshold, iou_threshold, use_sigmoid, *args, **kwargs):
        super(HandDetectTORCH, self).__init__()
        print("Using ONNX as inference backend")
        print(f"Using weight : {model_path}")

        self.model_path = model_path
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.use_sigmoid = use_sigmoid
        logger = NanoDetLightningLogger(cfg.save_dir)
        load_config(cfg,
                    "./gesture_config/repvgg_ghost_pan_512_288.yml")
        self.model = build_model(cfg.model)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(self.model, checkpoint, logger)
        self.model.eval()
        self.model.cuda("cuda:0")

    def infer_img(self, img):
        img = self._normalize(img)
        img = img[None, ...].astype(np.float32)
        img = np.transpose(img, (0, 3, 1, 2))
        img = torch.from_numpy(img)
        img = img.cuda("cuda:0")

        with torch.no_grad():
            out = self.model(img)

        return out[0].cpu().detach().numpy()


if __name__ == '__main__':
    backend = "onnx"
    if backend == "onnx":
        detector = HandDetectONNX(model_path="ghost_pan.onnx", prob_threshold=0.45, iou_threshold=0.4, use_sigmoid=False)
    elif backend == "torch":
        detector = HandDetectTORCH(model_path="workspace/repvgg_ghost_pan_512_288_True/model_best/model_best_deploy.ckpt", prob_threshold=0.45, iou_threshold=0.4, use_sigmoid=True)
    elif backend == "tflite":
        detector = HandDetectTFLITE(model_path="out_repvgg/qat_model_repvgg_0cle.tflite", prob_threshold=0.45, iou_threshold=0.4, use_sigmoid=True)
    elif backend == "opencv":
        detector = HandDetectOPENCV(model_path="repvgg_A-5_data14_output1.onnx", prob_threshold=0.45,
                                    iou_threshold=0.4, use_sigmoid=False)
    #
    img_dir = "/media/tclwh2/facepro/zcc/datasets/detection_datasets/testdata_set/images"
    # img_dir = "./test_img"
    output_draw = backend + "/draw_img"
    os.makedirs(output_draw, exist_ok=True)
    f1 = open(backend + "ghost_pan.json", "w")
    res_json = {}
    with open("testdata_name_to_id.json", "r") as f:
        name_to_id = json.load(f)

    for file in tqdm.tqdm(os.listdir(img_dir)):
        image_path = os.path.join(img_dir, file)
        # image_path = "./1img/17_163054621.jpg"
        # print(image_path)
        img = cv2.imread(image_path)
        draw_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert file in name_to_id.keys()
        _id = name_to_id[file]
        res, det_json = detector.detect(img, draw_img)
        res_json[_id] = det_json
        cv2.imwrite(os.path.join(output_draw, file), res)

    json.dump(res_json, f1)






