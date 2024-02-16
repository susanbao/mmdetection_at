from mmdet.apis import init_detector, inference_detector
from mmdet.apis import DetInferencer
import ipdb

ipdb.set_trace()

checkpoint = './checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'
config_path = './checkpoints/detr_r50_8xb2-150e_coco.py'
inferencer = DetInferencer(model=config_path, weights=checkpoint)

inferencer('./demo/demo.jpg', out_dir='./output')