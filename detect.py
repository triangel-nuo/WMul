import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import load_classifier, time_synchronized, TracedModel


def select_device(device=''):
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()

    return torch.device('cuda:0' if cuda else 'cpu')


def detect(source):
    torch.cuda.empty_cache()
    # 参数设置
    imgsz = 640
    conf_thres=0.25
    iou_thres=0.65
    
    device = select_device() #设置设备
    
    # Load model
    model = attempt_load('detector.pt', map_location=device)  
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride) 
    
    
    # model = TracedModel(model, device, imgsz)
    
    # 从source中读取数据 
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # print(names)

    # Run inference 会报警告
    if device.type != 'cpu':
        tmp = torch.zeros(1, 3, imgsz, imgsz).to(device)
        model(tmp.type_as(next(model.parameters())))  # run once
        
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    
    
    for path, img, im0s, vid_cap in dataset:
    # for img, im0s in dataset:  
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=False)[0]

        # Inference
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        # print(pred)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s
            
            p = Path(p)  # to Path
            save_path = str(p.name)  # img.jpg
            print(save_path)
            
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Write results
                max_area = 0.0
                for *xyxy, conf, cls in reversed(det):
                    area = abs(xyxy[0] - xyxy[2]) * abs(xyxy[1] - xyxy[3])
                    if area > max_area:
                        max_x1, max_y1, max_x2, max_y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        max_area = area
                        
                cropped = im0[int(max_y1):int(max_y2), int(max_x1):int(max_x2)]
            
            result = Image.fromarray(cropped).convert('RGB')
            # print(result)
            
            # cv2.imwrite(save_path, cropped)
            return result

    
def main():
    with torch.no_grad():
        detect('inference/221006000/car_10')

if __name__ == '__main__':
    main()
