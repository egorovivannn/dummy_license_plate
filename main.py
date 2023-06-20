import argparse
import torch
from detect_plates import get_boxes_from_imgs
from ocr_plate import recognize_chars, load_model
from PIL import Image
import numpy as np
import json



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run whole pipeline')
    parser.add_argument('--img_path', type=str, help='Input image path')
    parser.add_argument('--out_path', type=str, help='Output json path')

    args = parser.parse_args()

    detection = get_boxes_from_imgs(args.img_path)

    plates = []
    idx = 0
    for img_path in detection.keys():
        img = Image.open(img_path)
        img_np = np.array(img)
        for box in detection[img_path]['boxes']:
            x1, y1, x2, y2 = box
            plate = img_np[int(y1):int(y2), int(x1):int(x2)]
            Image.fromarray(plate).save(f'./results/{idx}.png')
            plates.append(plate)
            idx+=1
    
    ocr_model = load_model(weights_path='./weights/Final_LPRNet_model.pth', device='cuda')
    license_chars = recognize_chars(ocr_model, imgs=plates)
    with open(f'results/{args.out_path}.json', 'w', encoding='utf-8') as f:
        json.dump(license_chars, f, ensure_ascii=False, indent=4)