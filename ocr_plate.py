from LPRNet import build_lprnet
import torch
import cv2
import numpy as np
from typing import Union

WEIGHTS_PATH = '/home/ivan/weights/Final_LPRNet_model.pth'

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]
DEVICE = 'cuda'


def load_model(weights_path: str, device: str):
    model = build_lprnet(
        lpr_max_len=8, 
        class_num=68, 
        dropout_rate=0, 
        phase='test'
        )

    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()
    return model


def process_img(img: Union[str, np.array], height: int = 94, width: int = 24):
    if type(img)==str:
        image = cv2.imread(img)
    else:
        image = img
    img_height, img_width, _ = image.shape
    if img_height != height or img_width != width:
        image = cv2.resize(image, (height, width))

    image = image.astype('float32')
    image -= 127.5
    image *= 0.0078125
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0)
    return image


def get_preds_from_img(model, img: torch.tensor):
    with torch.no_grad():
        preds = model(img.to(DEVICE)).detach().cpu().numpy()

    pred = preds[0, :, :]
    pred_label = list()
    for j in range(pred.shape[1]):
        pred_label.append(np.argmax(pred[:, j], axis=0))

    no_repeat_blank_label = list()
    pre_c = pred_label[0]
    if pre_c != len(CHARS) - 1:
        no_repeat_blank_label.append(pre_c)
    for c in pred_label: # dropout repeate label and blank label
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == len(CHARS) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c
    chars_result = ''.join([CHARS[i] for i in no_repeat_blank_label])
    return chars_result

def recognize_chars(model, imgs: Union[list, str, np.array]):
    
    idx2chars = {}
    if type(imgs)==str:
        imgs = [imgs]

    for idx, img_item in enumerate(imgs):
        img = process_img(img_item)
        chars_result = get_preds_from_img(model, img)
        idx2chars[idx] = chars_result

    return idx2chars

