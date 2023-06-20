import yolov5
from typing import Union

def get_boxes_from_imgs(paths: Union[list, str]):

    # load model
    model = yolov5.load('keremberke/yolov5m-license-plate')
    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image

    path2preds = {}
    
    if type(paths)==str:
        paths = [paths]
    for img_path in paths:
        # inference with test time augmentation
        results = model(img_path, augment=True)
        # parse results
        predictions = results.pred[0]
        boxes = predictions[:, :4].detach().cpu().tolist() # x1, y1, x2, y2
        scores = predictions[:, 4].detach().cpu().tolist()
        path2preds[img_path] = {
            'boxes': boxes,
            'scores': scores
        }
    
    return path2preds