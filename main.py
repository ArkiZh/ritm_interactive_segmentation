import sys
from pathlib import Path

import torch

sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.utils.exp import load_config_file
from isegm.inference.predictors import get_predictor
from isegm.inference import clicker
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Session:
    ALL = dict()
    PREDICTOR = None
    PRE = None

    def __init__(self, img_path):
        self.session_id = img_path
        Session.ALL[img_path] = self
        Session.PRE = self

        self.clicker = clicker.Clicker()
        self.img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        self.prev_mask = None

    @staticmethod
    def get_session(img_path):
        if img_path not in Session.ALL:
            return Session(img_path)
        return Session.ALL.get(img_path)


def init_predictor():
    args_dict = {
        'checkpoint': Path('./weights/coco_lvis_h18s_itermask.pth'),
        'clicks_limit': None,
        'config_path': './config.yml',
        'cpu': False,
        'datasets': 'GrabCut,Berkeley',
        'device': torch.device(type='cpu'),  # torch.device(type='cuda', index=0),
        'eval_mode': 'cvpr',
        'exp_path': '',
        'gpus': '0',
        'iou_analysis': False,
        'logs_path': Path('experiments/evaluation_logs'),
        'min_n_clicks': 1,
        'mode': 'NoBRS',
        'model_name': None,
        'n_clicks': 20,
        'print_ious': False,
        'save_ious': False,
        'target_iou': 0.9,
        'thresh': 0.49,
        'vis_preds': False}
    from argparse import Namespace
    args = Namespace(**args_dict)

    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)

    logs_path = args.logs_path / 'others' / args.checkpoint.stem
    logs_path.mkdir(parents=True, exist_ok=True)

    model = utils.load_is_model(args.checkpoint, args.device)

    predictor_params = {}
    zoomin_params = {'target_size': 600}

    predictor = get_predictor(model, args.mode, args.device,
                              prob_thresh=args.thresh,
                              predictor_params=predictor_params,
                              zoom_in_params=zoomin_params)
    return predictor


def invoke(is_positive, y, x, img_path=None):
    if Session.PREDICTOR is None:
        Session.PREDICTOR = init_predictor()

    if img_path:
        s = Session.get_session(img_path)
    else:
        s = Session.PRE
    predictor = s.PREDICTOR

    click = clicker.Click(is_positive=is_positive, coords=(y, x))
    s.clicker.add_click(click)

    pred_thr = 0.5
    with torch.no_grad():
        predictor.set_input_image(s.img)
        pred_probs = predictor.get_prediction(s.clicker, s.prev_mask)
    pred_mask = pred_probs > pred_thr
    s.prev_mask = torch.tensor(pred_mask)[None, None, ...]

    from isegm.utils import vis
    img_blend = vis.draw_with_blend_and_clicks(s.img, mask=pred_mask,clicks_list=s.clicker.clicks_list)
    return img_blend, pred_mask


if __name__ == '__main__':

    while True:
        v = input("Input (is_positive, y, x, img_path):")
        vv = [_.strip() for _ in v.split(",")]
        if len(vv) < 3:
            continue

        is_positive = vv[0] == "1"
        y = int(vv[1])
        x = int(vv[2])
        img_path = vv[3] if len(vv) > 3 else None
        pred, mask = invoke(is_positive, y, x, img_path)
        cv2.imshow(v,cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

