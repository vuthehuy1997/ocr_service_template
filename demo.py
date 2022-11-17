import os
import numpy as np
from PIL import Image
import time
import yaml
from modules import Predictor
import utils.preprocess as preprocess
import argparse
import json
import io
import cv2

MAX_WIDTH = 2048

def text_line_detection(filename):
    img = Image.open(os.path.join(DET_DATA_DIR, filename)).convert('RGB')
    txt_filename = filename.replace('.jpg', '.txt')
    _, text_line_bboxes = predictor.text_line_detect(img)
    _, text_line_texts = predictor.text_line_recognize(np.array(img), text_line_bboxes)
    text_line_bboxes, text_line_texts
    txt_file = open(os.path.join(DET_PRED_DIR, txt_filename), 'w')
    for bbox, text in zip(text_line_bboxes, text_line_texts):
        tlbr = preprocess.xyxy2tlbr(bbox[:4])
        top_left, top_right, bottom_right, bottom_left = tlbr
        top_left = (int(top_left[0]), int(top_left[1]))
        top_right = (int(top_right[0]), int(top_right[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
        txt_file.write(f'{top_left[0]},{top_left[1]},{top_right[0]},{top_right[1]},{bottom_right[0]},{bottom_right[1]},{bottom_left[0]},{bottom_left[1]},{text}\n')
    txt_file.close() 

def text_line_recognition_fromdet(filename):
    img = Image.open(os.path.join(RECOG_DATA_DIR, filename)).convert('RGB')
    txt_filename = filename.replace('.jpg', '.txt')
    gt_file = open(os.path.join(RECOG_DATA_DIR, txt_filename), 'r')
    gt_lines = gt_file.readlines()
    
    points_list = []
    for line in gt_lines:
        x1,y1,x2,y2,x3,y3,x4,y4 = line.split(',')[:8]
        top_left = (int(x1), int(y1))
        top_right = (int(x2), int(y2))
        bottom_right = (int(x3), int(y3))
        bottom_left = (int(x4), int(y4))
        points_list.append((top_left, top_right, bottom_right, bottom_left))

    _, text_line_texts = predictor.text_line_recognize_points(np.array(img), points_list)
    print(text_line_texts)

    txt_file = open(os.path.join(RECOG_PRED_DIR, txt_filename), 'w')
    for points, text in zip(points_list, text_line_texts):
        txt_file.write(f'{points[0][0]},{points[0][1]},{points[1][0]},{points[1][1]},{points[2][0]},{points[2][1]},{points[3][0]},{points[3][1]},{text}\n')
    txt_file.close() 

def text_line_recognition(filename, gt_text):
    img = Image.open(os.path.join(RECOG_DATA_DIR, filename)).convert('RGB')
    pred_text = predictor.text_line_recognition.predict(img)
    RECOG_PRED_FILE.write(f'{filename}\t{gt_text}\t{pred_text}\n')

def e2e(filename, kie='rule'):
    img = Image.open(os.path.join(E2E_DATA_DIR, filename)).convert('RGB')
    width, height = img.size
    if width > MAX_WIDTH:
        new_size = (MAX_WIDTH, int(MAX_WIDTH/width*height))
        print('resize: ', new_size)
        img = img.resize(new_size)
        
    json_filename = filename.replace('jpg', 'json')
    if kie == 'pick':
        final_dict, draw_image = predictor.predict(img, filename, 'pick', True)
    elif kie == 'kie_mmocr':
        final_dict, draw_image = predictor.predict(img, filename, 'kie_mmocr', True)
    else:
        final_dict, draw_image = predictor.predict(img, filename, 'rule', True)
    # preprocessed_image.save('test.jpg')
    json_string = json.dumps(final_dict, ensure_ascii=False, indent=4).encode('utf8')
    with io.open(os.path.join(E2E_PRED_DIR, json_filename),'w',encoding='utf8') as f:
        f.write(json_string.decode())
    draw_image.save(os.path.join(E2E_PRED_DIR, filename))


def create_folder(folder_list):
    for folder_path in folder_list:
        if os.path.exists(folder_path) is False:
            os.makedirs(folder_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', required=True,
        help='Config file', default='configs/config.yaml'
    )
    parser.add_argument(
        '--module', required=True,
        help='Module name to predict. eg: detection, recognition, recognition_fromdet, e2e_rule, e2e_pick',
    )
    parser.add_argument(
        '--data_dir', required=True,
        help='Dataset Directory'
    )
    parser.add_argument(
        '--pred_dir',
        help='Dataset Directory', default='output/demo'
    )
    
    args = parser.parse_args()
    config_path = args.config
    module_name = args.module
    data_dir_path = args.data_dir
    pred_dir_path = args.pred_dir
    with open(config_path) as f:
        config = yaml.safe_load(f)
    predictor = Predictor(config)

    start = time.time()
    if module_name == 'detection':
        DET_DATA_DIR = data_dir_path
        DET_PRED_DIR = pred_dir_path
        create_folder([DET_PRED_DIR])
        # Text line Detection
        filename_list = os.listdir(DET_DATA_DIR)
        for filename in filename_list:
            print('detect ', filename)
            text_line_detection(filename)
    elif module_name == 'recognition_fromdet':
        RECOG_DATA_DIR = data_dir_path
        RECOG_PRED_DIR = pred_dir_path
        create_folder([RECOG_PRED_DIR])
        # Text line recognition from detection
        filename_list = [i for i in os.listdir(RECOG_DATA_DIR) if os.path.splitext(i)[1] == '.jpg']
        for filename in filename_list:
            print('recognition ', filename)
            text_line_recognition_fromdet(filename)
    elif module_name == 'recognition':
        RECOG_DATA_DIR = data_dir_path
        RECOG_GT_FILE = open(os.path.join(RECOG_DATA_DIR, 'gt.txt'), 'r')
        RECOG_PRED_DIR = pred_dir_path
        create_folder([RECOG_PRED_DIR])
        # Text line recognition
        RECOG_PRED_FILE = open(os.path.join(RECOG_PRED_DIR, 'pred.txt'), 'w')
        gt_lines = RECOG_GT_FILE.readlines()
        for line in gt_lines:
            if line[-1] == '\n':
                line = line[:-1]
            filename, gt_text = line.split('\t')
            print('recog ', filename)
            text_line_recognition(filename, gt_text)
        RECOG_PRED_FILE.close()
    elif module_name == 'e2e_rule':
        # E2E_DATA_DIR = os.path.join(data_dir_path, 'images')
        E2E_DATA_DIR = data_dir_path
        E2E_PRED_DIR = pred_dir_path
        create_folder([E2E_PRED_DIR])
        # E2E
        filename_list = os.listdir(E2E_DATA_DIR)
        filename_list.sort()
        for filename in filename_list:
            print('e2e ', filename)
            e2e(filename, 'rule')
    elif module_name == 'e2e_pick':
        # E2E_DATA_DIR = os.path.join(data_dir_path, 'images')
        E2E_DATA_DIR = data_dir_path
        E2E_PRED_DIR = pred_dir_path
        create_folder([E2E_PRED_DIR])
        # E2E
        filename_list = os.listdir(E2E_DATA_DIR)
        filename_list.sort()
        for filename in filename_list:
            print('e2e ', filename)
            e2e(filename, 'pick')
    print('All time: ', time.time() - start)
