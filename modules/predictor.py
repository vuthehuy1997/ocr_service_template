import numpy as np
from PIL import Image
import time
from pprint import pprint
import shutil
import os
import cv2
import re

from .object_detection import Object_Detection
from .text_line_detection import Text_Line_Detection
from .text_line_recognition import Text_Line_Recognition
from .kie import Key_Information_Extraction
from .postprocess import Postprocess

import utils.preprocess as preprocess
import utils.custom_plots as custom_plots

class Predictor:
    def __init__(self, config):
        self.config = config
        
        self.object_detection = Object_Detection(config['object_detection'])
        self.text_line_detection = Text_Line_Detection(config['text_line_detection'])
        self.text_line_recognition = Text_Line_Recognition(config['text_line_recongnition'])
        self.kie = Key_Information_Extraction(config['kie'])
        self.postprocess_rule = Postprocess(config['postprocess'])

       
    def preprocess(self, image):
        start_time = time.time()
        ### Code here
        preprocessed_image = self.object_detection.predict(image)
        ### End code
        end_time = time.time()
        elapsed_time = end_time - start_time
        return elapsed_time, preprocessed_image

    def text_line_detect(self, image):
        start_time = time.time()
        text_line_bboxes = self.text_line_detection.predict(image)
        end_time = time.time()
        if len(text_line_bboxes) == 0:
            raise Exception('Not found any text line')
        elapsed_time = end_time - start_time
        return elapsed_time, text_line_bboxes

    def text_line_recognize(self, image, text_line_bboxes):
        start_time = time.time()
        text_line_images = []
        for idx, text_line_bbox in enumerate(text_line_bboxes):
            text_line_image = preprocess.four_point_transform(image, text_line_bbox)
            text_line_images.append(text_line_image)
        text_line_texts = []
        text_line_probs = []
        text_line_texts, text_line_probs = self.text_line_recognition.predict_batch(text_line_images)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return elapsed_time, text_line_texts

    def key_information_extract(self, name, image, text_line_bboxes, text_line_texts):
        result = self.write_output(text_line_bboxes, text_line_texts)
        start_time = time.time()
        # Code here
        final_dict = self.kie(name, image, result)
        # End code
        end_time = time.time()
        elapsed_time = end_time - start_time
        return elapsed_time, final_dict

    def postprocess(self, kie_dict):
        start_time = time.time()
        # Code here
        final_dict = self.postprocess_rule.predict(kie_dict)
        # End code
        end_time = time.time()
        elapsed_time = end_time - start_time
        return elapsed_time, final_dict

    def predict(self, image, name, kie='rule', is_draw_image = False):
	    # Preprocessing: Corner Detection and Stretch
        # print(image.shape)
        preprocessed_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # preprocessed_image = image
        preprocessed_time, preprocessed_image = self.preprocess(preprocessed_image)
        # Text Line Detection
        text_line_detection_time, text_line_bboxes = self.text_line_detect(preprocessed_image)
        # Text Line Recognition
        text_line_recogize_time, text_line_texts = self.text_line_recognize(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB), text_line_bboxes)

        if is_draw_image:
            draw_image = custom_plots.display(preprocessed_image, text_line_bboxes, text_line_texts)
        else:
            draw_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
            draw_image = Image.fromarray(draw_image)
        print('detection: ', text_line_bboxes, '\nrecognition: ', text_line_texts)
        # Key Information Extraction
        kie_time, final_dict = self.key_information_extract(name, preprocessed_image, text_line_bboxes, text_line_texts)
        print('kie: ', final_dict)
        postprocess_time, final_dict = self.postprocess(final_dict)

        print('preprocessed_time time: ', preprocessed_time)
        print('text_line_detection time: ', text_line_detection_time)
        print('text_line_recognition time: ', text_line_recogize_time)
        print('kie time: ', kie_time)
        print('postprocess time: ', postprocess_time)
        print('rs: ', final_dict)
        module_time = {
            'preprocess': preprocessed_time,
            'detection': text_line_detection_time,
            'recognition': text_line_recogize_time,
            'kie': kie_time,
            'postprocess': postprocess_time
        }
        return final_dict, draw_image, module_time

    def predict_ocr(self, image):
        preprocessed_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Text Line Detection
        text_line_detection_time, text_line_bboxes = self.text_line_detect(image)
        # Text Line Recognition
        text_line_recogize_time, text_line_texts = self.text_line_recognize(preprocessed_image, text_line_bboxes)

        print('text_line_detection time: ', text_line_detection_time)
        print('text_line_recognition time: ', text_line_recogize_time)
        return text_line_bboxes, text_line_texts

    def write_output(self, det_res, rec_res):
        result = ''
        for res in zip(det_res, rec_res):
            # print('zip', res)
            bbox, value = res
            bbox = [item for sublist in bbox for item in sublist]
            s = [str(1)]
            for coordinate in bbox:
                s.append(str(coordinate))
            line = ','.join(s)
            line = line + ',' + value
            result += line + '\n'
        result = result.rstrip('\n')
        return result