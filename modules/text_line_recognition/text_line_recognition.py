import torch
import sys
import argparse
from PIL import Image

from vietocr.tool.config import Cfg


class Text_Line_Recognition:
    """ Text_Line_Recognition
        Model is used to recognition text from image
        Args:
            config: config dict for model
    """
    def __init__(self, config_):
        config = Cfg.load_config_from_file(config_['config'])
        config['device'] = config_['device']
        config['weights'] = config_['weight_path']
        print('config recognition: ', config)
        if config_['use_onnx']:
            from vietocr.tool.predictor_onnx import Predictor
        else:
            from vietocr.tool.predictor import Predictor
            self.model = Predictor(config)

        self.batch_size = config_['batch_size']
        print('Load Text_Line_Recognition Model finished')

    def predict(self, img):
        """ Predict text from image
        Args:
            img: Numpy Image (BGR)
        Returns:
            string: text in image
            float: Probability of result
        """
        text, prob = self.model.predict(img, return_prob=True)
        return text, prob

    def predict_batch(self, imgs):
        """ Predict texts from images
        Args:
            imgs: List Numpy Image (BGR)
        Returns:
            list string: texts in image
            list float: Probability of results
        """
        texts, probs, _ = self.model.predict_batch(imgs, self.batch_size)
        return texts, probs


if __name__ == '__main__':
    img = Image.open('/data/publication_safety/ocr_for_cavet/support/images/test3.png').convert('RGB')

    config_ = {
        'config': './configs/config_text_line_recognition.yml',
        'weight_path': './weights/text_line_recognition/transformerocr.pth',
        'device': 'cuda:1',
        'batch_size': 1,
    }

    predictor = Text_Line_Recognition(config_)
    pred_text = predictor.predict(img)
    print(pred_text)