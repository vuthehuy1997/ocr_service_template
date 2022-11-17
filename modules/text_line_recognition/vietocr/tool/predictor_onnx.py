from vietocr.tool.translate import build_model, translate, translate_beam_search, process_input, process_image, predict
from vietocr.tool.utils import download_weights
from vietocr.model.vocab import Vocab

import torch
from collections import defaultdict

import math
import numpy as np
from torchvision.transforms import functional as F
import onnxruntime
from PIL import Image
import time

def translate_onnx(img, session, max_seq_length=128, sos_token=1, eos_token=2):
    """data: BxCxHxW"""
    cnn_session, encoder_session, decoder_session = session
    
    # create cnn input
    # print(img.shape)
    cnn_input = {cnn_session.get_inputs()[0].name: img}
    tmp_time = time.time()
    src = cnn_session.run(None, cnn_input)
    cnn_time = time.time()-tmp_time
    
    # create encoder input
    encoder_input = {encoder_session.get_inputs()[0].name: src[0]}

    tmp_time = time.time()
    encoder_outputs, hidden = encoder_session.run(None, encoder_input)
    encoder_time = time.time()-tmp_time

    translated_sentence = [[sos_token] * len(img)]
    max_length = 0

    decoder_time = 0
    while max_length <= max_seq_length and not all(
        np.any(np.asarray(translated_sentence).T == eos_token, axis=1)
    ):
        tgt_inp = translated_sentence
        decoder_input = {decoder_session.get_inputs()[0].name: tgt_inp[-1], decoder_session.get_inputs()[1].name: hidden, decoder_session.get_inputs()[2].name: encoder_outputs}
        
        tmp_time = time.time()
        output, hidden, _ = decoder_session.run(None, decoder_input)
        decoder_time += time.time()-tmp_time

        output = np.expand_dims(output, axis=1)
        output = torch.Tensor(output)

        values, indices = torch.topk(output, 1)
        indices = indices[:, -1, 0]
        indices = indices.tolist()

        translated_sentence.append(indices)
        max_length += 1

        del output

    translated_sentence = np.asarray(translated_sentence).T

    return translated_sentence, (cnn_time, encoder_time, decoder_time)

class TransformInput():
    def __init__(self, height, min_width, max_width):
        self.height = height
        self.min_width = min_width
        self.max_width = max_width
    
    def get_size(self, w, h):
        new_w = int(self.height * float(w) / float(h))
        round_to = 10
        new_w = math.ceil(new_w/round_to)*round_to
        new_w = max(new_w, self.min_width)
        new_w = min(new_w, self.max_width)
        return new_w

    def transform(self, image):
        '''
            image: numpy 
        '''
        # img = Image.fromarray(image)
        # img = img.convert('RGB')
        img = image

        w, h = img.size
        width_size = self.get_size(w, h)

        img = img.resize((width_size, self.height), Image.ANTIALIAS)
        img = np.asarray(img).transpose(2,0, 1)
        img = img/255

        return img
    
    def __call__(self, imgs):
        output = [self.transform(img) for img in imgs]
        max_width = max(i.shape[-1] for i in output)
        for i in range(len(output)):
            w = output[i].shape[-1]
            padd = np.zeros((output[i].shape[0], output[i].shape[1], max_width - output[i].shape[2]))
            output[i] = np.concatenate((output[i],padd), axis=2)
        output = np.array(output, dtype=np.float32)
        return output

class Predictor():
    def __init__(self, config):

        # device = config['device']
        
        # model, vocab = build_model(config)
        # weights = '/tmp/weights.pth'

        # if config['weights'].startswith('http'):
        #     weights = download_weights(config['weights'])
        # else:
        #     weights = config['weights']

        # model.load_state_dict(torch.load(weights, map_location=torch.device(device)))

        vocab = Vocab(config['vocab'])
        cnn_session = onnxruntime.InferenceSession(config['weignts_onnx']['cnn'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        encoder_session = onnxruntime.InferenceSession(config['weignts_onnx']['encoder'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        decoder_session = onnxruntime.InferenceSession(config['weignts_onnx']['decoder'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        self.session = (cnn_session, encoder_session, decoder_session)

        self.config = config
        # self.model = model
        self.vocab = vocab
        # self.device = device
        self.process_input = TransformInput(self.config['dataset']['image_height'],
                                            self.config['dataset']['image_min_width'], 
                                            self.config['dataset']['image_max_width'])

    def predict(self, img, return_prob=False):
        img = process_input(img, self.config['dataset']['image_height'], 
                self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])        
        img = np.array(img)

        s, (cnn_time, encoder_time, decoder_time) = translate_onnx(img, self.model)
        s = s[0].tolist()
        prob = prob[0]

        s = self.vocab.decode(s)
        
        if return_prob:
            return s, prob
        else:
            return s

    def predict_batch(self, imgs, batch_size):
        process_input_times, cnn_times, encoder_times, decoder_times = 0,0,0,0
        st = time.time()
        bucket = self.process_input(imgs)
        process_input_times = time.time() - st
        batch_sents = []
        
        for i in range(0, len(bucket), batch_size):
            batch = np.stack(bucket[i: i + batch_size], 0)
            s, (cnn_time, encoder_time, decoder_time) = translate_onnx(np.array(batch), self.session)
            batch_sents.extend(s.tolist())
            cnn_times += cnn_time
            encoder_times += encoder_time
            decoder_times += decoder_time
        return self.vocab.batch_decode(batch_sents), (process_input_times, cnn_times, encoder_times, decoder_times)
