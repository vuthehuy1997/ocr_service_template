from vietocr.tool.translate import build_model, translate, translate_beam_search, process_input, predict
from vietocr.tool.utils import download_weights

import torch
from collections import defaultdict

import math
import numpy as np
from torchvision.transforms import functional as F
import time

class TransformInput():
    def __init__(self, height, min_width, max_width, device):
        self.height = height
        self.min_width = min_width
        self.max_width = max_width
        self.device = device

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
        img = F.to_tensor(image).to(self.device)
        h, w = img.shape[1:]

        width_size = self.get_size(w, h)
        img = F.resize(img, (self.height, width_size))#, antialias = True)

        return img
    
    def __call__(self, imgs):
        output = [self.transform(img) for img in imgs]
        max_width = max(i.shape[-1] for i in output)
        for i in range(len(output)):
            w = output[i].shape[-1]
            output[i] = F.pad(output[i], [0, 0, max_width - w, 0], fill = 0, padding_mode = 'constant')
        return output
class Predictor():
    def __init__(self, config):

        device = config['device']
        
        model, vocab = build_model(config)
        weights = '/tmp/weights.pth'

        if config['weights'].startswith('http'):
            weights = download_weights(config['weights'])
        else:
            weights = config['weights']

        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))

        self.config = config
        self.model = model
        self.vocab = vocab
        self.device = device
        self.process_input = TransformInput(self.config['dataset']['image_height'],
                                            self.config['dataset']['image_min_width'], 
                                            self.config['dataset']['image_max_width'],
                                            self.device)

    def predict(self, img, return_prob=False):
        img = process_input(img, self.config['dataset']['image_height'], 
                self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])        
        img = img.to(self.config['device'])

        if self.config['predictor']['beamsearch']:
            sent = translate_beam_search(img, self.model)
            s = sent
            prob = None
        else:
            s, prob, _ = translate(img, self.model)
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
        # print('---time process_input: ', time.time()-st)
        batch_sents = []
        batch_probs = []
        for i in range(0, len(bucket), batch_size):
            
            batch = torch.stack(bucket[i: i + batch_size], 0).to(self.device, non_blocking=True) # 1 gpu
            # st = time.time()
            s, p, (cnn_time, encoder_time, decoder_time) = translate(batch, self.model)
            # print('---time batch {}: {}'.format(i, time.time()-st))
            batch_sents.extend(s.tolist())
            batch_probs.append(p.tolist())
            cnn_times += cnn_time
            encoder_times += encoder_time
            decoder_times += decoder_time
        return self.vocab.batch_decode(batch_sents), batch_probs, (process_input_times, cnn_times, encoder_times, decoder_times)
