import os
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import time
from flask import request
from werkzeug.utils import secure_filename
from random import random
from random import choice
import requests
import json

CUR_FOLDER = os.getcwd()
SAVED_IMAGES_FOLDER = os.path.join(CUR_FOLDER, 'saved_images')
SAVED_FEEDBACK_FOLDER = os.path.join(CUR_FOLDER, 'saved_feedback')
SAVED_PRED_FOLDER = os.path.join(CUR_FOLDER, 'saved_pred')
META_FILENAME = 'meta.json'

desktop_agents = [
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0']


def random_headers():
    return {'User-Agent': choice(desktop_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# Utils func
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file(input_field_name):
    file = request.files.get(input_field_name, None)
    id = request.form.get('mns', None)
    if file != None:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_bytes = file.read()
            # img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return filename, image_bytes, id
    return None, None, None

def download_image(image_url):
    # header = random_headers()
    # response = requests.get(image_url, headers=header, stream=True, verify=False, timeout=5)
    response = requests.get(image_url)
    image = Image.open(io.BytesIO(response.content)).convert('RGB')
    return image

def byte2image(bytes):
    return Image.open(io.BytesIO(bytes)).convert("RGB")

def image2byte(img):
    buffered = io.BytesIO()
    img_base64 = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img
    img_base64.save(buffered, format="JPEG")
    img_s = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_s

def image2content(img):
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)
    return img_io

def byte2str(bytes_data):
    str_decode = base64.b64encode(bytes_data)
    str_decode = str_decode.decode('utf-8')
    return str_decode

def get_image_content(url):
    try:
        response = requests.get(url)
        return response.content
    # return Image.open(io.BytesIO(response.content))
    except:
        return None

def byte2image(byte_decode):
    return Image.open(io.BytesIO(byte_decode)).convert("RGB")

def str2image(str_encode):
    str_bytes = bytes(str_encode, 'utf-8')
    byte_decode = base64.b64decode(str_bytes)
    return byte2image(byte_decode)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_image(image, id=None):
    if os.path.exists(SAVED_IMAGES_FOLDER) is False:
        os.makedirs(SAVED_IMAGES_FOLDER)
    if id is None:
        image_filename = time.strftime("%Y%m%d-%H%M%S") + '.jpg'
    else:
        image_filename = str(id) + '-' + time.strftime("%Y%m%d-%H%M%S") + '.jpg'
    saved_filepath = os.path.join(SAVED_IMAGES_FOLDER, image_filename)
    image.save(saved_filepath)
    return image_filename

def get_list_field():
    meta_file = open(META_FILENAME, 'r')
    fields = json.load(meta_file)
    list_field = []
    for field in fields:
        list_field.append(str(field["id"])+ "_" + field["name"])
    return list_field

def save_feedback(filename, feedback):
    if os.path.exists(SAVED_FEEDBACK_FOLDER) is False:
        os.makedirs(SAVED_FEEDBACK_FOLDER)
    json_filename = filename.replace('jpg', 'json')
    json_filepath = os.path.join(SAVED_FEEDBACK_FOLDER, json_filename)
    with open(json_filepath, 'w', encoding='utf8') as f:
        json.dump(feedback, f, indent=4, ensure_ascii=False)

def save_pred(filename, pred):
    if os.path.exists(SAVED_PRED_FOLDER) is False:
        os.makedirs(SAVED_PRED_FOLDER)
    json_filename = filename.replace('jpg', 'json')
    json_filepath = os.path.join(SAVED_PRED_FOLDER, json_filename)
    with open(json_filepath, 'w', encoding='utf8') as f:
        json.dump(pred, f, indent=4, ensure_ascii=False)
