import os
from flask import Flask, jsonify, abort, send_file, request, send_from_directory, jsonify
import torch
import json
# from PIL import Image
import time
import yaml

from utils import get_file, byte2image, download_image, image2content, save_image, save_feedback, save_pred, get_list_field, byte2str, image2byte
from modules import Predictor

# Define const
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

API_METHOD_INPUT = {
    'GET': 'link',
    'POST': 'image'
}

RESPONSE_TEMPLATE = {
    'status': '1',
    'code': '200',
    'message': '',
    'data': {
        'images': [],
        'results': []
    }
}

app = Flask(__name__)

@app.route("/")
def home():
    return "OCR Visa"


@app.route("/image", methods=['GET', 'POST'])
def ocr_image():
    response = {
        'status': '1',
        'code': '200',
        'message': '',
        'data': {
            'images': [],
            'results': []
        }
    }
    start = time.time()
    if request.method not in API_METHOD_INPUT.keys():
        abort(405)
    input_field_name = API_METHOD_INPUT[request.method]
    image_url = None
    id = None
    if request.method == "POST":
        _, file_content, id = get_file(input_field_name)
        if file_content == None:
            response['status'] = '0'
            response['code'] = '200'
            response['message'] = 'Can not found image content'
            return response
        image = byte2image(file_content)
    else:
        image_url = request.args.get(input_field_name, default='', type=str)
        print('url: ',image_url)
        image = download_image(image_url)
    width, height = image.size
    if width > MAX_WIDTH:
        new_size = (MAX_WIDTH, int(MAX_WIDTH/width*height))
        print('resize: ', new_size)
        image = image.resize(new_size)

    kie, out_image, module_time = predictor.predict(image, 'test', kie='rule', is_draw_image=True)
    io_image = image2content(out_image)
    end = time.time()
    print('process time:', end - start)
    return send_file(io_image, mimetype='image/jpeg')

@app.route("/json", methods=['GET', 'POST'])
def ocr_json():
    start = time.time()
    if request.method not in API_METHOD_INPUT.keys():
        abort(405)
    input_field_name = API_METHOD_INPUT[request.method]
    image_url = None
    id = None
    response = {
        'status': '1',
        'code': '200',
        'message': '',
        'data': {
            'images': [],
            'results': []
        }
    }
    if request.method == "POST":
        _, file_content, id = get_file(input_field_name)
        if file_content == None:
            response['status'] = '0'
            response['code'] = '200'
            response['message'] = 'Can not found image content'
            return response
        image = byte2image(file_content)
    else:
        try:
            image_url = request.args.get(input_field_name, default='', type=str)
            print('url: ',image_url)
            image = download_image(image_url)
        except Exception as ex:
            response['status'] = '0'
            response['code'] = '200'
            response['message'] = str(ex)
            return response
    try:
        saved_image_filename = save_image(image, id)
        print('filename: ', saved_image_filename)
        width, height = image.size
        if width > MAX_WIDTH:
            new_size = (MAX_WIDTH, int(MAX_WIDTH/width*height))
            print('resize: ', new_size)
            image = image.resize(new_size)
        kie, out_image, module_time = predictor.predict(image, saved_image_filename, kie='rule')
        response['id'] = saved_image_filename
        response['data']['images'].append(
            {
                'base64': image2byte(out_image),
            }
        )
        response['data']['results'] = [
            {'key': k, 'value': v} for k, v in kie.items()
        ]
        print(response['data']['results'])
        save_pred(saved_image_filename, kie)

    except Exception as ex:
        response['status'] = '0'
        response['code'] = '200'
        response['message'] = str(ex)
        return response

    response['message'] = 'Success'
    end = time.time()
    print('process time:', end - start)
    return response

if __name__ == '__main__':

    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)
    try:
        import setproctitle
        setproctitle.setproctitle(config['service']['name'])
    except (ImportError, AttributeError):
        pass 
    print(config)

    predictor = Predictor(config)
    MAX_WIDTH = config['service']['max_width']
    
    app.run(host='0.0.0.0', port=config['service']['port'], debug=False)
