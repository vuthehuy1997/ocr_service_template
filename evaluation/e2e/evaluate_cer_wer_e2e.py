import argparse
import os
import json
from jiwer import wer,cer


def get_cer_wer(actual_sent, pred_sent, key):
    actual_sent = actual_sent.lower()
    pred_sent = pred_sent.lower()
    # actual_sent = actual_sent.replace('.', ' ').replace(',', ' ')
    # actual_sent = ' '.join(actual_sent.split())
    # pred_sent = pred_sent.replace('.', ' ').replace(',', ' ')
    # pred_sent = ' '.join(pred_sent.split())

    # actual_sent = normal_str(actual_sent.lower())
    # pred_sent = normal_str(pred_sent.lower())
    
    # cer = cer(actual_sent, pred_sent)
    # wer = wer(actual_sent, pred_sent)
    # ser = actual_sent != pred_sent
    val_cer = min(1,cer(actual_sent, pred_sent))
    val_wer = min(1,wer(actual_sent, pred_sent))
    val_ser = actual_sent != pred_sent
    if actual_sent != pred_sent:
        print('({}) [{}] <> [{}]'.format(key, actual_sent, pred_sent))
    return val_cer, val_wer, val_ser

def normal_str(input_string):
    assert type(input_string) == str
    map_str = {'à': 'a', 'ả': 'a', 'ã': 'a', 'á': 'a', 'ạ': 'a', 'ă': 'a', 'ằ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ắ': 'a', 'ặ': 'a', 'â': 'a', 'ầ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ấ': 'a', 'ậ': 'a', 'À': 'A', 'Ả': 'A', 'Ã': 'A', 'Á': 'A', 'Ạ': 'A', 'Ă': 'A', 'Ằ': 'A', 'Ẳ': 'A', 'Ẵ': 'A', 'Ắ': 'A', 'Ặ': 'A', 'Â': 'A', 'Ầ': 'A', 'Ẩ': 'A', 'Ẫ': 'A', 'Ấ': 'A', 'Ậ': 'A', 'đ': 'd', 'Đ': 'D', 'è': 'e', 'ẻ': 'e', 'ẽ': 'e', 'é': 'e', 'ẹ': 'e', 'ê': 'e', 'ề': 'e', 'ể': 'e', 'ễ': 'e', 'ế': 'e', 'ệ': 'e', 'È': 'E', 'Ẻ': 'E', 'Ẽ': 'E', 'É': 'E', 'Ẹ': 'E', 'Ê': 'E', 'Ề': 'E', 'Ể': 'E', 'Ễ': 'E', 'Ế': 'E', 'Ệ': 'E', 'ì': 'i', 'ỉ': 'i', 'ĩ': 'i', 'í': 'i', 'ị': 'i', 'Ì': 'I', 'Ỉ': 'I', 'Ĩ': 'I', 'Í': 'I', 'Ị': 'I', 'ò': 'o', 'ỏ': 'o', 'õ': 'o', 'ó': 'o', 'ọ': 'o', 'ô': 'o', 'ồ': 'o', 'ổ': 'o', 'ỗ': 'o', 'ố': 'o', 'ộ': 'o', 'ơ': 'o', 'ờ': 'o', 'ở': 'o', 'ỡ': 'o', 'ớ': 'o', 'ợ': 'o', 'Ò': 'O', 'Ỏ': 'O', 'Õ': 'O', 'Ó': 'O', 'Ọ': 'O', 'Ô': 'O', 'Ồ': 'O', 'Ổ': 'O', 'Ỗ': 'O', 'Ố': 'O', 'Ộ': 'O', 'Ơ': 'O', 'Ờ': 'O', 'Ở': 'O', 'Ỡ': 'O', 'Ớ': 'O', 'Ợ': 'O', 'ù': 'u', 'ủ': 'u', 'ũ': 'u', 'ú': 'u', 'ụ': 'u', 'ư': 'u', 'ừ': 'u', 'ử': 'u', 'ữ': 'u', 'ứ': 'u', 'ự': 'u', 'Ù': 'U', 'Ủ': 'U', 'Ũ': 'U', 'Ú': 'U', 'Ụ': 'U', 'Ư': 'U', 'Ừ': 'U', 'Ử': 'U', 'Ữ': 'U', 'Ứ': 'U', 'Ự': 'U', 'ỷ': 'y', 'ỹ': 'y', 'ý': 'y', 'ỵ': 'y', 'Ỷ': 'Y', 'Ỹ': 'Y', 'Ý': 'Y', 'Ỵ': 'Y'}
    return input_string.translate(str.maketrans(map_str))
    
def main():
    from jiwer import wer,cer
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True, help='gt folder') 
    parser.add_argument('--pred', required=True, help='pred folder')
    args = parser.parse_args()

    gt_path = args.gt
    pred_path = args.pred


    file_names = [i for i in os.listdir(gt_path) if os.path.splitext(i)[1] == '.json']
    file_names.sort()

    all_cer = 0
    all_wer = 0
    all_ser = 0
    total_text = 0
    all_cer_dict = {}
    all_wer_dict = {}
    all_ser_dict = {}
    total_text_dict = {}

    for file_name in file_names:
        print('---------------------------------------------------')
        print(file_name)
        with open(os.path.join(gt_path, file_name), "r") as f:
            gt = json.load(f)  
        # assert os.path.exists(os.path.join(pred_path, file_name)), \
        #         "Can not find {} in predict folder".format(os.path.join(pred_path, file_name))
        if os.path.exists(os.path.join(pred_path, file_name)):
            with open(os.path.join(pred_path, file_name), "r") as f:
                pred = json.load(f)  
        else:
            pred = {}
            for key in gt:
                pred[key] = ''

        file_cer = 0
        file_wer = 0
        file_ser = 0
        total_file_text = 0
        for key in gt:
            # if key in [
            #     'Kinh doanh vận tải (Commercial Use)',
            #     'Cải tạo (Modification)',
            #     'Có lắp thiết bị giám sát hành trình (Equipped with Tachograph)',
            #     'Có lắp camera / (Equipped with camera)',
            #     'Không cấp tem kiểm định (Inspection stamp was not issued)'
            # ]:
            #     continue
            if len(gt[key]) != 0:
                cer, wer, ser = get_cer_wer(gt[key],pred[key], key)
                file_cer += cer
                file_wer += wer
                file_ser += ser
                
                total_file_text += 1
                if ser != 0:
                    print(key)
                    print('----- cer : {}'.format(cer))
                    print('----- wer: {}'.format(wer))
                    print('----- ser: {}'.format(ser))

                if key in all_cer_dict:
                    all_cer_dict[key] += cer
                    all_wer_dict[key] += wer
                    all_ser_dict[key] += ser
                    total_text_dict[key] += 1
                else:
                    all_cer_dict[key] = cer
                    all_wer_dict[key] = wer
                    all_ser_dict[key] = ser
                    total_text_dict[key] = 1


        if (total_file_text != 0):
            print('File: ')
            print('----- file cer : {}'.format(file_cer/total_file_text))
            print('----- file wer: {}'.format(file_wer/total_file_text))
            print('----- file ser: {}'.format(file_ser/total_file_text))
            print('----- wer > {}: {}'.format(0.15, file_wer/total_file_text > 0.15 ))
        all_cer += file_cer
        all_wer += file_wer
        all_ser += file_ser
        total_text += total_file_text
    
    print('-------------------------------\n ERROR ON KEY:')
    for key in all_cer_dict:
        print(key)
        print('----- number text: ', total_text_dict[key])
        print('----- total cer : {}'.format(all_cer_dict[key]/total_text_dict[key]))
        print('----- total wer: {}'.format(all_wer_dict[key]/total_text_dict[key]))
        print('----- total ser: {}'.format(all_ser_dict[key]/total_text_dict[key]))


    print('All')
    print('Number file: ', len(file_names))
    print('Number text: ', total_text)
    if (total_text != 0):
        print('----- total cer : {}'.format(all_cer/total_text))
        print('----- total wer: {}'.format(all_wer/total_text))
        print('----- total ser: {}'.format(all_ser/total_text))

if __name__ == '__main__':
    main()
    