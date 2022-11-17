import sys

sys.path.append('./modules/text_line_recognition')

import argparse
from vietocr.tool.utils import compute_accuracy
import os
from jiwer import wer,cer

def precision(actual_sents, pred_sents):

    acc_full_seq = compute_accuracy(actual_sents, pred_sents, mode='full_sequence')
    acc_per_char = compute_accuracy(actual_sents, pred_sents, mode='per_char')

    return acc_full_seq, acc_per_char

def get_cer_wer_fullseqacc(actual_sents, pred_sents):
    total_wer = 0
    total_cer = 0
    total_fullseq_acc = 0
    for (actual_sent, pred_sent) in zip(actual_sents, pred_sents):
        actual_sent = actual_sent.lower()
        pred_sent = pred_sent.lower()
        if actual_sent == pred_sent:
            total_fullseq_acc += 1
        else:
            print('Wrong seq', actual_sent, pred_sent)
        actual_list = actual_sent.split(' ')
        total_cer += cer(actual_sent, pred_sent)
        print('{} {}, CER: {}'.format(actual_sent, pred_sent, cer(actual_sent, pred_sent)))
        total_wer += wer(actual_sent, pred_sent)
        print('{} {}, WER: {}'.format(actual_sent, pred_sent, wer(actual_sent, pred_sent)))
    total_fullseq_acc = total_fullseq_acc / len(actual_sents) * 100
    return total_cer/len(actual_sents), total_wer/len(actual_sents), total_fullseq_acc
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='input file predict', default='predict.txt') #format line: file_name /t gt /t predict
    args = parser.parse_args()

    gts = []
    preds = []

    lines = open(args.input, "r")
    for line in lines:
        line = line.split('\t')
        print(line)
        gts.append(line[1])
        preds.append(line[2].replace('\n', ''))

    cer, wer, full_seq_acc = get_cer_wer_fullseqacc(gts,preds)
    print('cer: ', cer)
    print('wer: ', wer)

    print('full seq accuracy: ', full_seq_acc)

    print('done')

if __name__ == '__main__':
    main()
