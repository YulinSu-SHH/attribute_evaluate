from cProfile import label
import json
import os
import torch
import argparse
import numpy as np
from functools import reduce
from sklearn.metrics import precision_score, recall_score, f1_score
from prettytable import PrettyTable
import time
import json
import pandas as pd


def multilabel_accuracy(output, target, thresh=0.5):
    batch_size = target.size(0)

    output = torch.nn.Sigmoid()(output)
    output[output >= thresh] = 1
    output[output < thresh] = 0
    
    correct = torch.sum(torch.all(torch.eq(output, target), dim=0))
    return [100 * correct / batch_size]


def precision(output, target):
    prediction = output.argmax(dim=1)
    score = torch.FloatTensor([precision_score(target, prediction, average='macro')])
    res = [score * 100]
    return res


def recall(output, target):
    prediction = output.argmax(dim=1)
    score = torch.FloatTensor([recall_score(target, prediction, average='macro')])
    res = [score * 100]
    return res


def f1(output, target):
    prediction = output.argmax(dim=1)
    score = torch.FloatTensor([f1_score(target, prediction, average='macro')])
    res = [score * 100]
    return res


class ImageNetEvaluator(object):
    def __init__(self,
                 topk=[1, 3],
                 multilabel=False,):
        super(ImageNetEvaluator, self).__init__()
        self.topk = topk
        self.multilabel = multilabel


    def load_res(self, res_file):
        res_dict = {}
        with open(res_file) as f:
                lines = f.readlines()
        for line in lines:
            info = json.loads(line)
        
            for key in info.keys():
                if key not in res_dict.keys():
                    res_dict[key] = [info[key]]
                else:
                    res_dict[key].append(info[key])
        return res_dict

    def eval(self, gt_file, pred_file):
        gt_dict = self.load_res(gt_file)
        pred_dict = self.load_res(pred_file)
        gt, pred = gt_dict['gt'],pred_dict['pred']
        gt = torch.from_numpy(np.array(gt))
        pred = torch.from_numpy(np.array(pred))
        res_dict={}
        res_dict['attribute']=gt_dict['attribute'][0]
        res_dict['labels']=gt_dict['labels'][0]
        if not self.multilabel:
            for k in self.topk:
                n_k = min(pred.size(1), k)
                idx_pred_descending=torch.argsort(pred,1,descending=True)
                idx_gt=torch.argmax(gt,1,keepdim=True).repeat(1,idx_pred_descending.size(1))
                correct_k=torch.eq(idx_pred_descending[:,:n_k],idx_gt[:,:n_k]).int()
                acc_k=torch.sum(correct_k)/pred.size(0)*100.
                res_dict[f'top{k}'] = acc_k.item()
                res_dict[f'top{k}'] =  round(res_dict[f'top{k}'], 2)
            res_dict['precision'] =  precision(pred, torch.argmax(gt,1))[0].item()
            res_dict['recall'] = recall(pred, torch.argmax(gt,1))[0].item()
            res_dict['f1'] = f1(pred, torch.argmax(gt,1))[0].item()
            res_dict['precision'] =  round(res_dict['precision'], 2)
            res_dict['recall'] =  round(res_dict['recall'], 2)
            res_dict['f1'] =  round(res_dict['f1'], 2)
        else:
            res_dict['acc'] = multilabel_accuracy(pred, gt)[0].item()

        return res_dict

def pretty_print(result,num_rows,out_file_root):
    x = PrettyTable()
    for key in result.keys():
        x.add_column(key, [result[key][attr] for attr in range(num_rows)])

    out_filename = "result.txt"
    f = open(os.path.join(out_file_root,out_filename),"a")
    current_time = time.strftime('%Y_%m_%d %H_%M_%S',time.localtime(time.time()))
    f.write(current_time+'\n')
    f.write(str(x)+'\n')
    f.close()

def save_to_csv(result,out_file_root):
    dataframe = pd.DataFrame(result)
    dataframe.to_csv(os.path.join(out_file_root,"result.csv"),index=False,sep=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_root", type=str, default="./")
    parser.add_argument("--topk", type=str, default="0,1")
    parser.add_argument("--multilabel", type=bool, default=False)
    args = parser.parse_args()

    pred_file_root=os.path.join(args.file_root,'pred')
    gt_file_root=os.path.join(args.file_root,'gt')
    out_file_root=os.path.join(args.file_root,'result')
    if not os.path.exists(out_file_root):
        os.makedirs(out_file_root)

    pred_file_path,gt_file_path=[],[]

    for root, dirs, files in os.walk(pred_file_root):
        for file in files:
            attr=file.split('_', 1)[1][:-6]
            gt_filename="gt_"+attr+".jsonl"
            pred_file_path.append(os.path.join(pred_file_root, file))
            gt_file_path.append(os.path.join(gt_file_root, gt_filename))

    result={}
    my_evaluate=ImageNetEvaluator(topk=args.topk,multilabel=args.multilabel)
    for attr in range(len(pred_file_path)):      
        metric=my_evaluate.eval(gt_file_path[attr],pred_file_path[attr])
        for key in metric.keys():
            if key not in result.keys():
                result[key]=[metric[key]]
            else:
                result[key].append(metric[key])
    
    pretty_print(result,len(pred_file_path),out_file_root)
    save_to_csv(result,out_file_root)

