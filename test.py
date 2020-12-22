import os
import cv2
import time
import numpy as np
import torch
import torchvision.transforms.functional as TF
from model.lpr import BiSeNet
import argparse


def isEqual(labelGT, labelP, num_char):
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(num_char)]
    return sum(compare)


def transform(img):
    img = cv2.resize(img, (160, 50))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=2)
    img = np.concatenate((img, img, img), axis=2)
    img = TF.to_tensor(img)
    img = torch.unsqueeze(img, dim=0)
    return img


def main():
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, required=True, help='dataset to test')
    parser.add_argument('--backbone', type=str, default=None, required=True, help='backbone, select from "resnet18, resnet34, resnet50, resnet101"')
    parser.add_argument('--weightfile', type=str, default=None, required=True, help='path to weightfile')
    args = parser.parse_args()

    num_char = 7
    num_class = 35
    img_path = os.path.join(os.getcwd(), "dataset", args.dataset, 'image')
    char_path = os.path.join(os.getcwd(), "dataset", args.dataset, 'char')

    model = BiSeNet(num_class, num_char, args.backbone).cuda()
    model.load_state_dict(torch.load(args.weightfile), True)
    model.eval()

    name_list = os.listdir(img_path)
    count, correct = 0, 0
    for i, name in enumerate(name_list):
        count += 1
        name = os.path.splitext(name)[0]
        char = np.loadtxt(os.path.join(char_path, name + '.txt'), dtype=int)
        img = cv2.imread(os.path.join(img_path, name + '.png'))
        img = transform(img)
        img = img.cuda()

        with torch.no_grad():
            string_pred = model(img)
        string_pred = [x.data.cpu().numpy().tolist() for x in string_pred]
        string_pred = [y[0].index(max(y[0])) for y in string_pred]

        if isEqual(string_pred, char, num_char-1) == num_char-1:
            correct += 1

    print('precision:{}'.format(float(correct) / count))


def test_fps():
    model = BiSeNet(36, 'resnet101')
    model = model.cuda()
    model.eval()

    total = 0.0
    num = 5000
    data = torch.rand([1, 3, 50, 160]).cuda()
    with torch.no_grad():
        for i in range(100):
            print(i)
            _ = model(data)
        for i in range(num):
            print(i)
            t1 = time.time()
            _ = model(data)
            t2 = time.time()
            total += t2 - t1
    print('num:{} total_time:{}s avg_time:{}s'.format(num, total, total / num))


if __name__ == '__main__':
    main()
