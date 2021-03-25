from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import sys

import torch
import torchvision.transforms as transforms

from PIL import Image
from utils.config import cfg
from modeling.model_builder import create
from datasets.encoder import DataEncoder
from datasets.transform import resize
import numpy as np
import cv2
category = ['nopon']

def test_model():
    """Model testing loop."""
    logger = logging.getLogger(__name__)
    colors = np.random.randint(0, 255, size=(1, 3), dtype="uint8")

    model = create(cfg.MODEL.TYPE, cfg.MODEL.CONV_BODY, cfg.MODEL.NUM_CLASSES)
    checkpoint = torch.load(os.path.join('checkpoint', cfg.TEST.WEIGHTS))
    model.load_state_dict(checkpoint['net'])


    if not torch.cuda.is_available(): 
        logger.info('cuda not find')
        sys.exit(1)

    #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    #model.cpu()
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])

    img_dir = os.path.join("/home/mia_dev/xeroblade2/dataset/train/img/")
    g = os.walk(r"/home/mia_dev/xeroblade2/dataset/test/2021-03-22_21-49-34")
    img_list=[]
    for path, dir_list, file_list in g:
        for file_name in file_list:
            img_list.append(file_name)
    img_nums = len(img_list)

    test_scales = cfg.TEST.SCALES
    dic = {}
    for i in range(20) : 
        dic[str(i)] = []

    for im in range(img_nums):
        if im % 100 == 0 : logger.info('{} imgs were processed, total {}'. format(im, img_nums))
        img = Image.open(os.path.join("/home/mia_dev/xeroblade2/dataset/test/2021-03-22_21-49-34/", img_list[im].strip()))
        print(os.path.join("/home/mia_dev/xeroblade2/dataset/test/2021-03-22_21-49-34/", img_list[im].strip()))
        img_size = img.size
        img = img.resize(test_scales)

        x = transform(img)
        x=x.cuda()
        x = x.unsqueeze(0)
        x = torch.autograd.Variable(x)
        loc_preds, cls_preds = model(x)

        loc_preds = loc_preds.data.squeeze().type(torch.FloatTensor)
        cls_preds = cls_preds.data.squeeze().type(torch.FloatTensor)

        encoder = DataEncoder(test_scales)
        boxes, labels, sco, is_found = encoder.decode(loc_preds, cls_preds, test_scales)
        if is_found :
            img, boxes = resize(img, boxes, img_size)

            boxes = boxes.ceil()
            xmin = boxes[:, 0].clamp(min = 1)
            ymin = boxes[:, 1].clamp(min = 1)
            xmax = boxes[:, 2].clamp(max = img_size[0] - 1)
            ymax = boxes[:, 3].clamp(max = img_size[1] - 1)

            nums = len(boxes)
            print(nums)
            for i in range(nums) : 
                dic[str(labels[i].item())].append([img_list[im].strip(), sco[i].item(), xmin[i].item(), ymin[i].item(), xmax[i].item(), ymax[i].item()])

    temp=''
    for key in dic.keys() : 
        #logger.info('category id: {}, category name: {}'. format(key, category[int(key)]))
        #file_name = cfg.TEST.OUTPUT_DIR + 'comp4_det_test_'+category[int(key)]+'.txt'
        #with open(file_name, 'w') as comp4 :
            nums = len(dic[key])
            for i in range(nums) : 
                img, cls_preds, xmin, ymin, xmax, ymax = dic[key][i]

                if temp!=img:
                  temp=img
                  imgs = cv2.imread("/home/mia_dev/xeroblade2/dataset/test/2021-03-22_21-49-34/" + img)
                else:
                  imgs=imgs
                print(cls_preds)
                if cls_preds > 0 :
                    cls_preds = '%.6f' % cls_preds
                    loc_preds = '%.6f %.6f %.6f %.6f' % (xmin, ymin, xmax, ymax)
                    rlt = '{} {} {}\n'.format(img, cls_preds, loc_preds)
                    #comp4.write(rlt)

                    box_w = xmax - xmin
                    box_h = ymax - ymin
                    color = [int(c) for c in colors[0]]
                    print(box_w, box_h)
                    box_w=int(box_w)
                    box_h=int(box_h)
                    # print(cls_conf)
                    x1=int(xmin)
                    x2=int(xmax)
                    y1=int(ymin)
                    y2=int(ymax)

                    imgs = cv2.rectangle(imgs, (x1, y1 + box_h), (x2, y1), color, 2)
                    cv2.putText(imgs, 'nopon', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(imgs, str("%.2f" % float(cls_preds)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)
                    print("/home/mia_dev/xeroblade2/dataset/result/retinanet/"+img.split('/')[-1])
                    cv2.imwrite("/home/mia_dev/xeroblade2/dataset/result/retinanet/"+img.split('/')[-1],imgs)
