import os

f=open("/home/mia_dev/xeroblade2/PyTorch-YOLOv3-master/data/custom/train.txt",'w')

g = os.walk(r"/home/mia_dev/xeroblade2/dataset/train/img")
for path,dir_list,file_list in g:
    for file_name in file_list:
        f.write(r"/home/mia_dev/xeroblade2/dataset/train/img/"+file_name)
        f.write('\n')