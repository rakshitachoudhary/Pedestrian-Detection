from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import cv2
import os
import torch
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class CustomDataset(Dataset):
    def __init__(self, annotations, direc, transform=None):
        with open(os.path.join(direc, annotations)) as json_file:
            self.annotations = json.load(json_file)
        self.root_dir = direc
        self.transform = transform
        self.ind = 0

    def __len__(self):
        return len(self.annotations["images"])

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.annotations["images"][index]["file_name"])
        img = cv2.imread(path)
        ann = []
        if "annotations" not in self.annotations:
            return (img, 0)
        t = len(self.annotations["annotations"])
        while(self.ind < t and self.annotations["annotations"][self.ind]["image_id"] == index):
            ann.append(self.annotations["annotations"][self.ind])
            self.ind += 1
        return (img, ann)

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=str, help='path to the dataset root directory')
ap.add_argument('--test', type=str, help='path to test json')
ap.add_argument('--out', type=str, help='path to output json')
args = vars(ap.parse_args())

dataset = CustomDataset(annotations=args["test"], direc=args["root"])
batch_sz = dataset.__len__()
data_loader = DataLoader(dataset=dataset, batch_size=batch_sz,shuffle=False, collate_fn=my_collate)

features, labels = next(iter(data_loader))


color = [0,0,255]
model = detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91, pretrained_backbone=True)
# torch.save(model.state_dict(), os.path.join(args["root"] ,args["model"]))
model.eval()
output = []
for j in range(batch_sz):
    print("Image "+str(j))
    frame = features[j]
    orig = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    frame = torch.FloatTensor(frame)

    detections = model(frame)[0]
    for i in range(0, len(detections["boxes"])):
        confidence = detections["scores"][i]
        if confidence > 0.9:
            idx = int(detections["labels"][i])-1
            if idx != 0:
                continue
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            print("person: {:.2f}".format(confidence * 100))
            cv2.rectangle(orig, (startX, startY), (endX, endY), color, 2)
            output.append({"image_id": j, "category_id": 1, "bbox": [float(startX),float(startY),float(endX-startX),float(endY-startY)], "score":float(confidence)})
    # cv2.imwrite("partB/image"+str(j)+".png", orig)
    #cv2.imshow("Output", orig)
    #cv2.waitKey(10)

with open(os.path.join(args["root"] ,args["out"]) , 'w') as file:
    json.dump(output,file)
