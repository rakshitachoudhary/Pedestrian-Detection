import cv2
import os
import numpy as np
import copy
import random
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skimage.feature import hog
import argparse, json
from joblib import dump, load


ap = argparse.ArgumentParser()
ap.add_argument('--root', type=str, help='path to the dataset root directory')
ap.add_argument('--train', type=str, help='path to train json')
ap.add_argument('--test', type=str, help='path to test json')
ap.add_argument('--out', type=str, help='path to output json')
args = vars(ap.parse_args())

train_file = {}
test_file = {}
with open(os.path.join(args["root"], args["train"])) as json_file:
    train_file = json.load(json_file)
with open(os.path.join(args["root"], args["test"])) as json_file2:
    test_file = json.load(json_file2)

num_images = len(train_file["images"])
num_images_test = len(test_file["images"])

X=[]
Y=[]

print("Preparing Training data – positive and negative samples")
def checkOverlap(x1_start,y1_start,x1_end,y1_end,x2_start,y2_start,x2_end,y2_end):
    if(x1_start > x2_end or x2_start > x1_end):
        return False
    if(y1_start > y2_end or y2_start > y1_end):
        return False
    return True

templatewidth=0
templateheight=0
numpositivetemplates=0

ind = 0
for f in range(0,num_images):
    frame = cv2.imread(os.path.join(args["root"], train_file["images"][f]["file_name"]),0)
    temp_frame=copy.deepcopy(frame)
    annotations = {}
    with open(os.path.join(args["root"], args["train"])) as json_file:
        annotations = json.load(json_file)
    positive=[]
    negetive=[]
    t = len(annotations["annotations"])
    while(ind < t and annotations["annotations"][ind]["image_id"] == f):
        box = annotations["annotations"][ind]["bbox"]
        x_start=int(box[0])
        y_start=int(box[1])
        x_end=int(x_start+box[2])
        y_end=int(y_start+box[3])
        positive.append([x_start,y_start,x_end,y_end])
        ind+=1
        numpositivetemplates += 1
        temp = copy.deepcopy(frame[y_start:y_end, x_start:x_end])
        X.append(temp)
        templatewidth+=x_end-x_start
        templateheight+=y_end-y_start
        Y.append(1)
    height, width = frame.shape
    for i in range(len(positive)):
        x_start,x_end,y_start,y_end=0,0,height,width
        t=True
        count=0
        while(t):
            count+=1
            x_start=random.randint(0, width-66)
            x_end=random.randint(x_start+64, width-1)
            y_start=random.randint(0, height-170)
            y_end=random.randint(y_start+168, height-1)
            tt=False
            for j in range(len(positive)):
                if checkOverlap(x_start,y_start,x_end,y_end,positive[j][0],positive[j][1],positive[j][2],positive[j][3]):
                    tt=True
            if not tt or count>500:
                t=False
        if count>10000:
            continue
        negetive.append([x_start,y_start,x_end,y_end])
        temp = copy.deepcopy(frame[y_start:y_end, x_start:x_end])
        X.append(temp)
        Y.append(0)
print("Training Data Preparation completed")



for i in range(len(X)):
    if i%50==0:
        print("Extracting hog features of training samples: ",i*100/len(X),"% done")
    X[i]=cv2.resize(X[i],(264,100))
    FeatureDescripter=hog(X[i],orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys',visualize=False,transform_sqrt=False,feature_vector=True)
    X[i]=FeatureDescripter

print("Extraction of HOG features of training samples completed")


# clf = load('svm.joblib')
clf = SVC(kernel='linear')
clf.fit(X, Y)
dump(clf, 'svm.joblib')


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def sliding_window(img, stepSize, windSize):
    for y in range(0, img.shape[0], stepSize):
        for x in range(0, img.shape[1], stepSize):
            yield (x, y, img[y: y + windSize[1], x:x + windSize[0]])

output = []
for f in range(0,num_images_test):
    print("frame: ",f)
    img = cv2.imread(os.path.join(args["root"], test_file["images"][f]["file_name"]),0)
    scale = 0
    detections = []
    windowSize=[100,264]
    downscale=1.5
    for resized in pyramid_gaussian(img, downscale=1.5): 
        if resized.shape[0] < windowSize[1] or resized.shape[1] < windowSize[0]:
            break
        for (x,y,window) in sliding_window(resized, stepSize=25, windSize=(100,264)):
            if window.shape[0] != windowSize[1] or window.shape[1] !=windowSize[0]:
                continue
            window=cv2.resize(window,(264,100))
            FeatureDescripter = hog(window,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys',visualize=False,transform_sqrt=False,feature_vector=True)  
            FeatureDescripter = FeatureDescripter.reshape(1, -1) 
            pred = clf.predict(FeatureDescripter) 
            if pred == 1:
                if clf.decision_function(FeatureDescripter) > 0.8:  
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), clf.decision_function(FeatureDescripter),int(windowSize[0]*(downscale**scale)),int(windowSize[1]*(downscale**scale))))
        scale+=1
        
    clone = copy.deepcopy(img)
    rects = np.array([[x_start, y_start, x_start + width, y_start + height] for (x_start, y_start, confidenceScore, width, height) in detections])
    scores = [confidenceScore[0] for (x_start, y_start, confidenceScore, width, height) in detections]
    #print("Confidence Score of Detections: ", scores)
    scores = np.array(scores)
    nms = non_max_suppression(rects, probs = scores, overlapThresh = 0.2)

    num = 0
    for (x_start, y_start, x_end, y_end) in nms:
        cv2.rectangle(clone, (x_start, y_start), (x_end, y_end), (0,255,0), 2)
        output.append({"image_id": f, "category_id": 1, "bbox": [float(x_start),float(y_start),float(x_end-x_start),float(y_end-y_start)], "score":float(scores[num])})
        num+=1
    #cv2.imshow("Output", clone)
    # cv2.imwrite("partC/image"+str(f)+".png", clone)
    #cv2.waitKey(1) 

cv2.destroyAllWindows()
with open(os.path.join(args["root"] ,args["out"]) , 'w') as file:
    json.dump(output,file)
