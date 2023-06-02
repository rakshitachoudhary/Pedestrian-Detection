import cv2
import os
import numpy as np
import argparse, json
import imutils
from imutils.object_detection import non_max_suppression

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=str, help='path to the dataset root directory')
ap.add_argument('--test', type=str, help='path to test json')
ap.add_argument('--out', type=str, help='path to output json')
args = vars(ap.parse_args())

test_file = {}
with open(os.path.join(args["root"], args["test"])) as json_file:
    test_file = json.load(json_file)

num_images = len(test_file["images"])

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

output = []
for f in range(0,num_images):
    print("frame: ",f)
    frame = cv2.imread(os.path.join(args["root"], test_file["images"][f]["file_name"]),1)
    clone = frame.copy()
    boxes, weights = hog.detectMultiScale(frame, winStride=(3,3),padding=(8, 8), scale=1.05)
    boxes = np.array([[x_start, y_start, x_start + width, y_start + height] for (x_start, y_start, width, height) in boxes])
    boxes = non_max_suppression(boxes, probs=None, overlapThresh=0.3)
    softmax_weights = softmax(weights)
    ind = 0
    for (x_start, y_start, x_end, y_end) in boxes:
        output.append({"image_id": f, "category_id": 1, "bbox": [float(x_start),float(y_start),float(x_end-x_start),float(y_end-y_start)], "score":float(softmax_weights[ind][0])})
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end),(0, 255, 0), 2)
        ind += 1
    # cv2.imwrite("partA/image"+str(f)+".png", frame)
    cv2.imshow("cs",frame)
    cv2.waitKey(10)
cv2.destroyAllWindows()
with open(os.path.join(args["root"] ,args["out"]) , 'w') as file:
    json.dump(output,file)