## Instructions to run

1. Navigate to the directory containing source code `src`
2. Open a terminal. 
    1. Pretrained HOG: Use the command `python3 eval_hog_pretrained.py --root <path to dataset root directory> -- test <path to test json> --out <path to output json>`
    2. Custom HOG: Use the command `python eval_hog_custom.py --root <path to dataset root directory> -- test <path to test json> -- train <path to train json> --out <path to output json>`
    3. Faster-RCNN: Use the command: `python eval_faster_rcnn.py --root <path to dataset root directory> -- test <path to test json> --out <path to output json>`
    
Note: `<path to dataset root directory>` corresponds to the directory where `PennFudanPed` folder is located.
