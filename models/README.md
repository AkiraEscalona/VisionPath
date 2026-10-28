# Detection model files

VisionPath uses **MobileNet-SSD (Caffe)** through OpenCV DNN (`cv2.dnn.readNetFromCaffe`).

* Source: https://github.com/chuanqi305/MobileNet-SSD
* Trained on PASCAL VOC; detects 20 classes: aeroplane, bicycle, bird, boat, bottle, bus, car, cat,
  chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor.
  (No stairs, doors, walls, poles, curbs or potholes - these are NOT detected.)
* Check the repository's license before redistributing the files.

Required files in this folder:

```
models/MobileNetSSD_deploy.prototxt
models/MobileNetSSD_deploy.caffemodel      (about 23 MB)
```

Automatic download:  `python scripts/download_model.py`
Verify:              `python main.py --check-model`

Manual download: open the repository above, download `deploy.prototxt` and
`MobileNetSSD_deploy.caffemodel`, save them here under the names shown.
