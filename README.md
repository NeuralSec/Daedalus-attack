# Daedalus-attack
The code of our paper "Daedalus: Breaking Non-Maximum Suppression in Object Detection via Adversarial Examples".

We propose an attack, in which we can tune the strength of the attack and specify the object category to attack, to break non-maximum suppression (NMS) in object detection. As the consequence, the detection model outputs extremely dense results as the redundant detection boxes are not filtered by NMS.

Some results are displayed here:
![Alt text](resources/l2attack.jpg)
*Adversarial examples made by our $L_2$ attack. The first row contains original images. The third row contains our low-confidence (0.3) adversarial examples. The fifth row contains our high-confidence (0.7) examples. The detection results from YOLO-v3 are in the rows below them. The confidence controls the density of the redundant detection boxes in the detection results.*

---

**Running the attack on YOLO-v3:**
1. Download [yolo.h5](https://1drv.ms/u/s!AqftEu9YAdEGidZ7vEm-4v4c2sV-Lw) and put it into '../data/';
2. Put the original images into '../Datasets/COCO/val2017/';
3. Run l2_yolov3.py.
