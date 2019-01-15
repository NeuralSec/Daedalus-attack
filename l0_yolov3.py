import os

# supress tensorflow logging other than errors
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys

from keras import backend as K
import numpy as np
import random as rd
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras.models import Model
from keras import losses

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.datasets import cifar10
from keras.models import load_model
from keras.callbacks import EarlyStopping

from model.yolo_model import YOLO
import cv2
import matplotlib.pyplot as plt
import time
from tensorflow.python import debug as tf_debug
from skimage import io

YOLO_OUT_SHAPE = (13, 13, 3, 85)  # yolo output shape
IMAGE_SHAPE = (1, 416, 416, 3)  # input image shape

MAX_ITERATIONS = 10000  # number of iterations to perform gradient descent
ABORT_EARLY = True  # abort gradient descent upon first valid solution
LEARNING_RATE = 1e-2  # larger values converge faster to less accurate results

lower_bound = 0
INITIAL_CONST = 1e2
LARGEST_CONST = 1e10
REDUCE_CONST = True  # try to lower c each iteration; faster to set to false
CONST_FACTOR = 2.0  # f>1, rate at which we increase constant, smaller better
EXAMPLE_NUM = 10

# ============runing setting==================
CONFIDENCE = 0.3
MAX_SEARCH = 5
START_FROM = 0
CUDA_GPU = '2'

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_GPU

PATH = 'adv_examples/L0/f3_eval/test/{0} confidence'.format(CONFIDENCE)

def get_classes(file):
    """Get classes name.

    # Argument:
        file: classes name for database.

    # Returns
        class_names: List, classes name.

    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names

file = 'data/coco_classes.txt'
all_classes = get_classes(file)

def process_image(img):
    """
    Resize, reduce and expand image.
    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416), interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)
    return image


def process_yolo_output(out, anchors, mask):
    """
    Tensor op: Process output features.
    # Arguments
        out - tensor (?, N, N, 3, 4 + 1 +80), output feature map of yolo.
        anchors - List, anchors for box.
        mask - List, mask for anchors.

    # Returns
        boxes - tensor (N, N, 3, 4), x,y,w,h for per box.
        box_confidence - tensor (N, N, 3, 1), confidence for per box.
        box_class_probs - tensor (N, N, 3, 80), class probs for per box.
    """
    grid_h, grid_w, num_boxes = map(int, out.shape[1: 4])

    anchors = [anchors[i] for i in mask]
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32, name='anchor_tensor'), [1, 1, len(anchors), 2])
    out = out[0]
    box_xy = tf.sigmoid(out[:, :, :, 0:2], name='box_xy')
    box_wh = tf.identity(tf.exp(out[:, :, :, 2:4]) * anchors_tensor, name='box_wh')

    box_confidence = tf.sigmoid(out[:, :, :, 4:5], name='objectness')
    box_class_probs = tf.sigmoid(out[:, :, :, 5:], name='class_probs')

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = tf.constant(np.concatenate((col, row), axis=-1), dtype=tf.float32)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (416, 416)
    box_xy -= (box_wh / 2.)

    # boxes -> (13, 13, 3, 4)
    boxes = tf.concat([box_xy, box_wh], axis=-1)
    # box_confidence -> (13, 13, 3, 1) 26 52
    # box_class_probs -> (13, 13, 3, 80)
    boxes = tf.reshape(boxes, [int(boxes.shape[0]) **
                               2, boxes.shape[2], boxes.shape[3]])
    box_confidence = tf.reshape(box_confidence,
                                [int(box_confidence.shape[0]) ** 2,
                                 box_confidence.shape[-2],
                                 box_confidence.shape[-1]])
    box_class_probs = tf.reshape(box_class_probs,
                                 [int(box_class_probs.shape[0]) ** 2,
                                  box_class_probs.shape[-2],
                                  box_class_probs.shape[-1]],
                                 name='class_probs')
    return boxes, box_confidence, box_class_probs


def process_output(raw_outs):
    """
    Tensor op: Extract b, c, and s from raw outputs.
    # Args:
        raw_outs - Yolo raw output tensor.
    #
        boxes - Tensors. (N1**2+N2**2+N3**2, 3, 4), classes: (N1**2+N2**2+N3**2, 3, 1), scores: (N1**2+N2**2+N3**2, 3, 80)
    """
    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    boxes, objecness, scores = [], [], []

    for out, mask in zip(raw_outs, masks):
        # out -> (1, 13, 13, 3, 85)
        # mask -> [6, 7, 8]
        # boxes(13X13, 3, 4), box_confidence(13X13, 3, 1),
        # box_class_probs(13X13, 3, 80) | 26 X 26 |
        b, c, s = process_yolo_output(out, anchors, mask)
        if boxes == []:
            boxes = b
            objecness = c
            scores = s
        else:
            boxes = tf.concat([boxes, b], 0, name='xywh')
            objecness = tf.concat([objecness, c], 0, name='objectness')
            scores = tf.concat([scores, s], 0, name='class_probs')
    return boxes, objecness, scores


def pdist(xy):
    """
    Tensor op: Computes pairwise distance between each pair of points
    # Args:
      xy - [N,2] matrix representing N box position coordinates
    # Content:
      dists - [N,N] matrix of (squared) Euclidean distances
    # Return:
      expectation of the Euclidean distances
    """
    xy2 = tf.reduce_sum(xy * xy, 1, True)
    dists = xy2 - 2 * tf.matmul(xy, tf.transpose(xy)) + tf.transpose(xy2)
    return tf.reduce_sum(dists)

def output_to_pdist(bx, by):
    """
    Tensor op: calculate expectation of box distance given yolo outpput bx & by.
    # Args:
        bx - YOLOv3 output batch x coordinates in shape (N, 3549, 3, 1)
        by - YOLOv3 output batch y coordinates in shape (N, 3549, 3, 1)
    """
    bxby = tf.concat([bx, by], axis=-1)
    bxby = tf.reshape(bxby, [-1, 2])
    return pdist(bxby)

def pairwise_IoUs(bs1, bs2):
    """
    Tensor op: Calculate pairwise IoUs given two sets of boxes.
    # Arguments:
        bs1, bs2 - tensor of boxes in shape (?, 4)
    # Content:
        X11,y11------x12,y11     X21,y21------x22,y21
         |              |           |            |
         |              |           |            |
        x11,y12-------x12,y12    x21,y22-------x22,y22
    # Returns:
        iou - a tensor of the matrix containing pairwise IoUs, in shape (?, ?)
    """
    x11, y11, w1, h1 = tf.split(bs1, 4, axis=1)  # (N, 1)
    x21, y21, w2, h2 = tf.split(bs2, 4, axis=1)  # (N, 1)
    x12 = x11 + w1
    y12 = y11 + h1
    x22 = x21 + w2
    y22 = y21 + h2
    xA = tf.maximum(x11, tf.transpose(x21))
    yA = tf.maximum(y11, tf.transpose(y21))
    xB = tf.minimum(x12, tf.transpose(x22))
    yB = tf.minimum(y12, tf.transpose(y22))
    # prevent 0 area
    interArea = tf.maximum((xB - xA + 1), 0) * tf.maximum((yB - yA + 1), 0)

    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    iou = interArea / (boxAArea + tf.transpose(boxBArea) - interArea)
    return iou


def expectation_of_IoUs(boxes):
    """
    Tensor op: Calculate the expectation given all pairwise IoUs.
    # Arguments
            boxes - boxes of objects. It takes (?, 4) shaped tensor;
    # Returns
            expt - expectation of IoUs of box pairs. Scalar tensor.
    """
    IoUs = pairwise_IoUs(boxes, boxes)
    expt = tf.reduce_mean(IoUs)
    return expt


def expectation_of_IoUs_accross_classes(boxes, box_scores):
    """
    Tensor op: Calculate IoU expectation for IoU expectations from different class.
    Arguments:
        #boxes - (3549, 3, 4) tensor output from yolo net
        #box_scores - (N1**2+N2**2+N3**2, 3, 80) tensor
    Content:
        #box_classes - (N1**2+N2**2+N3**2, 3, 1) tensor
    Returns:
        #expt_over_all_classes - The IoU expectation of box pairs over all classes.
    """
    box_classes = tf.cast(tf.argmax(box_scores, axis=-1), tf.int32, name='box_classes')
    class_counts = tf.bincount(box_classes)
    dominating_cls = tf.argmax(class_counts)
    dominating_cls = tf.cast(dominating_cls, tf.int32)
    index = tf.equal(box_classes, dominating_cls)
    index = tf.cast(index, tf.int32)
    others, dominating_boxes = tf.dynamic_partition(boxes, index, num_partitions=2, name='dynamic_partition')
    expt_over_all_classes = expectation_of_IoUs(dominating_boxes)
    
    return expt_over_all_classes


class YoloAttacker:
    """
    Daedalus adversarial example generator based on the Yolo v3 model.
    """

    def __init__(self, sess, model, learning_rate=LEARNING_RATE,
                 max_iterations=MAX_ITERATIONS, abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST, largest_const=LARGEST_CONST,
                 reduce_const=REDUCE_CONST, const_factor=CONST_FACTOR,
                 independent_channels=False, lower_bound=lower_bound, max_search = MAX_SEARCH):

        self.model = model
        self.sess = sess

        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const
        self.LARGEST_CONST = largest_const
        self.REDUCE_CONST = reduce_const
        self.const_factor = const_factor
        self.independent_channels = independent_channels

        self.grad = self.gradient_descent(sess, model)

        self.confidence = CONFIDENCE
        self.lower_bound = lower_bound
        self.max_search = max_search

        self.search_iteration = 1

    def gradient_descent(self, sess, model):

        shape = IMAGE_SHAPE

        # the variable to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np.float32))

        # the variables we're going to hold, use for efficiency
        canchange = tf.Variable(np.zeros(shape), dtype=np.float32)
        simg = tf.Variable(np.zeros(shape, dtype=np.float32))
        original = tf.Variable(np.zeros(shape, dtype=np.float32))
        timg = tf.Variable(np.zeros(shape, dtype=np.float32))
        const = tf.placeholder(tf.float32, [])

        # and the assignment to set the variables
        assign_modifier = tf.placeholder(np.float32, shape)
        assign_canchange = tf.placeholder(np.float32, shape)
        assign_simg = tf.placeholder(np.float32, shape)
        assign_original = tf.placeholder(np.float32, shape)
        assign_timg = tf.placeholder(np.float32, shape)

        # these are the variables to initialize when we run
        set_modifier = tf.assign(modifier, assign_modifier)
        setup = []
        setup.append(tf.assign(canchange, assign_canchange))
        setup.append(tf.assign(timg, assign_timg))
        setup.append(tf.assign(original, assign_original))
        setup.append(tf.assign(simg, assign_simg))

        newimg = ((tf.tanh(modifier + simg) + 1) / 2) * canchange + (1 - canchange) * original

        self.outs = self.model._yolo(newimg)
        # [(1, 13, 13, 3, 85), (1, 26, 26, 3, 85), (1, 52, 52, 3, 85)]
        # (3549, 3, 4), (3549, 3, 1), (3549, 3, 80) | 13 X 13 + 26 X 26 + 52 X 52
        boxes, objectness, classprobs = process_output(self.outs)

        Iou_expt = expectation_of_IoUs_accross_classes(boxes, classprobs)
        self.bx = boxes[..., 0:1]
        self.by = boxes[..., 1:2]
        self.bw = boxes[..., 2:3]
        self.bh = boxes[..., 3:4]
        self.obj_scores = objectness
        self.class_probs = classprobs
        self.box_scores = tf.multiply(self.obj_scores, tf.reduce_max(self.class_probs, axis=-1, keepdims=True))

        # # Optimisation metrics:
        self.l2dist = tf.reduce_sum(tf.square(newimg - (tf.tanh(timg) + 1) / 2), [1, 2, 3])
        self.image_sum = tf.reduce_sum(newimg)

        # Define DDoS losses: loss must be a tensor here!
        # Make the objectness of all detections to be 1.
        self.loss1_1_x = tf.reduce_mean(tf.square(self.box_scores - 1), [-3, -2, -1])           # X

        # Minimising the size of all bounding box.
        self.f1 = tf.reduce_mean(Iou_expt)
        self.f2 = tf.reduce_mean(tf.square(tf.multiply(self.bw, self.bh)), [-3, -2, -1])  # a
        self.f3 = self.f2 + 1/output_to_pdist(self.bx, self.by)

        # add two loss terms together
        self.loss_adv = self.loss1_1_x + self.f2
        loss1 = tf.reduce_mean(const * self.loss_adv)
        loss2 = tf.reduce_mean(self.l2dist)
        loss = loss1 + loss2

        outgrad = tf.gradients(loss, [modifier])[0]

        # setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        train = optimizer.minimize(loss, var_list=[modifier])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        init = tf.variables_initializer(var_list=[modifier, canchange, simg,
                                                  original, timg] + new_vars)


        def doit(oimgs, starts, valid, CONST):
            # convert to tanh-space
            imgs = np.arctanh((np.array(oimgs) * 2 - 1) * .999999)
            starts = np.arctanh((np.array(starts) * 2 - 1) * .999999)

            # initialize the variables
            sess.run(init)
            sess.run(setup, {assign_timg: imgs,
                             assign_simg: starts,
                             assign_original: oimgs,
                             assign_canchange: valid})

            while self.search_iteration <= self.max_search:
                # try solving for each value of the constant
                print('=== try const ===', CONST, "|=== search_iteration ===", self.search_iteration)
                first_flag = True
                init_adv_losses = None
                for step in range(self.MAX_ITERATIONS):
                    feed_dict = {const: CONST}

                    # remember the old value
                    oldmodifier = self.sess.run(modifier)

                    # perform the update step
                    _, works, l1= sess.run([train, loss1, self.loss_adv], feed_dict=feed_dict)
                    if first_flag:
                        init_adv_losses = l1
                        first_flag = False

                    def check_success(loss, init_loss):
                        """
                        Check if the initial loss value has been reduced by 'self.confidence' percent
                        """
                        return loss <= init_loss * (1 - self.confidence)

                    if check_success(l1, init_adv_losses) and (self.ABORT_EARLY or step == CONST - 1):
                        loss_shown, l2s, newimg_shown, l1 = sess.run([loss, loss2, newimg, self.loss_adv], feed_dict=feed_dict)
                        l0_attack_pixel = np.sum(valid)
                        # it worked previously, restore the old value and finish
                        self.sess.run(set_modifier, {assign_modifier: oldmodifier})
                        grads, scores, nimg = sess.run((outgrad, self.outs, newimg),
                                                       feed_dict=feed_dict)
                        l2s = np.array([l2s])
                        return grads, scores, nimg, CONST, l2s

                self.lower_bound = max(self.lower_bound, CONST)
                if self.LARGEST_CONST < 1e9:
                        CONST = (self.lower_bound + self.LARGEST_CONST) / 2
                else:
                        CONST *= 10
                self.search_iteration += 1

        return doit

    def attack_single(self, img):
        """
        Run the attack on a single image and label
        """

        # the pixels we can change
        valid = np.ones((1, IMAGE_SHAPE[1], IMAGE_SHAPE[2], IMAGE_SHAPE[3]))

        # the previous image
        prev = np.copy(img).reshape((1, IMAGE_SHAPE[1], IMAGE_SHAPE[2],
                                     IMAGE_SHAPE[3]))
        last_solution = np.zeros((1,416,416,3))
        last_distortion = np.zeros((1,))
        last_const = np.zeros((1,))
        const = self.INITIAL_CONST
        self.search_iteration = 1
        while True:
            # try to solve given this valid map
            res = self.grad([np.copy(img)], np.copy(prev),
                            valid, const)
            if res == None:
                # the attack failed, we return this as our final answer
                print("the attack failed, we return this as our final answer")
                return last_solution, last_distortion, last_const

            # the attack succeeded, now we pick new pixels to set to 0
            restarted = False
            gradientnorm, scores, nimg, const, l2s = res

            # save the results
            last_solution = prev = nimg
            last_distortion = l2s
            last_const = np.array([const])

            # adjust the value of const
            if self.REDUCE_CONST: 
                self.search_iteration += 1
                self.LARGEST_CONST = min(self.LARGEST_CONST, const)
                if self.LARGEST_CONST < 1e9:
                        const = (self.lower_bound + self.LARGEST_CONST) / 2
            print('*** calculate equal_count ***')
            equal_count = 416 ** 2 - np.sum(np.all(np.abs(img - nimg[0]) < .0001, axis=2))
            print("Forced equal:", np.sum(1 - valid),
                  "Equal count:", equal_count)
            if np.sum(valid) == 0:
                # if no pixels changed, return
                return [img], l2s, last_const

            if self.independent_channels:
                # we are allowed to change each channel independently
                valid = valid.flatten()
                totalchange = abs(nimg[0] - img) * np.abs(gradientnorm[0])
            else:
                # we care only about which pixels change, not channels independently
                # compute total change as sum of change for each channel
                valid = valid.reshape((IMAGE_SHAPE[1] ** 2, IMAGE_SHAPE[3]))
                totalchange = abs(np.sum(nimg[0] - img, axis=2)) * np.sum(np.abs(gradientnorm[0]), axis=2)
            totalchange = totalchange.flatten()

            # set some of the pixels to 0 depending on their total change
            did = 0
            for e in np.argsort(totalchange):
                if np.all(valid[e]):
                    did += 1
                    valid[e] = 0

                    if totalchange[e] > .01:
                        # if this pixel changed a lot, skip
                        break
                    if did >= .3 * equal_count ** .5:
                        # if we changed too many pixels, skip
                        print('we changed too many pixels, skip')
                        break

            valid = np.reshape(valid, (1, IMAGE_SHAPE[1], IMAGE_SHAPE[1], -1))
            # total nums of be masked based on l2 result
            print("Now forced equal:", np.sum(1 - valid))


    def attack(self, imgs):
        """
        Perform the L_0 attack on the given images.
        """
        r1 = []
        r2 = []
        for i, img in enumerate(imgs):
            print("Attack iteration", i)
            X_adv, dists, consts = self.attack_single(img)
            
            if not os.path.exists(PATH):
                os.makedirs(PATH)
            np.save(PATH + '/Distortions of image {0}.npy'.format(i), dists)
            for j in range(len(X_adv)):
                io.imsave(PATH + '/Best example of {1} CONST {0}.png'.format(consts, i+j), X_adv[j])
                print('====== save the result:', path+'/Best example of {1} CONST {0}.png'.format(consts, i+j), '======')
            r1.extend(X_adv)
            r2.extend(dists)

        return np.array(r1), np.array(r2)

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    ORACLE = YOLO(0.6, 0.5)  # The auguments do not matter.

    X_test = []
    i=0
    for (root, dirs, files) in os.walk('../COCO/val2017/'):
        if files:
            for f in files:
                # select 10 images
                if i >= EXAMPLE_NUM:
                    break
                print(f)
                path = os.path.join(root, f)
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
                image = process_image(image)
                X_test.append(image)
                i=i+1
    X_test = np.concatenate(X_test, axis=0)
    attacker = YoloAttacker(sess, ORACLE)
    attacker.attack(X_test)