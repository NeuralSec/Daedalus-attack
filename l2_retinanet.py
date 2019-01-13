import os
import sys
# supress tensorflow logging other than errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '..')

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

from keras_retinanet import models
import cv2
import matplotlib.pyplot as plt
from skimage import io
import time

# Parameter settings:
GPU_ID = 0							# which gpu to used
ATTACK_MODE = 'all'					# select attack mode from 'all', 'most', 'least' and 'single';
ATTACK_CLASS = None					# select the class to attack in 'single' mode
CONFIDENCE = 0.3					# the confidence of attack
EXAMPLE_NUM = 10					# total number of adversarial example to generate.
BATCH_SIZE = 1						# number of adversarial example generated in each batch

BINARY_SEARCH_STEPS = 5     		# number of times to adjust the constsant with binary search
INITIAL_consts = 50        			# the initial constsant c to pick as a first guess
CLASS_NUM = 80						# 80 for COCO dataset
MAX_ITERATIONS = 10000      		# number of iterations to perform gradient descent
ABORT_EARLY = True          		# if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2        		# larger values converge faster to less accurate results
IMAGE_SHAPE = (416, 416, 3)         # input image shape
SAVE_PATH = 'adv_examples/L2/f3/{0}/'.format(ATTACK_MODE)
# select GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(GPU_ID)


def process_image(img):
	"""
	Resize, reduce and expand image.
	# Argument:
		img: original image.

	# Returns
		image: ndarray(64, 64, 3), processed image.
	"""
	image = cv2.resize(img, (416, 416),
					   interpolation=cv2.INTER_CUBIC)
	image = np.array(image, dtype='float32')
	image /= 255.
	image = np.expand_dims(image, axis=0)
	return image

class Daedalus:
	"""
	Daedalus adversarial example generator based on the Yolo v3 model.
	"""
	def __init__(self, sess, model, target_class=ATTACK_CLASS, attack_mode=ATTACK_MODE, img_shape=IMAGE_SHAPE,
				 batch_size=BATCH_SIZE, confidence=CONFIDENCE, learning_rate=LEARNING_RATE, binary_search_steps=BINARY_SEARCH_STEPS,
				 max_iterations=MAX_ITERATIONS, abort_early=ABORT_EARLY, initial_consts=INITIAL_consts, boxmin=0, boxmax=1):

		# self.sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		self.sess = sess
		self.LEARNING_RATE = learning_rate
		self.MAX_ITERATIONS = max_iterations
		self.BINARY_SEARCH_STEPS = binary_search_steps
		self.ABORT_EARLY = abort_early
		self.initial_consts = initial_consts
		self.batch_size = batch_size
		self.repeat = binary_search_steps >= 6
		self.detection_model = model
		self.confidence = confidence
		self.img_dimension = img_shape[0]
		self.target_class = target_class
		self.attack_mode = attack_mode

		def select_class(target_class, boxes, objectness, box_scores, mode='all'):
			box_classes = tf.cast(tf.argmax(box_scores, axis=-1), tf.int32, name='box_classes')
			class_counts = tf.bincount(box_classes)
			print(class_counts)
			if mode == 'all':
				selected_boxes = tf.reshape(boxes, [BATCH_SIZE, -1, 4])
				selected_scores = tf.reshape(box_scores, [BATCH_SIZE, -1, CLASS_NUM])
				if objectness == None:
					return selected_boxes, None, selected_scores
				selected_objectness = tf.reshape(objectness, [BATCH_SIZE, -1, 1])
				return selected_boxes, selected_objectness, selected_scores
			elif mode == 'most':
				selected_cls = tf.argmax(class_counts)
			elif mode == 'least':
				class_counts = tf.where(tf.equal(class_counts,0), int(1e6)*tf.ones_like(class_counts, dtype=tf.int32), class_counts)
				selected_cls = tf.argmin(class_counts)
			elif mode == 'single':
				file = 'data/coco_classes.txt'
				with open(file) as f:
					class_names = f.readlines()
				class_names = [c.strip() for c in class_names]
				selected_cls = class_names.index(target_class)
			selected_cls = tf.cast(selected_cls, tf.int32)  
			index = tf.equal(box_classes, selected_cls)
			index = tf.cast(index, tf.int32)
			_, selected_boxes = tf.dynamic_partition(boxes, index, num_partitions=2, name='dynamic_partition')
			_, selected_scores = tf.dynamic_partition(box_scores, index, num_partitions=2, name='dynamic_partition')
			selected_boxes = tf.reshape(selected_boxes, [BATCH_SIZE, -1, 4])
			selected_scores = tf.reshape(selected_scores, [BATCH_SIZE, -1, CLASS_NUM])
			if objectness == None:
				return selected_boxes, None, selected_scores
			_, selected_objectness = tf.dynamic_partition(objectness, index, num_partitions=2, name='dynamic_partition')
			selected_objectness = tf.reshape(selected_objectness, [BATCH_SIZE, -1, 1])
			return selected_boxes, selected_objectness, selected_scores

		# the perturbation we're going to optimize:
		perturbations = tf.Variable(np.zeros((batch_size,
											  img_shape[0],
											  img_shape[1],
											  img_shape[2])), dtype=tf.float32, name='perturbations')
		# tf variables to sending data to tf:
		self.timgs = tf.Variable(np.zeros((batch_size,
										   img_shape[0],
										   img_shape[1],
										   img_shape[2])), dtype=tf.float32, name='self.timgs')
		self.consts = tf.Variable(np.zeros(batch_size), dtype=tf.float32, name='self.consts')

		# and here's what we use to assign them:
		self.assign_timgs = tf.placeholder(tf.float32, (batch_size,
														img_shape[0],
														img_shape[1],
														img_shape[2]))
		self.assign_consts = tf.placeholder(tf.float32, [batch_size])

		# Tensor operation: the resulting image, tanh'd to keep bounded from
		# boxmin to boxmax:
		self.boxmul = (boxmax - boxmin) / 2.
		self.boxplus = (boxmin + boxmax) / 2.
		self.newimgs = tf.tanh(perturbations + self.timgs) * self.boxmul + self.boxplus

		caffe_imgs = self.newimgs * 255.
		caffe_imgs = caffe_imgs[..., ::-1]
		caffe_offsets = np.concatenate([103.939*np.ones((batch_size, 416, 416, 1)),
								 	    116.779*np.ones((batch_size, 416, 416, 1)),
								        123.68*np.ones((batch_size, 416, 416, 1))], axis=-1)
		caffe_imgs = caffe_imgs - caffe_offsets
		
		# Get prediction from the model:
		boxes, classprobs = self.detection_model(caffe_imgs)
		boxes, _, classprobs = select_class(self.target_class, boxes, None, classprobs, mode=self.attack_mode)
		print(boxes, classprobs)
		self.x1 = boxes[..., 0:1]/self.img_dimension
		self.y1 = boxes[..., 1:2]/self.img_dimension
		self.x2 = boxes[..., 2:3]/self.img_dimension
		self.y2 = boxes[..., 3:4]/self.img_dimension
		self.bw = tf.math.abs(self.x2 - self.x1)
		self.bh = tf.math.abs(self.y1 - self.y2)
		self.class_probs = classprobs
		self.box_scores = tf.reduce_max(self.class_probs, axis=-1, keepdims=True)

		# Optimisation metrics:
		self.l2dist = tf.reduce_sum(tf.square(self.newimgs - (tf.tanh(self.timgs) * self.boxmul + self.boxplus)), [1, 2, 3])

		# Define DDoS losses: loss must be a tensor here!
		# Make the box confidence of all detections to be 1.
		self.loss1_1_x = tf.reduce_mean(tf.square(self.box_scores - 1), [-2, -1])

		# Minimising the size of all bounding box.
		self.f3 = 1e1 * tf.reduce_mean(tf.square(tf.multiply(self.bw, self.bh)), [-2, -1])

		# add two loss terms together
		self.loss_adv = self.loss1_1_x + self.f3
		self.loss1 = tf.reduce_mean(self.consts * self.loss_adv)
		self.loss2 = tf.reduce_mean(self.l2dist)
		self.loss = self.loss1 + self.loss2

		# Setup the adam optimizer and keep track of variables we're creating
		start_vars = set(x.name for x in tf.global_variables())
		optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
		self.train = optimizer.minimize(self.loss, var_list=[perturbations])
		end_vars = tf.global_variables()
		new_vars = [x for x in end_vars if x.name not in start_vars]

		# these are the variables to initialize when we run
		self.setup = []
		self.setup.append(self.timgs.assign(self.assign_timgs))
		self.setup.append(self.consts.assign(self.assign_consts))
		self.init = tf.variables_initializer(var_list=[perturbations] + new_vars)

	def attack_batch(self, imgs):
		"""
		Run the attack on a batch of images and labels.
		"""

		def check_success(loss, init_loss):
			"""
			Check if the initial loss value has been reduced by 'self.confidence' percent
			"""
			return loss <= init_loss * (1 - self.confidence)

		batch_size = self.batch_size

		# convert images to arctanh-space
		imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)

		# set the lower and upper bounds of the constsant.
		lower_bound = np.zeros(batch_size)
		consts = np.ones(batch_size) * self.initial_consts
		upper_bound = np.ones(batch_size) * 1e10

		# store the best l2, score, and image attack
		o_bestl2 = [1e10] * batch_size
		o_bestloss = [1e10] * batch_size
		o_bestattack = [np.zeros(imgs[0].shape)] * batch_size

		for outer_step in range(self.BINARY_SEARCH_STEPS):
			# completely reset adam's internal state.
			self.sess.run(self.init)

			# take in the current data batch.
			batch = imgs[:batch_size]

			# cache the current best l2 and score.
			bestl2 = [1e10] * batch_size
			# bestconfidence = [-1]*batch_size
			bestloss = [1e10] * batch_size

			# The last iteration (if we run many steps) repeat the search once.
			if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
				consts = upper_bound

			# set the variables so that we don't have to send them over again.
			self.sess.run(self.setup, {self.assign_timgs: batch,
									   self.assign_consts: consts})

			# start gradient descent attack
			print('adjust c to:', sess.run(self.consts))
			init_loss = sess.run(self.loss)
			init_adv_losses = sess.run(self.loss_adv)
			prev = init_loss * 1.1
			for iteration in range(self.MAX_ITERATIONS):
				# perform the attack on a single example
				_, l, l2s, l1s, nimgs, c = self.sess.run([self.train, self.loss, self.l2dist, self.loss_adv, self.newimgs, self.consts])
				# print out the losses every 10%
				if iteration % (self.MAX_ITERATIONS // 10) == 0:
					print('===iteration:', iteration, '===')
					print('attacked box number:', sess.run(self.bw).shape)
					print('loss values of box confidence and dimension:', sess.run([self.loss1_1_x, self.f3]))
					print('adversarial losses:', l1s)
					print('distortions:', l2s)
					path = SAVE_PATH+'retinanet/{0} confidence'.format(self.confidence)
					if not os.path.exists(path):
						os.makedirs(path)
					#[io.imsave(path+'/debug_img_{0}Iteration_{1}.png'.format(i, iteration), nimgs[i]) for i in range(nimgs.shape[0])]

				# check if we should abort search if we're getting nowhere.
				if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS // 10) == 0:
					if l > prev * .9999:
						break
					prev = l

				# update the best result found so far
				for e, (l1, l2, ii) in enumerate(zip(l1s, l2s, nimgs)):
					if l2 < bestl2[e] and check_success(l1, init_adv_losses[e]):
						bestl2[e] = l2
						bestloss[e] = l1
					if l2 < o_bestl2[e] and check_success(l1, init_adv_losses[e]):
						o_bestl2[e] = l2
						o_bestloss[e] = l1
						o_bestattack[e] = ii

			# adjust the constsant as needed
			for e in range(batch_size):
				if check_success(l1s[e], init_adv_losses[e]):
					# success, divide consts by two
					upper_bound[e] = min(upper_bound[e], consts[e])
					if upper_bound[e] < 1e9:
						consts[e] = (lower_bound[e] + upper_bound[e]) / 2
				else:
					# failure, either multiply by 10 if no solution found yet
					#          or do binary search with the known upper bound
					lower_bound[e] = max(lower_bound[e], consts[e])
					if upper_bound[e] < 1e9:
						consts[e] = (lower_bound[e] + upper_bound[e]) / 2
					else:
						consts[e] *= 10
		# return the best solution found
		o_bestl2 = np.array(o_bestl2)
		return o_bestattack, o_bestl2


	def attack(self, imgs):
		"""
		Perform the L_2 attack on the given images for the given targets.
		If self.targeted is true, then the targets represents the target labels.
		If self.targeted is false, then targets are the original class labels.
		"""
		r = []
		ds = []
		print('go up to', len(imgs))
		for i in range(0, len(imgs), self.batch_size):
			print('tick', i)
			X_adv, dists = self.attack_batch(imgs[i:i + self.batch_size])
			path = SAVE_PATH+'retinanet/{0} confidence'.format(self.confidence)
			if not os.path.exists(path):
				os.makedirs(path)
			np.save(path+'/Distortions of images {0} to {1}.npy'.format(i, i+self.batch_size), dists)
			for j in range(len(X_adv)):
				io.imsave(path+'/Best example of {1} Distortion {2}.png'.format(self.confidence, i+j, dists[j]), X_adv[j])			
			r.extend(X_adv)
			ds.extend(dists)
		return np.array(r), np.array(ds)


if __name__ == '__main__':
	sess = tf.InteractiveSession()
	init = tf.global_variables_initializer()
	sess.run(init)
	# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
	model_path = os.path.join('../RetinaNet/snapshots', 'resnet50_coco_best_v2.1.0.h5')
	# load retinanet model
	ORACLE = models.load_model(model_path, backbone_name='resnet50')
	ORACLE.layers.pop()
	ORACLE.outputs = [ORACLE.layers[-2].output, ORACLE.layers[-1].output] #remove nms from original model
	ORACLE.layers[-1].outbound_nodes = []
	ORACLE.summary()
	X_test = []
	for (root, dirs, files) in os.walk('../Datasets/COCO/val2017'):
		if files:
			for f in files:
				print(f)
				path = os.path.join(root, f)
				image = cv2.imread(path)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
				image = process_image(image)
				X_test.append(image)
				EXAMPLE_NUM -= 1
				if EXAMPLE_NUM == 0:
					break
	X_test = np.concatenate(X_test, axis=0)
	attacker = Daedalus(sess, ORACLE)
	X_adv, distortions = attacker.attack(X_test)
	np.savez(SAVE_PATH+'retinanet/{0} confidence/Daedalus example batch.npz'.format(CONFIDENCE), X_adv=X_adv, distortions=distortions)
	writer = tf.summary.FileWriter("log", sess.graph)
	writer.close()