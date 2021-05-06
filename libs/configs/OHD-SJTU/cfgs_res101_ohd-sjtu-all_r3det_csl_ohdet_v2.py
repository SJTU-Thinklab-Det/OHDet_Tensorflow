# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
OBB task:
iou threshold: 0.5
classname: small-vehicle
npos num: 5090
ap:  0.5177070294041706
classname: ship
npos num: 9886
ap:  0.8469567403220999
classname: plane
npos num: 2673
ap:  0.8940771079277483
classname: large-vehicle
npos num: 4293
ap:  0.7486987085711382
classname: helicopter
npos num: 72
ap:  0.43546105310811195
classname: harbor
npos num: 2065
ap:  0.5665761843231875
map: 0.6682461372760761
classaps:  [51.77070294 84.69567403 89.40771079 74.86987086 43.54610531 56.65761843]

AP50:95: [0.6682461372760761, 0.6509849247505275, 0.6045151876618581, 0.548188764744899, 0.4926493600493705, 0.37650443820961277, 0.24201622227210695, 0.12280794762532815, 0.03483493252183537, 0.0060885363236986975]
mmAP: 0.37468364514353136

OHD task:
iou threshold: 0.5
classname: small-vehicle
npos num: 5090
ap:0.2483484808750491, ha:0.5901956035010973
classname: ship
npos num: 9886
ap:0.45895037020083196, ha:0.6862242164302402
classname: plane
npos num: 2673
ap:0.529545493817519, ha:0.7091836706860923
classname: large-vehicle
npos num: 4293
ap:0.35230390696695973, ha:0.5844889558693144
classname: helicopter
npos num: 72
ap:0.2539875134751795, ha:0.6326529321116464
classname: harbor
npos num: 2065
ap:0.4073936878322649, ha:0.7685950360289598
map:0.37508824219463405, mha:0.6618900691045584
classaps:[24.83484809 45.89503702 52.95454938 35.2303907  25.39875135 40.73936878], classhas:[59.01956035 68.62242164 70.91836707 58.44889559 63.26529321 76.8595036 ]

AP50:95: [0.37508824219463405, 0.36690728568624165, 0.3490816140248984, 0.3219308243534246, 0.2940512562468297, 0.2280593229982315, 0.15636861277330996, 0.09357037429330711, 0.02753142371910701, 0.005434101244224074]
mmAP: 0.22180230575342086
HA50:95: [0.6618900691045584, 0.6652963870973309, 0.6682604913085717, 0.6789068770321355, 0.6895390354046049, 0.6881260821227814, 0.6911164847308799, 0.7323274020476842, 0.7691582342458855, 0.4271461741894664]
mmHA: 0.6671767237283898
"""

# ------------------------------------------------
VERSION = 'RetinaNet_OHD-SJTU-ALL_R3Det_CSL_OHDet_2x_20200818'
NET_NAME = 'resnet101_v1d'  # 'MobilenetV2'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,2,3"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 20
SMRY_ITER = 200
SAVE_WEIGHTS_INTE = 20000 * 2

SUMMARY_PATH = ROOT_PATH + '/output/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'

if NET_NAME.startswith("resnet"):
    weights_name = NET_NAME
elif NET_NAME.startswith("MobilenetV2"):
    weights_name = "mobilenet/mobilenet_v2_1.0_224"
else:
    raise Exception('net name must in [resnet_v1_101, resnet_v1_50, MobilenetV2]')

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
EVALUATE_DIR = ROOT_PATH + '/output/evaluate_result_pickle/'

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False
FIXED_BLOCKS = 1  # allow 0~3
FREEZE_BLOCKS = [True, False, False, False, False]  # for gluoncv backbone
USE_07_METRIC = True

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip

CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
ANGLE_CLS_WEIGHT = 0.5
HEAD_CLS_WEIGHT = 0.1
USE_IOU_FACTOR = True

BATCH_SIZE = 1
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 5e-4 * BATCH_SIZE * NUM_GPU
DECAY_STEP = [SAVE_WEIGHTS_INTE*12, SAVE_WEIGHTS_INTE*16, SAVE_WEIGHTS_INTE*20]
MAX_ITERATION = SAVE_WEIGHTS_INTE*20
WARM_SETP = int(1.0 / 8.0 * SAVE_WEIGHTS_INTE)


# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'OHD-SJTU-ALL-HEAD-600'  # 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 800
CLASS_NUM = 6
LABEL_TYPE = 0
RADUIUS = 4
OMEGA = 1

IMG_ROTATE = False
RGB2GRAY = False
VERTICAL_FLIP = False
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = False

# --------------------------------------------- Network_config
SUBNETS_WEIGHTS_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
SUBNETS_BIAS_INITIALIZER = tf.constant_initializer(value=0.0)
PROBABILITY = 0.01
FINAL_CONV_BIAS_INITIALIZER = tf.constant_initializer(value=-math.log((1.0 - PROBABILITY) / PROBABILITY))
WEIGHT_DECAY = 1e-4
USE_GN = False
NUM_SUBNET_CONV = 4
NUM_REFINE_STAGE = 1
USE_RELU = False
FPN_CHANNEL = 256

# ---------------------------------------------Anchor config
LEVEL = ['P3', 'P4', 'P5', 'P6', 'P7']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE = [8, 16, 32, 64, 128]
ANCHOR_SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
ANCHOR_RATIOS = [1, 1 / 2, 2., 1 / 3., 3., 5., 1 / 5.]
ANCHOR_ANGLES = [-90, -75, -60, -45, -30, -15]
ANCHOR_SCALE_FACTORS = None
USE_CENTER_OFFSET = True
METHOD = 'H'
USE_ANGLE_COND = False
ANGLE_RANGE = 90

# --------------------------------------------RPN config
SHARE_NET = True
USE_P5 = True
IOU_POSITIVE_THRESHOLD = 0.5
IOU_NEGATIVE_THRESHOLD = 0.4
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

NMS = True
NMS_IOU_THRESHOLD = 0.1
MAXIMUM_DETECTIONS = 100
FILTERED_SCORE = 0.05
VIS_SCORE = 0.4

# --------------------------------------------MASK config
USE_SUPERVISED_MASK = False
MASK_TYPE = 'r'  # r or h
BINARY_MASK = False
SIGMOID_ON_DOT = False
MASK_ACT_FET = True  # weather use mask generate 256 channels to dot feat.
GENERATE_MASK_LIST = ["P3", "P4", "P5", "P6", "P7"]
ADDITION_LAYERS = [4, 4, 3, 2, 2]  # add 4 layer to generate P2_mask, 2 layer to generate P3_mask
ENLAEGE_RF_LIST = ["P3", "P4", "P5", "P6", "P7"]
SUPERVISED_MASK_LOSS_WEIGHT = 1.0
