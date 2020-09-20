from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()

_C.MODEL.WEIGHTS = ""

_C.MODEL.BACKBONE = "resnet50"

_C.MODEL.BACKBONE_WEIGHTS = ""

_C.MODEL.NORM_LAYER = "BN"    # "BN", "SyncBN", "FrozenBN", "GN"  in models/layers/batch_norm.py

_C.MODEL.POOLING_LAYER = "avg"   # models/layers/pooling.py

_C.MODEL.ACTIVATE = "relu"   # mish, "swish" ...  in models/layers/create_act.py

_C.MODEL.CLASSES = 1000

_C.MODEL.OPT = "train"  # "train", "infer", "tmp", "jit", "onnx"
_C.MODEL.TYPE = "cls"    # "cls", "metric"
_C.MODEL.STRICT = _C.MODEL.OPT != "train"   # model.load_state_dict

_C.MODEL.METRIC = CN()
_C.MODEL.METRIC.DIM = 256
_C.MODEL.METRIC.POOLING_LAYER = 'gem'
_C.MODEL.METRIC.ACTIVATE = 'relu'
_C.MODEL.METRIC.NORM_LAYER = _C.MODEL.NORM_LAYER

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.OPTIMER = "sgd"   # "sdg", "adam"

_C.SOLVER.START_ITER = 0
_C.SOLVER.MAX_ITER = 1000   # dataloader 循环完训练集的次数

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"   # solver/lr_scheduler.py
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (100,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 5
_C.SOLVER.WARMUP_ITERS = 5
_C.SOLVER.WARMUP_METHOD = "linear"


# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.DATASETS = ""

_C.TRAIN.INPUT_WIDTH = 224
_C.TRAIN.INPUT_HEIGHT = 224
_C.TRAIN.INPUT_CHANNEL = 3
_C.TRAIN.INPUT_BATCH = 64          #dataloader的batch_size, 当使用分布式训练的时候即为每个卡上的batch_size

_C.TRAIN.NUM_WORKERS = 4

_C.TRAIN.TRANSFORM = "DefaultTransform"  # in datasets/transform.py

_C.TRAIN.METRIC = CN()
_C.TRAIN.METRIC.NUM_CLASSES = 6    #选每个训练batch所选择的类别个数
_C.TRAIN.METRIC.NUM_SAMPLES = 6    #每个训练类别的样本个数，所以当次的batchsize = NUM_CLASSES × NUM_SAMPLES
_C.TRAIN.METRIC.OUTER_HARDSAMPLER = True      #是否启用类间困难样本筛选，原理是验证期间对所有训练集制作模板并类间交叉验证相似度，对每一个类别选择相似度最高的INSTANCES_NUM个实例组成一组，训练时随机选择这些困难实例组
_C.TRAIN.METRIC.INTER_HARDSAMPLER = True      #是否启用类内困难样本筛选，原理是验证期间对所有训练集制作模板并类内交叉验证相似度，排除相似度高于阈值的同类实例
_C.TRAIN.METRIC.INTER_HARDSAMPLER_THRED = 0.99  #类内相似度阈值，高于该相似度的被排除

_C.TRAIN.APEX = False

# ---------------------------------------------------------------------------- #
# Loss
# ---------------------------------------------------------------------------- #
_C.LOSS = CN()        #分类支持多种loss，度量支持 triplet loss、ms loss、circle losss
_C.LOSS.NAME = ""       #可不设，分类任务默认使用celoss，度量任务默认使用circle loss


# ---------------------------------------------------------------------------- #
# Inference
# ---------------------------------------------------------------------------- #
_C.INFERENCE = CN()
_C.INFERENCE.TRANSFORM = "InferTransform"
_C.INFERENCE.BATCH_SIZE = 64


# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.NAME = ""           #指定测试方法，不设则用默认方法测试
_C.TEST.DATAPATH = ""       #指定测试集路径
_C.TEST.OUTPATH = ""        #指定测试集输出路径，指定该路径后，会将测试集按模型结果输出到该路径下
_C.TEST.METRIC = CN()
_C.TEST.METRIC.TEMPLATE_PATH = ""


# ---------------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.DIR = "."     #结果输出路径，包括权重和log文件
_C.OUTPUT.SAVE_SUFFIX = ""   #输出权重命名后缀
_C.OUTPUT.SAVE_INTERVAL = 10  #保存间隔，单位为epoch

_C.GPUS = CN()
_C.GPUS.CUDA_VISIBLE_DEVICES = '0'