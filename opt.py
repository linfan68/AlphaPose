import logging
from attr import *

@attrs()
class OptType(object):
  syncbn = attrib()
  dtype = attrib()
  gpu_id = attrib()
  dataset = attrib()
  vis = attrib()
  nClasses = attrib()
  det_model = attrib()
  confidence = attrib()
  loadModel = attrib()
  nEpochs = attrib()
  board = attrib()
  hardMining = attrib()
  lr_decay_epoch = attrib()
  format = attrib()
  load_from_pyt = attrib()
  optMethod = attrib()
  inputResH = attrib()
  use_pretrained_base = attrib()
  visdom = attrib()
  outputpath = attrib()
  rotate = attrib()
  nThreads = attrib()
  pre_resnet = attrib()
  mode = attrib()
  valIters = attrib()
  try_loadModel = attrib()
  trainIters = attrib()
  trainBatch = attrib()
  crit = attrib()
  inp_dim = attrib()
  save_img = attrib()
  nms_thresh = attrib()
  scale = attrib()
  validBatch = attrib()
  epoch = attrib()
  LR = attrib()
  inputpath = attrib()
  lr_decay = attrib()
  map = attrib()
  expID = attrib()
  inputlist = attrib()
  posebatch = attrib()
  hmGauss = attrib()
  logging_file = attrib()
  outputResH = attrib()
  outputResW = attrib()
  eps = attrib()
  snapshot = attrib()
  detbatch = attrib()
  inputResW = attrib()
  addDPG = attrib()

  def __init__(self, *args, **kwargs):
    return 'refer to Note: self-written __init__'

optDict = {
  "optMethod": "rmsprop",
  "addDPG": False,
  "lr_decay_epoch": "20,60",
  "detbatch": 1,
  "rotate": 40,
  "trainIters": 0,
  "trainBatch": 28,
  "crit": "MSE",
  "outputResH": 64,
  "LR": 0.0001,
  "confidence": 0.1,
  "det_model": "frcnn",
  "lr_decay": 0.1,
  "loadModel": None,
  "inp_dim": "608",
  "scale": 0.3,
  "hmGauss": 1,
  "outputpath": "examples/res/",
  "eps": 1e-08,
  "save_img": False,
  "nms_thresh": 0.6,
  "board": True,
  "mode": "normal",
  "dtype": "float32",
  "nThreads": 60,
  "nClasses": 33,
  "epoch": 0,
  "expID": "default",
  "inputResH": 256,
  "map": True,
  "dataset": "coco",
  "gpu_id": [
    0
  ],
  "nEpochs": 100,
  "hardMining": False,
  "pre_resnet": True,
  "validBatch": 24,
  "valIters": 0,
  "inputpath": "",
  "logging_file": "training.log",
  "posebatch": 80,
  "outputResW": 48,
  "inputResW": 192,
  "format": None,
  "use_pretrained_base": True,
  "load_from_pyt": False,
  "visdom": False,
  "try_loadModel": None,
  "inputlist": "",
  "snapshot": 1,
  "vis": False,
  "syncbn": False
}

opt = OptType(**optDict)
logger = logging.getLogger('')