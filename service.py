import os
import cv2
import time
import numpy as np
import gluoncv as gcv
import mxnet as mx
import os.path as osp

from opt import opt
from pose_utils import pose_nms
from sppe.models.sefastpose import FastPose_SE
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


# overwrite
opt.nClasses = 33

# person detection will run inference at inp_dim * inp_dim
opt.inp_dim = int(os.environ.get('INP_DIM', '192'))
use_cpu = int(os.environ.get('USE_CPU', '0'))

# single pose estimation will run inference at inputResH * inputResW
# tested: size smaller than 256 * 192 will cause precision loss
opt.inputResH = int(os.environ.get('INPUT_RES_H', '256'))
opt.inputResW = int(os.environ.get('INPUT_RES_W', '192'))
opt.outputResH = opt.inputResH // 4
opt.outputResW = opt.inputResW // 4

print('current options: ', opt)

if not use_cpu:
  ctx = mx.gpu()
else:
  ctx = mx.cpu()

def get_detect_net():
  # model config
  # print('Loading yolo3_darknet53_coco ...')
  net = gcv.model_zoo.get_model('yolo3_darknet53_coco', pretrained=True, root='models')
  net.set_nms(opt.nms_thresh, post_nms=-1)
  person_idx = net.classes.index('person')

  # print('Modifying output layers to ignore non-person classes...')
  net._clear_cached_op()
  len_per_anchor = 5 + len(net.classes)

  for output in net.yolo_outputs:
    num_anchors = output._num_anchors
    picked_channels = np.array(list(range(len_per_anchor)) * num_anchors)
    picked_channels = np.where((picked_channels < 5) |
                                (picked_channels == 5 + person_idx))

    parameters = output.prediction.params
    for k in parameters:
      if 'weight' in k:
        key_weight = k
        init_weight = parameters[k].data()[picked_channels]
        in_channels = parameters[k].data().shape[1]
      elif 'bias' in k:
        key_bias = k
        init_bias = parameters[k].data()[picked_channels]

    output.prediction = mx.gluon.nn.Conv2D(6 * num_anchors,
                                            in_channels=in_channels,
                                            kernel_size=1,
                                            padding=0,
                                            strides=1,
                                            prefix=output.prediction.prefix)

    output.prediction.collect_params().initialize()
    output.prediction.params[key_weight].set_data(init_weight)
    output.prediction.params[key_bias].set_data(init_bias)
    output._classes = 1
    output._num_pred = 6

  net._classes = ['person']

  net.collect_params().reset_ctx(ctx)
  net.hybridize()

  return net, person_idx

def get_estimate_net():

  # model config
  # print('Loading SPPE ...')
  net = FastPose_SE(ctx)
  net.load_parameters('models/duc_se.params')
  net.hybridize()
  net.collect_params().reset_ctx(ctx)
  return net

def get_batched_image_from_name(img_name):

  def load_image(img_name, tensor_size):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    cvImg = cv2.imread(img_name)
    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)

    img = mx.nd.array(cvImg)
    img_size = (img.shape[1], img.shape[0])

    # resize image
    tensor = gcv.data.transforms.image.resize_long(img, tensor_size, interp=9)
    tensor = mx.nd.image.to_tensor(tensor)
    tensor = mx.nd.image.normalize(tensor, mean=mean, std=std)
    tensor = tensor.expand_dims(0)

    # pad tensor
    pad_h = tensor_size - tensor.shape[2]
    pad_w = tensor_size - tensor.shape[3]
    pad_shape = (0, 0, 0, 0, 0, pad_h, 0, pad_w)
    tensor = mx.nd.pad(tensor, mode='constant',
                        constant_value=0.5, pad_width=pad_shape)

    return tensor, img, img_size

  tensor_size = int(opt.inp_dim)

  tensor_k, img_k, img_size_k = load_image(img_name, tensor_size)

  tensor_batch = [tensor_k]
  img_size_batch = [img_size_k]
  img_batch = [img_k]
  img_name_list = [img_name]

  tensor_batch = mx.nd.concatenate(tensor_batch, axis=0)
  img_size_batch = mx.nd.array(img_size_batch, dtype='float32')
  img_size_batch = img_size_batch.tile(reps=[1, 2])

  return (tensor_batch, img_batch, img_size_batch, img_name_list)

def get_batched_image(image):

  def load_image(image, tensor_size):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # cvImg = cv2.imread(img_name)
    cvImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img = mx.nd.array(cvImg)
    img_size = (img.shape[1], img.shape[0])

    # resize image
    tensor = gcv.data.transforms.image.resize_long(img, tensor_size, interp=9)
    tensor = mx.nd.image.to_tensor(tensor)
    tensor = mx.nd.image.normalize(tensor, mean=mean, std=std)
    tensor = tensor.expand_dims(0)

    # pad tensor
    pad_h = tensor_size - tensor.shape[2]
    pad_w = tensor_size - tensor.shape[3]
    pad_shape = (0, 0, 0, 0, 0, pad_h, 0, pad_w)
    tensor = mx.nd.pad(tensor, mode='constant',
                        constant_value=0.5, pad_width=pad_shape)

    return tensor, img, img_size

  tensor_size = int(opt.inp_dim)

  tensor_k, img_k, img_size_k = load_image(image, tensor_size)

  tensor_batch = [tensor_k]
  img_size_batch = [img_size_k]
  img_batch = [img_k]
  img_name_list = ['placeholder']

  tensor_batch = mx.nd.concatenate(tensor_batch, axis=0)
  img_size_batch = mx.nd.array(img_size_batch, dtype='float32')

  img_size_batch = img_size_batch.tile(reps=[1, 2])


  return (tensor_batch, img_batch, img_size_batch, img_name_list)

def detect(net, person_idx, batched_image):
  tensor_batch, img_batch, img_size_batch, img_name_list = batched_image

  class_idxs, scores, boxes = net(tensor_batch.copyto(ctx))
  class_idxs = class_idxs.copyto(mx.cpu())
  scores = scores.copyto(mx.cpu())
  boxes = boxes.copyto(mx.cpu())

  input_size = int(opt.inp_dim)
  q = []

  for i in range(scores.shape[0]):
    img, boxes, class_idxs, scores, img_name, img_size = img_batch[i], boxes[i], class_idxs[i, :, 0], scores[i, :, 0], img_name_list[i], img_size_batch[i]

    # rescale coordinates
    scaling_factor = mx.nd.min(input_size / img_size)
    boxes /= scaling_factor

    # cilp coordinates
    boxes[:, [0, 2]] = mx.nd.clip(boxes[:, [0, 2]], 0., img_size[0].asscalar() - 1)
    boxes[:, [1, 3]] = mx.nd.clip(boxes[:, [1, 3]], 0., img_size[1].asscalar() - 1)

    # select boxes
    mask1 = (class_idxs == person_idx).asnumpy()
    mask2 = (scores > opt.confidence).asnumpy()
    picked_idxs = np.where((mask1 + mask2) > 1)[0]

    if picked_idxs.shape[0] == 0:
      q.append((img, None, None, img_name))
    else:
      print('PPL detected: ', boxes[picked_idxs], scores[picked_idxs], img_name)
      q.append((img, boxes[picked_idxs], scores[picked_idxs], img_name))

  return q

def crop(detected_queue):
  def crop_fn(img, boxes):
    '''
    Crop persons based on given boxes
    INPUT:
        img: mx.nd, original image
        boxes: mx.nd, image size after resize
    OUTPUT:
        tensors: mx.nd, input tensor for pose estimation
        pt1: mx.nd, coordinates of left upper box corners
        pt2: mx.nd, coordinates of right bottom box corners
    '''
    mean = (0.485, 0.456, 0.406)
    std = (1.0, 1.0, 1.0)
    img_width, img_height = img.shape[1], img.shape[0]

    tensors = mx.nd.zeros([boxes.shape[0], 3, opt.inputResH, opt.inputResW])
    pt1 = mx.nd.zeros([boxes.shape[0], 2])
    pt2 = mx.nd.zeros([boxes.shape[0], 2])

    img = mx.nd.image.to_tensor(img)
    img = mx.nd.image.normalize(img, mean=mean, std=std)
    img = img.transpose(axes=[1, 2, 0])

    for i, box in enumerate(boxes.asnumpy()):
      box_width = box[2] - box[0]
      box_height = box[3] - box[1]
      if box_width > 100:
        scale_rate = 0.2
      else:
        scale_rate = 0.3

      # crop image
      left = int(max(0, box[0] - box_width * scale_rate / 2))
      up = int(max(0, box[1] - box_height * scale_rate / 2))
      right = int(min(img_width - 1,
                      max(left + 5, box[2] + box_width * scale_rate / 2)))
      bottom = int(min(img_height - 1,
                        max(up + 5, box[3] + box_height * scale_rate / 2)))
      crop_width = right - left
      crop_height = bottom - up
      cropped_img = mx.image.fixed_crop(img, left, up, crop_width, crop_height)

      # resize image
      resize_factor = min(opt.inputResW / crop_width, opt.inputResH / crop_height)
      new_width = int(crop_width * resize_factor)
      new_height = int(crop_height * resize_factor)
      tensor = mx.image.imresize(cropped_img, new_width, new_height)
      tensor = tensor.transpose(axes=[2, 0, 1])
      tensor = tensor.reshape(1, 3, new_height, new_width)

      # pad tensor
      pad_h = opt.inputResH - new_height
      pad_w = opt.inputResW - new_width
      pad_shape = (0, 0, 0, 0, pad_h // 2, (pad_h + 1) // 2, pad_w // 2, (pad_w + 1) // 2)
      tensor = mx.nd.pad(tensor, mode='constant',
                          constant_value=0.5, pad_width=pad_shape)
      tensors[i] = tensor.reshape(3, opt.inputResH, opt.inputResW)
      pt1[i] = (left, up)
      pt2[i] = (right, bottom)

    return tensors, pt1, pt2

  q = []

  for item in detected_queue:
    img, boxes, scores, img_name = item

    if boxes is None:
      q.append((None, img, None, None, None, None, img_name))
      continue

    tensors, pt1, pt2 = crop_fn(img, boxes)

    q.append((tensors, img, boxes, scores, pt1, pt2, img_name))

  return q


def estimate(enet, cropped_queue):
  def transform_fn(hms, pt1, pt2, inp_h, inp_w, res_h, res_w):
    '''
    Transform pose heatmaps to coordinates
    INPUT:
        hms: mx.nd, pose heatmaps
        pt1: mx.nd, coordinates of left upper box corners
        pt2: mx.nd, coordinates of right bottom box corners
        inp_h: int, input tensor height
        inp_w: int, input tensot width
        res_h: int, output heatmap height
        res_w: int, output heatmap width
    OUTPUT:
        preds: mx.nd, pose coordinates in box frames
        preds_tf: mx.nd, pose coordinates in image frames
        maxval: mx.nd, pose scores
    '''
    pt1 = pt1.expand_dims(axis=1)
    pt2 = pt2.expand_dims(axis=1)

    # get keypoint coordinates
    idxs = mx.nd.argmax(hms.reshape(hms.shape[0], hms.shape[1], -1), 2, keepdims=True)
    maxval = mx.nd.max(hms.reshape(hms.shape[0], hms.shape[1], -1), 2, keepdims=True)
    preds = idxs.tile(reps=[1, 1, 2])
    preds[:, :, 0] %= hms.shape[3]
    preds[:, :, 1] /= hms.shape[3]

    # get pred masks
    pred_mask = (maxval > 0).tile(reps=[1, 1, 2])
    preds *= pred_mask

    # coordinate transformation
    box_size = pt2 - pt1
    len_h = mx.nd.maximum(box_size[:, :, 1:2], box_size[:, :, 0:1] * inp_h / inp_w)
    len_w = len_h * inp_w / inp_h
    canvas_size = mx.nd.concatenate([len_w, len_h], axis=2)
    offsets = pt1 - mx.nd.maximum(0, canvas_size / 2 - box_size / 2)
    preds_tf = preds * len_h / res_h + offsets

    return preds, preds_tf, maxval

  q = []

  for item in cropped_queue:
    tensors, img, boxes, box_scores, pt1, pt2, img_name = item

    if tensors is None:
      q.append((img, None, None, None, None, img_name))
      continue

    heatmaps = []
    num_poses = tensors.shape[0]

    batch_size = 1
    num_batches = (num_poses + batch_size - 1) // batch_size

    for k in range(num_batches):
      # get batch tensor
      begin_idx = k * batch_size
      end_idx = min(begin_idx + batch_size, num_poses)
      tensor_batch = tensors[begin_idx:end_idx]
      # get prediction
      heatmap_batch = enet(tensor_batch.copyto(ctx))
      heatmap_batch = heatmap_batch[:, :17, :, :]
      heatmaps.append(heatmap_batch.copyto(mx.cpu()))

    # coordinate transformation
    heatmaps = mx.nd.concatenate(heatmaps, axis=0)
    pose_hms, pose_coords, pose_scores = transform_fn(heatmaps, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
    q.append((img, boxes, box_scores, pose_coords, pose_scores, img_name))

  print('estimated len(q):', len(q))
  return q

def process(estimated_queue):
  q = []

  for item in estimated_queue:
    img, boxes, box_scores, pose_coords, pose_scores, img_name = item

    if boxes is None:
      q.append((None, None, None, img_name))
      continue

    final_result, boxes, box_scores = pose_nms(boxes.asnumpy(),
                                                box_scores.asnumpy(),
                                                pose_coords.asnumpy(), pose_scores.asnumpy())
    # print(final_result)


    q.append((final_result, boxes, box_scores, img_name))

  return q

def roundFloat(num):
  return round(num * 1000) / 1000

def parse_results(results):
  def area(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]

    return abs(x2-x1) * abs(y2-y1)

  assert len(results) == 1, 'Pose detection results are from more than one image'
  final_result, boxes, box_scores, img_name = results[0]

  ps = []
  if final_result is None:
    return ps

  for i in range(len(final_result)):
    partData = np.concatenate([
      final_result[i]['keypoints'],
      final_result[i]['kp_score']],
      axis=1)

    confidence = roundFloat(final_result[i]['proposal_score'][0] / 3)

    partData = np.reshape(partData, (-1)).tolist()
    partData = list(map(roundFloat, partData))
    box = boxes[i].tolist()
    box = list(map(roundFloat, box))
    boxScore = roundFloat(box_scores[i])

    ps.append({
      'partData': partData,
      'confidence': confidence,
      'box': box,
      'boxScore': boxScore
    })

  ps.sort(key = lambda x:area(x['box']), reverse = True)
  return ps

def inference_test(img_name, net, person_idx, enet):
  form = lambda x: str(round(x * 1000, 2)).rjust(6) + ' ms'
  batched_image = get_batched_image_from_name(img_name)

  t0 = time.time()
  detected_queue = detect(net, person_idx, batched_image)
  t1 = time.time()
  print('\n', ' detect:', form(t1-t0))

  cropped_queue = crop(detected_queue)

  t2 = time.time()
  estimated_queue = estimate(enet, cropped_queue)
  results = process(estimated_queue)
  persons = parse_results(results)
  t3 = time.time()
  print('estimate:', form(t3-t2))

  # print(persons)
  return persons

def inference(img, net, person_idx, enet):
  batched_image = get_batched_image(img)
  detected_queue = detect(net, person_idx, batched_image)
  cropped_queue = crop(detected_queue)
  estimated_queue = estimate(enet, cropped_queue)
  results = process(estimated_queue)
  persons = parse_results(results) 
  return persons

if __name__ == "__main__":

  enet = get_estimate_net()
  net, person_idx = get_detect_net()

  img_name = './examples/demo/000033.jpg'

  persons = []
  # for i in range(10):
  persons = inference_test(img_name, net, person_idx, enet)

  print(persons)