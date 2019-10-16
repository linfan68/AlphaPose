import json
from opt import opt
from service import *
# overwrite
opt.nClasses = 33

# person detection will run inference at inp_dim * inp_dim
opt.inp_dim = 192

# single pose estimation will run inference at inputResH * inputResW
# tested: size smaller than 256 * 192 will cause precision loss
opt.inputResH = 256
opt.inputResW = 192

class PoseEstnService(object):

    def __init__(self):
        self._context = None
        self._batch_size = 0
        self._num_requests = 0
        self._model_dir = '.'
        self.initialized = False

        self.enet = None
        self.person_idx = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context

        self._batch_size = context.system_properties["batch_size"]
        self._model_dir = context.system_properties["model_dir"]

        self.enet = get_estimate_net()

        self.initialized = True

    def preprocess(self, batch):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """

        # Take the input data and pre-process it make it inference ready
        # assert self._batch_size == len(batch), "Invalid input batch size: {}".format(len(batch))
        # return None

        def load_image(blob, box):
            mean = (0.485, 0.456, 0.406)
            std = (1.0, 1.0, 1.0)

            nparr = np.frombuffer(blob, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cvImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            img = mx.nd.array(cvImg)
            img_width, img_height = img.shape[1], img.shape[0]

            img = mx.nd.image.to_tensor(img)
            img = mx.nd.image.normalize(img, mean=mean, std=std)
            img = img.transpose(axes=[1, 2, 0])

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

            tensor = mx.nd.pad(tensor, mode='constant', constant_value=0.5, pad_width=pad_shape)
            tensor = tensor.reshape(3, opt.inputResH, opt.inputResW)

            return tensor, box, (left, up), (right, bottom)

        tensors = mx.nd.zeros([self._batch_size, 3, opt.inputResH, opt.inputResW])
        pt1 = mx.nd.zeros([self._batch_size, 2])
        pt2 = mx.nd.zeros([self._batch_size, 2])

        boxes = mx.nd.zeros([self._batch_size, 4])
        box_scores = mx.nd.ones([self._batch_size])

        self._num_requests = len(batch)
        for i in range(self._num_requests):
            box = json.loads(batch[i]['box'].decode())
            tensors[i], boxes[i], pt1[i], pt2[i] = load_image(batch[i]['body'], box)

        return (tensors, pt1, pt2, boxes, box_scores)

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
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

        tensors, pt1, pt2, boxes, box_scores = model_input

        heatmap_batch = self.enet(tensors.copyto(ctx))
        heatmap_batch = heatmap_batch[:, :17, :, :]
        heatmap_batch = heatmap_batch.copyto(mx.cpu())

        pose_hms, pose_coords, pose_scores = transform_fn(heatmap_batch, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)

        return pose_hms, pose_coords, pose_scores, boxes, box_scores

    def postprocess(self, inference_output):
        """
        Return predict result in batch.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        pose_hms, pose_coords, pose_scores, boxes, box_scores = inference_output

        roundFloat = lambda x: round(x * 1e3) / 1e3

        pose_coords = pose_coords.asnumpy()
        pose_scores = pose_scores.asnumpy()
        boxes = boxes.asnumpy()
        box_scores = box_scores.asnumpy()

        ps = []
        for i in range(self._num_requests):

            partData = np.concatenate([pose_coords[i], pose_scores[i]], axis=1)
            confidence = roundFloat(np.mean(pose_scores[i]))

            partData = np.reshape(partData, (-1)).tolist()
            partData = list(map(roundFloat, partData))

            box = boxes[i].tolist()
            box = list(map(roundFloat, box))

            boxScore = confidence

            ps.append(
                json.dumps({
                    'partData': partData,
                    'confidence': confidence,
                    'box': box,
                    'boxScore': boxScore
                })
            )

        return ps


    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """

        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)

_service = PoseEstnService()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)