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

class PoseDetnEstnService(object):

    def __init__(self):
        self._context = None
        self._batch_size = 0
        self._model_dir = '.'
        self.initialized = False

        self.net = None
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

        self.net, self.person_idx = get_detect_net()
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

        # print(batch)
        def load_image(blob, tensor_size):
          mean = (0.485, 0.456, 0.406)
          std = (0.229, 0.224, 0.225)

          # cvImg = cv2.imread(img_name)
          nparr = np.frombuffer(blob, np.uint8)
          image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
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
        tensor_batch, img_batch, img_size_batch, img_name_list = [], [], [], []

        for blob in batch:
          tensor_k, img_k, img_size_k = load_image(blob['body'], tensor_size)
          tensor_batch.append(tensor_k)
          img_batch.append(img_k)
          img_size_batch.append(img_size_k)
          img_name_list.append('placeholder')

        tensor_batch = mx.nd.concatenate(tensor_batch, axis=0)
        img_size_batch = mx.nd.array(img_size_batch, dtype='float32')
        img_size_batch = img_size_batch.tile(reps=[1, 2])

        return (tensor_batch, img_batch, img_size_batch, img_name_list)

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        detected_queue = detect(self.net, self.person_idx, model_input)
        cropped_queue = crop(detected_queue)
        estimated_queue = estimate(self.enet, cropped_queue)
        return estimated_queue

    def postprocess(self, inference_output):
        """
        Return predict result in batch.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        results = process(inference_output)
        # return ["OK"] * self._batch_size
        persons = parse_results(results) 
        return persons

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """

        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)

_service = PoseDetnEstnService()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)