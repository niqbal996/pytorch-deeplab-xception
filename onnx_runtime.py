"""
(optional) Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime
========================================================================
In this tutorial, we describe how to convert a model defined
in PyTorch into the ONNX format and then run it with ONNX Runtime.
ONNX Runtime is a performance-focused engine for ONNX models,
which inferences efficiently across multiple platforms and hardware
(Windows, Linux, and Mac and on both CPUs and GPUs).
ONNX Runtime has proved to considerably increase performance over
multiple models as explained `here
<https://cloudblogs.microsoft.com/opensource/2019/05/22/onnx-runtime-machine-learning-inferencing-0-4-release>`__
For this tutorial, you will need to install `ONNX <https://github.com/onnx/onnx>`__
and `ONNX Runtime <https://github.com/microsoft/onnxruntime>`__.
You can get binary builds of ONNX and ONNX Runtime with
``pip install onnx onnxruntime``.
Note that ONNX Runtime is compatible with Python versions 3.5 to 3.7.
``NOTE``: This tutorial needs PyTorch master branch which can be installed by following
the instructions `here <https://github.com/pytorch/pytorch#from-source>`__
"""

# Some standard imports
import io
import numpy as np

from torch import nn
from modeling.deeplab import *
from glob import glob
from torchvision import transforms
from dataloaders import custom_transforms as tr
import os
from PIL import Image
import torch.onnx
import torch.nn as nn
import torch.nn.init as init

######################################################################
# Ordinarily, you would now train this model; however, for this tutorial,
# we will instead download some pre-trained weights. Note that this model
# was not trained fully for good accuracy and is used here for
# demonstration purposes only.
#
# It is important to call ``torch_model.eval()`` or ``torch_model.train(False)``
# before exporting the model, to turn the model to inference mode.
# This is required since operators like dropout or batchnorm behave
# differently in inference and training mode.
#
cuda = False
model_url = '/home/naeem/git/pytorch-deeplab-xception/run/cropweed/deeplab-resnet/experiment_12_only_RGB_3_in_channels/checkpoint.pth.tar'
# Define network
model = DeepLab(num_classes=3,
                backbone='resnet',
                in_channels=3,
                output_stride=16,
                sync_bn=True,
                freeze_bn=False)

# Using cuda
# if args.cuda:
#     self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
#     patch_replication_callback(self.model)
#     self.model = self.model.cuda()

# Resuming checkpoint
if not os.path.isfile(model_url):
    raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
checkpoint = torch.load(model_url)
if cuda:
    model.module.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint['state_dict'])
# if not args.ft:
#     self.optimizer.load_state_dict(checkpoint['optimizer'])
best_pred = checkpoint['best_pred']
print("=> loaded NN model . . . '{}' " .format(model_url))


# set the model to inference mode
model.eval()

######################################################################
# Exporting a model in PyTorch works via tracing or scripting. This
# tutorial will use as an example a model exported by tracing.
# To export a model, we call the ``torch.onnx.export()`` function.
# This will execute the model, recording a trace of what operators
# are used to compute the outputs.
# Because ``export`` runs the model, we need to provide an input
# tensor ``x``. The values in this can be random as long as it is the
# right type and size.
# Note that the input size will be fixed in the exported ONNX graph for
# all the input's dimensions, unless specified as a dynamic axes.
# In this example we export the model with an input of batch_size 1,
# but then specify the first dimension as dynamic in the ``dynamic_axes``
# parameter in ``torch.onnx.export()``.
# The exported model will thus accept inputs of size [batch_size, 1, 224, 224]
# where batch_size can be variable.
#
# To learn more details about PyTorch's export interface, check out the
# `torch.onnx documentation <https://pytorch.org/docs/master/onnx.html>`__.
#

# Input to the model
img = Image.open("/home/naeem/datasets/structured_cwc/test/img/bonirob_2016-04-29-12-12-51_17_frame97.png")
size = img.size

tmp = ''
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# TODO this can mess up the pixel-wise GPS positions
composed_transforms = transforms.Compose([
            tr.FixedResize(size=513),
            tr.Normalize(mean=mean, std=std),
            tr.ToTensor()])
# TODO 'label' has dummy data here
sample = {'image': img, 'label': img, 'path': tmp}
sample = composed_transforms(sample)

x, target = sample['image'], sample['label']

#
# # TODO this can mess up the pixel-wise GPS positions
# composed_transforms = transforms.Compose([
#             tr.FixedResize(size=512),
#             tr.Normalize(mean=mean, std=std),
#             tr.ToTensor()])
# sample = {'image': x, 'label': tmp, 'path': tmp}
# sample = composed_transforms(sample)
#
# image, target = sample['image'], sample['label']
torch_out = model(x.unsqueeze(0))

# Export the model
torch.onnx.export(model,  # model being run
                  x.unsqueeze(0),  # model input (or a tuple for multiple inputs)
                  "./onnx/deeplab_crops.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=11,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                'output': {0: 'batch_size'}})

######################################################################
# We also computed ``torch_out``, the output after of the model,
# which we will use to verify that the model we exported computes
# the same values when run in ONNX Runtime.
#
# But before verifying the model's output with ONNX Runtime, we will check
# the ONNX model with ONNX's API.
# First, ``onnx.load("super_resolution.onnx")`` will load the saved model and
# will output a onnx.ModelProto structure (a top-level file/container format for bundling a ML model.
# For more information `onnx.proto documentation <https://github.com/onnx/onnx/blob/master/onnx/onnx.proto>`__.).
# Then, ``onnx.checker.check_model(onnx_model)`` will verify the model's structure
# and confirm that the model has a valid schema.
# The validity of the ONNX graph is verified by checking the model's
# version, the graph's structure, as well as the nodes and their inputs
# and outputs.
#

import onnx

onnx_model = onnx.load("./onnx/deeplab_crops.onnx")
onnx.checker.check_model(onnx_model)

######################################################################
# Now let's compute the output using ONNX Runtime's Python APIs.
# This part can normally be done in a separate process or on another
# machine, but we will continue in the same process so that we can
# verify that ONNX Runtime and PyTorch are computing the same value
# for the network.
#
# In order to run the model with ONNX Runtime, we need to create an
# inference session for the model with the chosen configuration
# parameters (here we use the default config).
# Once the session is created, we evaluate the model using the run() api.
# The output of this call is a list containing the outputs of the model
# computed by ONNX Runtime.
#

import onnxruntime

ort_session = onnxruntime.InferenceSession("./onnx/deeplab_crops.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x.unsqueeze(0))}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

######################################################################
# We should see that the output of PyTorch and ONNX Runtime runs match
# numerically with the given precision (rtol=1e-03 and atol=1e-05).
# As a side-note, if they do not match then there is an issue in the
# ONNX exporter, so please contact us in that case.
#


######################################################################
# Running the model on an image using ONNX Runtime
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# So far we have exported a model from PyTorch and shown how to load it
# and run it in ONNX Runtime with a dummy tensor as an input.

######################################################################
# For this tutorial, we will use a famous cat image used widely which
# looks like below
#
# .. figure:: /_static/img/cat_224x224.jpg
#    :alt: cat
#

######################################################################
# First, let's load the image, pre-process it using standard PIL
# python library. Note that this preprocessing is the standard practice of
# processing data for training/testing neural networks.
#
# We first resize the image to fit the size of the model's input (224x224).
# Then we split the image into its Y, Cb, and Cr components.
# These components represent a greyscale image (Y), and
# the blue-difference (Cb) and red-difference (Cr) chroma components.
# The Y component being more sensitive to the human eye, we are
# interested in this component which we will be transforming.
# After extracting the Y component, we convert it to a tensor which
# will be the input of our model.
#

from PIL import Image
import torchvision.transforms as transforms


######################################################################
# Now, as a next step, let's take the tensor representing the
# greyscale resized cat image and run the super-resolution model in
# ONNX Runtime as explained previously.
#

ort_inputs = {ort_session.get_inputs()[0].name: np.expand_dims(to_numpy(image), axis=0)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]

######################################################################
# At this point, the output of the model is a tensor.
# Now, we'll process the output of the model to construct back the
# final output image from the output tensor, and save the image.
# The post-processing steps have been adopted from PyTorch
# implementation of super-resolution model
# `here <https://github.com/pytorch/examples/blob/master/super_resolution/super_resolve.py>`__.
#

img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

# get the output image follow post-processing step from PyTorch implementation
final_img = img_out_y.convert("RGB")

# Save the image, we will compare this with the output image from mobile device
final_img.save("./onnx/onnx_output.png")


######################################################################
# .. figure:: /_static/img/cat_superres_with_ort.jpg
#    :alt: output\_cat
#
#
# ONNX Runtime being a cross platform engine, you can run it across
# multiple platforms and on both CPUs and GPUs.
#
# ONNX Runtime can also be deployed to the cloud for model inferencing
# using Azure Machine Learning Services. More information `here <https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-onnx>`__.
#
# More information about ONNX Runtime's performance `here <https://github.com/microsoft/onnxruntime#high-performance>`__.
#
#
# For more information about ONNX Runtime `here <https://github.com/microsoft/onnxruntime>`__.
#
