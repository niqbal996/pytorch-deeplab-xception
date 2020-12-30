import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import torch
import os


class Analysis(object):
    def __init__(self, tensors_dict, label=1, path=None):
        self.channel = 1
        self.key = None
        self.dict = tensors_dict
        self.out_dir = path
        self.prediction = None
        # self.tensor = self.get_tensor_from_key()
        # self.image = self.tensor2image()
        # self.binary_image = self.image2binary()
        # self.crop_filter = self.binary_image[0:129, 129:129 * 2]
        # self.weed_filter = self.binary_image[774:774 + 129, 387:387 + 129]
        self.focus = label
        self.keys = [x for x in self.dict.keys() if not isinstance(x, list)]
        self.analysis = self.generate_images()
        # self.mask = self.image2mask()

    def image2binary(self, image, threshold=50):
        binary_image = image.copy() * 255
        binary_image[binary_image > threshold] = 255
        binary_image[binary_image < threshold] = 0
        return binary_image
    
    def image2mask(self, image, crop_filter, tensor):
        out = np.zeros(image.shape, dtype=np.bool)
        arra = image.copy()
        arra = arra.astype(np.bool)
        filta = crop_filter.copy()
        filta = filta.astype(np.bool)
        plate = np.ones_like(filta, dtype=np.bool)
        ctr = 0
        for row in range(math.floor(out.shape[0]/tensor.shape[0])):
            for col in range(math.floor(out.shape[0]/tensor.shape[1])):
                if ctr <= arra.shape[0]:
                    curr = arra[(row * filta.shape[0]):(row + 1) * filta.shape[0],
                                (col * filta.shape[1]):(col + 1) * filta.shape[1]] & filta

                    if np.count_nonzero(curr) > 0:
                        out[(row * filta.shape[0]):(row + 1) * filta.shape[0],
                            (col * filta.shape[1]):(col + 1) * filta.shape[1]] = plate
                    else:
                        out[(row * filta.shape[0]):(row + 1) * filta.shape[0],
                            (col * filta.shape[1]):(col + 1) * filta.shape[1]] = ~plate

                    ctr += 1
        return out

    def get_tensor_from_key(self, key):
        # TODO here taking tensor 0 but can be added as an argument for other tensors
        if len(self.dict[key]) > 1:
            tensor = self.dict[key][self.channel]
        else:
            tensor = self.dict[key][0]
        tensor = tensor.detach().cpu().numpy().reshape(tensor.shape[1], tensor.shape[2], tensor.shape[3])

        return tensor

    def tensor2image(self, tensor):
        arr = np.zeros((math.ceil(np.sqrt(tensor.shape[0])) * tensor.shape[1],
                        math.ceil(np.sqrt(tensor.shape[0])) * tensor.shape[2]),
                       dtype=np.float32)
        ctr = 0
        for row in range(int(np.sqrt(tensor.shape[0]))):
            for col in range(int(np.sqrt(tensor.shape[0]))):
                if ctr <= tensor.shape[0]:
                    arr[(row * tensor.shape[1]):(row + 1) * tensor.shape[1], 
                        (col * tensor.shape[2]):(col + 1) * tensor.shape[2]] = tensor[ctr, :, :]
                    ctr += 1
        return arr
    
    def visualize_tensor(self, image, cv=True):
            if cv:
                print(np.max(image))
                cv2.imshow('tensor as image', image)
                cv2.waitKey()
            else:
                plt.imshow(image)
                plt.show()

    def generate_images(self):
        image_tensors = {}
        for key in self.keys:
            image_tensors[key] = {}
            image_tensors[key]['tensor'] = self.get_tensor_from_key(key)
            image_tensors[key]['image'] = self.tensor2image(image_tensors[key]['tensor'])
            image_tensors[key]['binary_image'] = self.image2binary(image_tensors[key]['image'])
            image_tensors[key]['crop_filter'] = image_tensors[key]['binary_image'][0:129, 129:129 * 2]
            image_tensors[key]['weed_filter'] = image_tensors[key]['binary_image'][774:774 + 129, 387:387 + 129]
            # image_tensors[key]['mask'] = self.image2mask(image_tensors[key]['image'],
            #                                              image_tensors[key]['crop_filter'],
            #                                              image_tensors[key]['tensor'])
            # see.visualize_tensor(see.image)
            # see.save_tensor(see.image, self.saver.experiment_dir)

        return image_tensors

    # def backtrace(self, out):
    #     self.prediction = out
    #     class_regions = np.argmax(out, axis=1)
    #     class_regions = np.squeeze(class_regions, axis=0)
    #     activation_map = np.zeros(self.prediction.shape[2:4])
    #     activation_map[class_regions == self.focus] = self.prediction[0, self.channel, class_regions == self.focus]
    #     # Note! we must propagate from the last layer backwards towards the first layers, hence, the reversed iterator.
    #     for key in reversed(self.keys):
    #         if len(self.dict[key]) > 1:
    #             tensor = self.dict[key][self.channel]
    #         else:
    #             tensor = self.dict[key][0]
    #         curr_activation = activation_map.copy()
    #         # curr_activation = np.resize(curr_activation, (tensor.shape[2], tensor.shape[3]))
    #         curr_activation = curr_activation[np.newaxis, np.newaxis, :, :]
    #         curr_activation = F.interpolate(torch.from_numpy(curr_activation),
    #                                         size=[tensor.shape[2], tensor.shape[3]],
    #                                         mode='bilinear',
    #                                         align_corners=True)
    #         curr_activation = curr_activation.data.numpy().reshape(curr_activation.shape[2:4])
    #         for filter in range(tensor.shape[1]):
    #             curr_filter = tensor[:, filter, :, :].numpy().squeeze(0)
    #             mask = np.zeros_like(curr_filter)
    #             mask[np.where(curr_activation != 0)] = curr_filter[np.where(curr_activation != 0)]
    #             img1 = self.normalize_array(mask)
    #             # img2 = self.normalize_array(curr_filter)
    #             # cv2.imshow('image', np.concatenate([img1, img2], axis=1))
    #             cv2.imshow('image', mask)
    #             cv2.waitKey(0)
    #             # print('activation_max {} and filter max {}'.format(np.max(curr_activation), np.max(curr_filter)))
    #
    #     return 0

    def save_activations(self):
        for key in self.keys:
            image = self.analysis[key]['image']
            # print('min: {} max: {}'.format(np.min(image), np.max(image)))
            image = ((image / np.max(image)) * 255).astype(np.int8)
            # cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if self.out_dir is not None:
                cv2.imwrite(os.path.join(self.out_dir, '{}.png'.format(self.key)), image)
            else:
                raise Exception('[ERROR] You have not provided any output directory to save files to!')

    def normalize_array(self, arra):
        arra = (arra / np.max(arra)) * 255
        arra = arra.astype(np.uint8)
        return arra
