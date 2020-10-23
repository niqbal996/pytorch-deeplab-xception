import numpy as np
import cv2
import matplotlib.pyplot as plt


class Analysis(object):
    def __init__(self, tensor_name, tensors_dict):
        self.key = tensor_name
        self.dict = tensors_dict
        self.tensor = self.get_tensor_from_key()
        self.image = self.tensor2image()
        self.mask = self.image2mask()
        self.binary_image = self.image2binary()
        self.crop_filter = self.binary_image[0:129, 129:129 * 2]     
        self.weed_filter = self.binary_image[774:774+129, 387:387+129] 
        
    def image2binary(self, threshold=50):
        binary_image = self.image.copy()
        binary_image[binary_image > threshold] = 255
        binary_image[binary_image < threshold] = 0
        return binary_image
    
    def image2mask(self):
        out = np.zeros(self.tensor.shape, dtype=np.bool)
        arra = self.tensor.copy()
        arra = arra.astype(np.bool)
        filta = self.crop_filter.copy()
        filta = filta.astype(np.bool)
        plate = np.ones_like(filta, dtype=np.bool)
        ctr = 0
        for row in range(16):
            for col in range(16):
                if ctr <= arra.shape[0]:
                    curr = arra[(row * arra.shape[1]):(row + 1) * arra.shape[1],
                                (col * arra.shape[2]):(col + 1) * arra.shape[2]] & filta

                    if np.count_nonzero(curr) > 0:
                        out[(row * arra.shape[1]):(row + 1) * arra.shape[1],
                            (col * arra.shape[2]):(col + 1) * arra.shape[2]] = plate
                    else:
                        out[(row * arra.shape[1]):(row + 1) * arra.shape[1],
                            (col * arra.shape[2]):(col + 1) * arra.shape[2]] = ~plate

                    ctr += 1
        return out

    def get_tensor_from_key(self):
        # TODO here taking tensor 0 but can be added as an argument for other tensors
        tensor = self.dict[self.key][0]
        tensor = tensor.detach().cpu().numpy().reshape(tensor.shape[1], tensor.shape[2], tensor.shape[3])

        return tensor

    def tensor2image(self):
        arr = np.zeros((int(np.sqrt(self.tensor.shape[0])) * self.tensor.shape[1], 
                        int(np.sqrt(self.tensor.shape[0])) * self.tensor.shape[2]),
                       dtype=np.float32)
        ctr = 0
        for row in range(int(np.sqrt(self.tensor.shape[0]))):
            for col in range(int(np.sqrt(self.tensor.shape[0]))):
                if ctr <= self.tensor.shape[0]:
                    arr[(row * self.tensor.shape[1]):(row + 1) * self.tensor.shape[1], 
                        (col * self.tensor.shape[2]):(col + 1) * self.tensor.shape[2]] = self.tensor[ctr, :, :]
                    ctr += 1
        return arr
    
    def visualize_tensor(self, cv=True):
        if cv:
            cv2.imshow('tensor as image', self.image)
            cv2.waitKey()
        else:
            plt.imshow(self.image)
            plt.show()
        
        
    
    
        
        

        
# arra = visualize_tensor(x)
# arra = (arra * 255)
# fig1 = plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(arra)
# plt.subplot(1, 2, 2)
# plt.imshow(ref)
# plt.show()
# bina = arra.copy()
# bina[bina > 50] = 255
# bina[bina < 50] = 0
# filta = bina[0:129, 129:129 * 2]        # crop filter
# filta = bina[774:774+129,387:387+129]     # weed filter
# mask = gen_mask(bina, filta)
# plt.subplot(1, 2, 1)
# plt.imshow(arra)
# plt.subplot(1, 2, 2)
# plt.imshow(bina)
# # plt.show()

# plt.imshow(filta)
# plt.show()
