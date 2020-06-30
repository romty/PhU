import numpy as np
import scipy.io
from sklearn.metrics import mean_absolute_error
from sewar.full_ref import ssim, uqi, vifp
from unwrap import unwrap
import model
import data_generator as dg
import cv2

def make_pair(start, end):
    pairs = []
    for i in range(start, end):
        pairs.append([('wrap/wraped_' + str(i + 1) + '.mat'), ('unwrap/un_wraped_' + str(i + 1) + '.mat')])

    return pairs
from random import shuffle
test_pair = make_pair(3750, 4000)
from random import sample, choice
temp = choice(test_pair)
print (temp[0], "GAB", temp[1])
import matplotlib.pyplot as plt
img = scipy.io.loadmat(temp[0])['wrap']
mask_x= unwrap(img,wrap_around_axis_0=False, wrap_around_axis_1=False, wrap_around_axis_2=False)
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(dg.normalize_angle(img), cmap='jet')
plt.subplot(122)
plt.imshow(dg.normalize_angle(mask_x), cmap='jet')
plt.show()
class_map=1
test_generator= dg.DataGenerator(test_pair, class_map, batch_size=20, dim=(256,256,1) ,shuffle=True)
test_steps = test_generator.__len__()
test_steps

class eval_denoising:
    def __init__(self, I1, I2,  # I1 and I2 are the two images to compare
                 I3=None,  # Image bruitée
                 PSNR_peak=255):  # default value for PSNR
        self.I1 = I1  # result
        self.I2 = I2  # objective
        self.Idiff = I2 - I1

        if I3 != None:
            self.I_noise_only = I3 - I1
        else:
            self.I_noise_only = None

        self.euclidian_distance = None

        self.MAE = None

        self.RMSE = None

        self.peak = PSNR_peak
        self.PSNR = None

        self.SSIM = None

        self.VIF = None

    def compute_euclidian_distance(self):
        """
        Compute euclidian distance between two images
        """
        self.euclidian_distance = np.linalg.norm(self.I1 - self.I2)
        return ()

    def compute_MAE(self):
        """
        Computes the Mean Absolute Error between two images
        """
        self.MAE = np.mean(np.abs(self.I1 - self.I2))
        return ()

    def compute_RMSE(self):
        """
        Computes the Root Mean Square Error between two images
        """
        self.RMSE = np.sqrt(((self.I1 - self.I2) ** 2).mean())
        return ()

    def compute_PSNR(self):
        """
        Computes the Peak Signal to Noise Ratio between two images
        """
        img_1 = np.array(self.I1 / np.max(self.I1))
        img_2 = np.array(self.I2 / np.max(self.I2))
        x = (img_1.squeeze() - img_2.squeeze()).flatten()
        self.PSNR = 10 * np.log10(self.peak ** 2 / np.mean(x ** 2))
        return ()

    def compute_SSIM(self):
        """
        Compute Universal Quality Image Index between two images
        """
        self.SSIM = ssim(self.I1, self.I2)
        return ()

    def compute_VIF(self):
        """
        Compute Visual Information Fidelity between two images
        """
        self.VIF = vifp(self.I1, self.I2)
        return ()

    def all_evaluate(self):
        """
        Compute and display all available results
        """
        #self.compute_euclidian_distance()
       # print("Euclidian distance : ", self.euclidian_distance)
        self.compute_MAE()
        print("MAE : ", self.MAE)
        self.compute_RMSE()
        print("RMSE : ", self.RMSE)
        #self.compute_SSIM()

        #self.compute_UQI()
       # print("UQI : ", self.UQI)
        #self.compute_VIF()
        #print("VIF : ", self.VIF)
        return  self.MAE, self.RMSE, self.SSIM


if (__name__ == "__main__"):
    path = "predicted_unwrapped/"
    window_size = 256
    n_CLASSES = 1
    ##############################
    unet_model = model.unet(window_size, window_size, n_CLASSES, data_format='channels_last')
    unet_model.summary()
    unet_model.load_weights('unet_weights.h5')
    print ('u-net model_loaded')
    runet_model = model.r_unet(window_size, window_size, n_CLASSES, data_format='channels_last')
    runet_model.summary()
    runet_model.load_weights('r_unet_weights.h5')
    print ('r_unet model_loaded')
    r2unet_model = model.r2_unet(window_size, window_size, n_CLASSES, data_format='channels_last')
    r2unet_model.summary()
    r2unet_model.load_weights('r2_unet_weights.h5')
    print ('r_unet model_loaded')
    ##############################

    for i in range(10):#len(test_pair)
        temp = test_pair[i]
        test_img = scipy.io.loadmat(temp[0])['wrap']
        forpredict = np.reshape(dg.normalize_angle(test_img), (-1, test_img.shape[0], test_img.shape[1],1 ))
        print (forpredict.shape)
        mask_x = unwrap(test_img, wrap_around_axis_0=False, wrap_around_axis_1=False, wrap_around_axis_2=False)
        unet_wrapped = unet_model.predict(forpredict)
        runet_wrapped = runet_model.predict(forpredict)
        r2unet_wrapped = r2unet_model.predict(forpredict)
        print (unet_wrapped.shape,runet_wrapped.shape,  r2unet_wrapped.shape)
        unet_wrapped = np.squeeze(unet_wrapped, axis=-1)
        runet_wrapped = np.squeeze(runet_wrapped, axis=-1)
        r2unet_wrapped = np.squeeze(r2unet_wrapped, axis=-1)
        print (unet_wrapped.shape, runet_wrapped.shape, r2unet_wrapped.shape)
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(unet_wrapped[0], interpolation= 'bicubic',cmap='jet')
        axs[0, 0].set_title('unet_wrapped')
        axs[0, 1].imshow(runet_wrapped[0],interpolation= 'bicubic', cmap='jet')
        axs[0, 1].set_title('runet_wrapped')
        axs[1, 0].imshow(r2unet_wrapped[0], interpolation= 'bicubic',cmap='jet')
        axs[1, 0].set_title('r2unet_wrapped')
        axs[1, 1].imshow(mask_x, cmap='jet')
        axs[1, 1].set_title('groundtruth')
        #plt.subplot(121)
        #plt.imshow(unet_wrapped[0], cmap='jet')
       # plt.subplot(122)
       # plt.imshow(r2unet_wrapped[0], cmap='jet')
        #plt.subplot(123)
       # plt.imshow(runet_wrapped[0], cmap='jet')
        #plt.subplot(124)
        #plt.imshow(mask_x, cmap='jet')
        plt.show()
        unet_evaluation = eval_denoising(unet_wrapped[0],mask_x)
        unet_evaluation.all_evaluate()
        (score, diff) = ssim(unet_wrapped[0],mask_x)
        print("SSIM : ", score)
        #runet_evaluation = eval_denoising(runet_wrapped[0], mask_x)
        #runet_evaluation.all_evaluate()
        #r2unet_evaluation = eval_denoising(r2unet_wrapped[0], mask_x)
        #r2unet_evaluation.all_evaluate()
    ###############################

    #runet_model = model.r_unet(window_size, window_size, n_CLASSES, data_format='channels_last')
    #runet_model.summary()
   # runet_model.load_weights('r_unet_weights.h5')
    #print ('r_unet model_loaded')
    ##############################
    #r_predict = runet_model.predict_generator(test_generator, steps=test_steps)
    ###############################

    #r2unet_model =model.r2_unet(window_size, window_size, n_CLASSES, data_format='channels_last')
   # r2unet_model.summary()
   # r2unet_model.load_weights('r2_unet_weights.h5')
    #print ('r_unet model_loaded')
    ##############################
    #r2_predict = r2unet_model.predict_generator(test_generator, steps=test_steps)

    ###############################
