import numpy as np
import torch
import torch.nn.functional as F
from feature_based.SIFT_SURF_ORB_RIFT_SCB_HOPC import trans2DArray2matlab

import matlab.engine
eng = matlab.engine.start_matlab()

class SiftFlowTorch(object):
    """ Computes dense SIFT Flow [1] descriptors from a batch of images. It
    uses PyTorch to perform operations on GPU (if available) to significantly
    speedup the process. This implementation is a port of the origina
    implementation available at
    https://people.csail.mit.edu/celiu/SIFTflow/.

    This code is able to process a batch of images simultaneously for better
    performance. The most expensive operation when running in GPU mode is the
    allocation of the space for the descriptors on the GPU. However, this step
    is only performed when the shape of the input batch changes. Subsequent
    calls using batches with the same shape as before will reuse the memory and
    will, therefore, be much faster.

    Usage:
        from sift_flow_torch import SiftFlowTorch

        sift_flow = SiftFlowTorch()
        imgs = [
            read_some_image,
            read_another_image
        ]
        descs = sift_flow.extract_descriptor(imgs) # This first call can be
                                                   # slower, due to memory
                                                   # allocation
        imgs2 = [
            read_yet_another_image,
            read_even_one_more_image
        ]
        descs2 = sift_flow.extract_descriptor(imgs2) # Subsequent calls are
                                                     # faster, if images retain
                                                     # the same shape

        # descs[0] is the descriptor of imgs[0] and so on.

    Args:
        cell_size : int, optional
            Size of the side of the cell used to compute the descriptor.
        step_size, : int, optional
            Distance between the descriptor sampled points.
        is_boundary_included : boolean
            If False, the descriptor is not computed for pixels in the image
            boundaries, to avoid boundary effects.
        num_bins : int, optional
            Number of bins of the descriptor.
        cuda : boolean, optional
            If True, operations are done on GPU (if available).
        fp16 : boolean, optional
            Whether to use half-precision floating points for computing the
            descriptors. Half-precision mode uses less memory and it may be
            slightly faster but less accurate.
        return_numpy : boolean, optional
            If True, transfers the descriptor from pytorch to numpy before
            returning. This will increase the running time due to the memory
            transfer.

    References:
    [1] C. Liu; Jenny Yuen; Antonio Torralba. "SIFT Flow: Dense correspondence
        across scenes and its applications." IEEE Transactions on Pattern
        Analysis and Machine Intelligence 33.5 (2010): 978-994.
    """
    def __init__(self,
                 cell_size=2,
                 step_size=1,
                 is_boundary_included=True,
                 num_bins=8,
                 cuda=True,
                 fp16=False,
                 return_numpy=False):
        self.cell_size = cell_size
        self.step_size = step_size
        self.is_boundary_included = is_boundary_included
        self.num_bins = num_bins
        self.return_numpy = return_numpy
        self.cuda = cuda and torch.cuda.is_available()
        self.fp16 = fp16 and self.cuda
        if cuda and not torch.cuda.is_available():
            print('WARNING! CUDA mode requested, but',
                  'torch.cuda.is_available() is False.',
                  'Operations will run on CPU.')
        if fp16 and not self.cuda:
            print('WARNING! FP16 can only be used in CUDA mode, but',
                  'CUDA is not enabled. FP32 will be used instead.')

        # this parameter controls decay of the gradient energy falls into a bin
        # run SIFT_weightFunc.m to see why alpha = 9 is the best value
        self.alpha = 9

        self.theta = 2 * np.pi / self.num_bins

        self.gradient = None
        self.imax_mag = None
        self.sin_bins = None
        self.cos_bins = None
        self.offsets = None
        self.descs = None
        self.grad_filter = None
        self.max_batch_size = 1

        self.filter = self._compute_filter()

        if self.fp16:
            self.epsilon = 1e-3
        else:
            self.epsilon = 1e-10

    def extract_descriptor(self,
                           images):
        """ Main function of this class, which extracts the descriptors from
        a batch of images.

        Args:
            images : list of 3D array of int or float.
                List of images to form the batch. All images should have the
                same shape [Hi, Wi, Ci], with any number of channels Ci. The
                pixel values are assumed to be in the interval [0, 255].

        Returns:
            descs : 4D array of floats
                Grid of SIFT Flow descriptors for the given image as an array
                of dimensionality (N, Co, Ho, Wo) where
                ``N = len(images)``
                ``Co = 16 * num_bins``
                ``Ho = floor((Hi - D) / step_size)``
                ``Wo = floor((Wi - D) / step_size)``
                ``D = 0 if is_boundary_included else 4*cell_size``
        """
        images = np.stack(images, axis=0).transpose(0, 3, 1, 2).astype(np.float32)
        images = torch.from_numpy(images)
        if self.fp16:
            images = images.half()
        else:
            images = images.float()
        if self.cuda:
            images = images.cuda()
        images /= 255.0

        self.batch_size = images.shape[0]
        self.max_batch_size = max(self.max_batch_size, self.batch_size)

        if self.grad_filter is None:
            kernel = np.array(
                [-1, 0, 1, -2, 0, 2, -1, 0, 1], np.float32).reshape(3, 3)
            gf = np.zeros((images.shape[1], images.shape[1], 3, 3))
            for i in range(gf.shape[0]):
                gf[i, i] = kernel
            self.grad_filter = torch.from_numpy(gf)
            if self.fp16:
                self.grad_filter = self.grad_filter.half()
            else:
                self.grad_filter = self.grad_filter.float()
            if self.cuda:
                self.grad_filter = self.grad_filter.cuda()
        images_pad = F.pad(
            images, (1, 1, 1, 1), mode='replicate')
        dx = F.conv2d(images_pad, self.grad_filter)#images_pad的shape为[batchsize,channesl,H,w],self.grad_filter的[channesls,channels,3,3],conv2d中3个通道依次用卷积计算梯度
        dy = F.conv2d(images_pad, self.grad_filter.permute(0, 1, 3, 2))

        # Get the maximum gradient over the channels and estimate the
        # normalized gradient
        magsrc = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
        mag, max_mag_idx = torch.max(magsrc, dim=1, keepdim=True)#获得RGB通道的梯度的最大值mag和哪个通道max_mag_idx
        if (self.imax_mag is None or
                self.imax_mag.shape[0] < self.max_batch_size or
                self.imax_mag.shape[2:] != images.shape[2:]):
            self.imax_mag = torch.zeros(
                (self.max_batch_size,) + images.shape[1:])#imax_mag的shpae为[max_batch_size,imgages.shape[1],imgges.shape[2],images.shape[3]],即[B,C,H,W]
            if self.fp16:
                self.imax_mag = self.imax_mag.half()
            else:
                self.imax_mag = self.imax_mag.float()
            if self.cuda:
                self.imax_mag = self.imax_mag.cuda()
        imax_mag = self.imax_mag[:self.batch_size]
        imax_mag[:] = 0
        imax_mag = imax_mag.scatter_(1, max_mag_idx, 1)#将max_mag_idx([2,1,436,1024])按照one_hot编码[2,3,436,1024]

        if (self.gradient is None or
                self.gradient.shape[0] < self.max_batch_size or
                self.gradient.shape[2:] != images.shape[2:]):
            self.gradient = torch.zeros(
                self.max_batch_size, 2, images.shape[2], images.shape[3])
            if self.fp16:
                self.gradient = self.gradient.half()
            else:
                self.gradient = self.gradient.float()
            if self.cuda:
                self.gradient = self.gradient.cuda()
        gradient = self.gradient[:self.batch_size]
        gradient[:, 0] = (
            torch.sum(dx * imax_mag, dim=1) / (mag[:, 0] + self.epsilon))
        gradient[:, 1] = (
            torch.sum(dy * imax_mag, dim=1) / (mag[:, 0] + self.epsilon))

        # Get the pixel-wise energy for each orientation band
        bin_shape = (self.max_batch_size, 1, images.shape[2], images.shape[3])#(2,1,436,1024)
        if self.sin_bins is None or self.sin_bins.shape != bin_shape:
            idx = torch.arange(self.num_bins).reshape(1, -1, 1, 1)#num_bins=8,[[[[0],[1],...[7]]]]
            if self.fp16:
                idx = idx.half()
            else:
                idx = idx.float()
            if self.cuda:
                idx = idx.cuda()
            idx = idx.repeat(
                self.max_batch_size, 1, images.shape[2], images.shape[3])#idx的shape为[2,8,436,1024]
            self.sin_bins = torch.sin(idx * self.theta)#theta是2*pi/8,表示把360度分为8份
            self.cos_bins = torch.cos(idx * self.theta)
        imband = (self.cos_bins[:self.batch_size] * gradient[:, :1] +
                  self.sin_bins[:self.batch_size] * gradient[:, 1:2])
        imband = torch.max(imband, torch.zeros_like(imband))
        if self.alpha > 1:
            imband = torch.pow(imband, self.alpha)
        imband *= mag

        # Filter out the SIFT feature
        imband_cell = self._filter_features(imband)

        # Allocate buffer for the sift image
        siftdim = self.num_bins * 16
        sift_height = images.shape[2] // self.step_size
        sift_width = images.shape[3] // self.step_size
        x_shift = 0
        y_shift = 0
        if not self.is_boundary_included:
            sift_height = (images.shape[2]-4*self.cell_size) // self.step_size
            sift_width = (images.shape[3]-4*self.cell_size) // self.step_size
            x_shift = 2 * self.cell_size
            y_shift = 2 * self.cell_size

        self._compute_offsets(
            images.shape, sift_height, sift_width, x_shift, y_shift)

        desc_shape = (
            self.max_batch_size, siftdim, sift_height, sift_width)
        if self.descs is None or self.descs.shape != desc_shape:
            self.descs = torch.empty(desc_shape)
            if self.fp16:
                self.descs = self.descs.half()
            else:
                self.descs = self.descs.float()
            if self.cuda:
                self.descs = self.descs.cuda()
        descs = self.descs[:self.batch_size]
        for i in range(4):
            for j in range(4):
                idx = 4 * i + j
                feats = F.grid_sample(
                    imband_cell, self.offsets[
                        :self.batch_size, :, :, 2*idx:2*idx+2])
                    # mode='nearest')

                start = idx*self.num_bins
                end = (idx+1)*self.num_bins
                descs[:, start:end] = feats

        mag = descs.norm(dim=1, keepdim=True)
        descs /= mag + 0.01

        if self.return_numpy:
            descs = descs.detach().cpu().numpy()

        return descs

    def _compute_filter(self):
        filt = np.zeros(
            (self.num_bins, self.num_bins, 1, self.cell_size*2+1))
        for i in range(self.num_bins):
            filt[i, i, 0, 0] = 0.25
            filt[i, i, 0, self.cell_size+1] = 0.25
            for j in range(1, self.cell_size+1):
                filt[i, i, 0, j+1] = 1.0
        filt = torch.from_numpy(filt)
        if self.fp16:
            filt = filt.half()
        else:
            filt = filt.float()
        if self.cuda:
            filt = filt.cuda()
        return filt

    def _compute_offsets(self,
                         images_shape,
                         sift_height,
                         sift_width,
                         x_shift,
                         y_shift):
        if (self.offsets is None or
                self.offsets.shape[0] < self.max_batch_size or
                sift_height != self.offsets.shape[1] or
                sift_width != self.offsets.shape[2]):
            xv, yv = torch.meshgrid(
                [torch.arange(0, sift_height),
                 torch.arange(0, sift_width)])
            grid = torch.stack([yv, xv], dim=2).unsqueeze(0)
            if self.fp16:
                grid = grid.half()
            else:
                grid = grid.float()
            if self.cuda:
                grid = grid.cuda()
            grid *= self.step_size
            self.offsets = torch.zeros(
                self.max_batch_size, sift_height, sift_width, 32)
            if self.fp16:
                self.offsets = self.offsets.half()
            else:
                self.offsets = self.offsets.float()
            if self.cuda:
                self.offsets = self.offsets.cuda()
            for i in range(-1, 3):
                for j in range(-1, 3):
                    off_y = y_shift + grid[:, :, :, 1] + i*self.cell_size
                    off_y = torch.max(off_y, torch.zeros_like(off_y))
                    off_y = torch.min(
                        off_y, torch.zeros_like(off_y) + images_shape[2] - 1)
                    off_x = x_shift + grid[:, :, :, 0] + j*self.cell_size
                    off_x = torch.max(off_x, torch.zeros_like(off_x))
                    off_x = torch.min(
                        off_x, torch.zeros_like(off_x) + images_shape[3] - 1)
                    idx = 4 * (i + 1) + j + 1
                    self.offsets[:, :, :, 2*idx] = (
                        off_x.repeat(images_shape[0], 1, 1))
                    self.offsets[:, :, :, 2*idx+1] = (
                        off_y.repeat(images_shape[0], 1, 1))
            self.offsets[:, :, :, ::2] /= images_shape[3] - 1
            self.offsets[:, :, :, 1::2] /= images_shape[2] - 1
            self.offsets *= 2
            self.offsets -= 1

    def _filter_features(self,
                         imband):
        radius = self.filter.shape[3] // 2
        imband_smooth = imband
        imband_smooth = F.pad(
            imband_smooth, (radius, radius, 0, 0), mode='replicate')
        imband_smooth = F.conv2d(imband_smooth, self.filter)
        # print('imband_smooth', imband_smooth.min(), imband_smooth.max(), torch.abs(imband_smooth).sum())
        imband_smooth = F.pad(
            imband_smooth, (0, 0, radius, radius), mode='replicate')
        imband_smooth = F.conv2d(
            imband_smooth, self.filter.permute(0, 1, 3, 2))
        # print('imband_smooth2', imband_smooth.min(), imband_smooth.max(), torch.abs(imband_smooth).sum())
        return imband_smooth

def SIFT_feature(img):
    sift_step_size = 1
    sift_flow = SiftFlowTorch(
        cell_size=2,
        step_size=sift_step_size,
        is_boundary_included=True,
        num_bins=8,
        cuda=True,
        fp16=True,
        return_numpy=False)

    imgs = img if isinstance(img,list) else [img]
    imgs = [_[:,:,np.newaxis] for _ in imgs]
    descs = sift_flow.extract_descriptor(imgs)
    descs = [np.array(_).transpose(1, 2, 0) for _ in descs]
    return descs


def kornia_dense_SIFT(img):
    import kornia
    img = torch.rand(2, 1, 200, 300)
    SIFT = kornia.feature.DenseSIFTDescriptor()
    descs = SIFT(img)
    return descs


def HardNet_feature(img):
    pass

def rift_feature(img,t):
    eng.cd('H:/mlx_thermal_Images/code/TAMM/RIFT_registration_fuben')
    des=eng.RIFT_feature_demo_python(trans2DArray2matlab(img, matlab_type=t))

    return des