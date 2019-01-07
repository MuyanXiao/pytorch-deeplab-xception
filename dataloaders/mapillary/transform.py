from PIL import Image
from torchvision.transforms import functional as tfn
import torch
import numpy as np


class SegmentationTransform:
    def __init__(self, longest_max_size, rgb_mean, rgb_std):
        self.longest_max_size = longest_max_size
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

    def __call__(self, img, normalize=True):
        """# Scaling

        if len(self.longest_max_size)>1:
            img = img.resize(tuple(self.longest_max_size), resample=resample_mode)
        else:
            scale = self.longest_max_size[0]/float(max(img.size[0],img.size[1]))
            if scale != 1.:
                out_size = tuple(int(dim * scale) for dim in img.size)
                img = img.resize(out_size, resample=resample_mode)


        # Scaling
        scale = self.longest_max_size / float(max(img.size[0], img.size[1]))
        if scale < 1.:
            out_size = tuple(int(dim * scale) for dim in img.size)
            img = img.resize(out_size, resample=Image.BILINEAR)
        
        """
        # Convert to torch and normalize
        if normalize:
            img = tfn.to_tensor(img)
            # print(img.size())
            # img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
            # img.div_(img.new(self.rgb_std).view(-1, 1, 1))
            img.sub_(img.new_zeros(1).add_(self.rgb_mean))
            img.div_(img.new_zeros(1).add_(self.rgb_std))
        else:
            img = torch.from_numpy(np.array(img))

        return img
