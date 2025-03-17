import glob
import os
import xml.etree.ElementTree as ET

import cv2 as cv
try:
    import cupy as np

    if not np.is_available():
        raise RuntimeError("CUDA not available; reverting to NumPy.")
except (ImportError, RuntimeError) as e:
    import numpy as np


class ImageStack:
    def __init__(self, image_dir, file_ext='.tif'):
        self.image_dir = image_dir
        self.image_filenames = glob.glob(os.path.join(self.image_dir, f'*{file_ext}*'))
        self.raw = np.stack([cv.imread(file) for file in self.image_filenames])
        self.image = self.raw
        self.stack_height = self.raw.shape[2]

    def median_filter(self, size=3):
        for i in range(self.stack_height):
            self.image[:, :, i] = cv.medianBlur(self.image[:, :, i], size)

    def local_directional_contrast_filter(self, nhood=50, threshold_factor=2):
        '''https://pubmed.ncbi.nlm.nih.gov/22312585/'''
        for i in range(self.stack_height):
            layer = self.image[:, :, i]
            medians = cv.medianBlur(layer, nhood)
            invalid = medians / threshold_factor > layer > threshold_factor * medians
            layer[invalid] = medians[invalid]
            self.image[:, :, i] = layer

    def show(self):
        for i in range(self.stack_height):
            cv.imshow(f'{self.image_path} Layer {i}', self.image[:, :, i])

    def save(self, filename=None):
        for i in range(self.stack_height):
            if filename is None:
                parts = self.image_filenames[i].split('.')
            else:
                parts = filename.split('.')
                filename = parts[0] + f'_layer{i}_processed.tif' + parts[1]
                cv.imwrite(filename, self.image)

class HSDFMImage(ImageStack):
    def __init__(self, image_dir):
        super().__init__(image_dir)
        self.vascular_map = None
        self.threshold = None

        ## TODO: Update to match metadata format of MATLAB automater instead of PV.
        ## TODO: Inlcude extraction of exposure times AND filter wavelengths for ALL images in stack as ndarrays
        self.xml_file = glob.glob(os.path.join(image_dir, '*.xml'))[0]
        self.tree = ET.parse(self.xml_file)
        for state in self.tree.getroot().findall('.//PVStateValue'):
            if state.get('key') == 'camera_exposureTime':
                self.t = int(state.get('value'))
                break

    def normalize(self, bg, ref):
        self.image = ((self.image / self.t) - (bg.image / bg.t)) / ((ref.image / ref.t) - (bg.image / bg.t))

    def map_vasculature(self, method='threshold', threshold=None):
        match method:
            case 'threshold' | 't':
                self.threshold = threshold
                if self.threshold is None:
                    self.threshold, self.mask = cv.threshold(self.image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                else:
                    self.mask = cv.threshold(self.image, self.threshold, 255, cv.THRESH_BINARY)

            case 'gabor' | 'g':
                '''https://doi.org/10.1364/BOE.7.003247'''

            case 'matched' | 'm':
                '''https://doi.org/10.1109/42.34715'''

    def fit_model(self, model):
        pass


class MPMImage(ImageStack):
    def __init__(self, image_dir):
        super().__init__(image_dir)


class HSDFMPMStack:
    def __init__(self, hsdf_image, mpm_image):
        self.hsdf_image = hsdf_image
        self.mpm_image = mpm_image

    def show(self):
        self.hsdf_image.show()
        self.mpm_image.show()

    def fit_krogh(self):
        # Find average distance from center of vasclature in hsdf_image.vascular_map to p50 for mpm_image NADH channel
        pass