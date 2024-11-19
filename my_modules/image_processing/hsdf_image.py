import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


class HSDFMImage:
    def __init__(self, image_dir):
        self.normalized = None
        self.image_dir = image_dir
        self.image_filenames = glob.glob(os.path.join(image_dir, '*.tif*'))
        self.raw = np.array(Image.open(self.image_filenames[0]))
        self.xml_file = glob.glob(os.path.join(image_dir, '*.xml'))[0]
        self.tree = ET.parse(self.xml_file)
        for state in self.tree.getroot().findall('.//PVStateValue'):
            if state.get('key') == 'camera_exposureTime':
                self.t = int(state.get('value'))
                break

    def normalize(self, bg, ref):
        self.normalized = ((self.raw / self.t) - (bg.raw / bg.t)) / ((ref.raw / ref.t) - (bg.raw / bg.t))

    def show(self):