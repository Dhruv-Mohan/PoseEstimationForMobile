import json
import requests
import os
import numpy as np
from utils.landmark_point import l_point
from utils.pointIO import *

"""Class to parse and extract data from json files"""

class json_reader(object):
    def __init__(self, json_file_path):
        with open(json_file_path, 'r') as json_file:
            self.__data__ = json.load(json_file)
            self.__image_name__ = None
            self.__image_url__ = None
            self.__pts_name__ = None
            self.__landmark_pts__ = []
            self.__num_landmark_pts = None

    def load_image_reqs(self):
        self.__image_name__ = self.__data__['image_name_with_ext']
        self.__image_url__ = self.__data__['url']

    def load_pts_reqs(self):
        self.__pts_name__ = os.path.splitext(self.__image_name__)[0] + '.pts'

    def get_data(self):
        return self.__data__

    def load_landmarks(self):
        landmarks = self.__data__['landmarks']

        for landmark in landmarks:
            lm_pt = l_point(**landmark)
            self.__landmark_pts__.append(lm_pt)
        self.__num_landmark_pts = len(self.__landmark_pts__)

    def get_landmark_points(self):
        self.load_landmarks()
        points_coord = []
        for landmark in self.__landmark_pts__:
            points_coord.append([landmark.get_pt()])
        return np.asarray(points_coord).reshape([-1,2])

    def save_image(self, folder_name='images', folder_path=''):
        """Saves image in output folder"""
        #print('saving image')
        output_path = os.path.join(folder_path, folder_name)
        if not os.path.exists(output_path):
            os.mkdir(folder_path)

        self.load_image_reqs()
        req = requests.get(self.__image_url__)

        output_path_name = os.path.join(output_path, self.__image_name__)

        with open(output_path_name, 'wb') as out:
            out.write(req.content)

    def save_landmark_points(self, folder_path='', folder_name='pts', menpo=True):
        """Saves ground truth landmark points in folder_path/image_name.pts"""
        #print('saving pts')
        output_path = os.path.join(folder_path, folder_name)
        if not os.path.exists(output_path):
            os.mkdir(folder_path)
        self.load_pts_reqs()

        output_path_name = os.path.join(output_path, self.__pts_name__)
        points = self.get_landmark_points()
        #print(points.shape)
        with open(output_path_name, 'w') as out:
            write_pts(out, points, menpo)

