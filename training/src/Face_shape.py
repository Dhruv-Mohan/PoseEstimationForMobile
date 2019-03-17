import tensorflow as tf
import cv2
import math
import numpy as np
import os
import shutil

INPUT_WIDTH = 256
INPUT_HEIGHT = 256


class Face_classifier(object):
    def __init__(self, frozen_graph="./4.pb", output_node_names="output"):
        with tf.gfile.GFile(frozen_graph, "rb") as f:
            restored_graph_def = tf.GraphDef()
            restored_graph_def.ParseFromString(f.read())

        tf.import_graph_def(
            restored_graph_def,
            input_map=None,
            return_elements=None,
            name=""
        )

        self.graph = tf.get_default_graph()
        self.image = self.graph.get_tensor_by_name("image:0")
        self.output = self.graph.get_tensor_by_name("%s:0" % output_node_names)
        #self.em = cv2.ml.EM_load(face_classifier)
        self.sess = tf.Session()
        self.shapes = ["SQUARE", "DIAMOND", "ROUND", "LONG", "OVAL"]
        self.lip_landmarks = [0, 16, 2, 3, 4 ,10, 9, 8, 14, 18]
        self.eye_brow_landmarks = [70, 69, 68, 67, 66, 62]

    @staticmethod
    def get_locs_from_hmap(part_map_resized):
        return (np.unravel_index(part_map_resized.argmax(), part_map_resized.shape))


    @staticmethod
    def soft_argmax(patch_t, coord):
        patch = cv2.getRectSubPix(patch_t, (9, 9), (coord[1], coord[0]))
        patch_sum = np.sum(patch)
        x = np.linspace(-4, 4, 9)
        x = np.expand_dims(x, -1)
        y = np.transpose(x)
        x_pos = ((np.sum(patch * y) / patch_sum) + coord[1]) * 16 + 6.5
        y_pos = ((np.sum(patch * x) / patch_sum) + coord[0]) * 16 + 6.5
        return x_pos, y_pos
        pass

    @staticmethod
    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))

    @staticmethod
    def process_landmarks(pts):
        pts2 = pts[20:53, :]
        pts1 = np.expand_dims(pts[9, :], 0)
        pts3 = np.expand_dims(pts[55, :], 0)
        pts4 = np.expand_dims(pts[64, :], 0)
        pts = np.concatenate([pts1, pts2, pts3, pts4], axis=0)
        return pts

    @staticmethod
    def get_slope(p1, p2):

        a = (p1[1] - p2[1])
        b = (p1[0] - p2[0])
        c = b*p1[1] - a*p1[0]
        y = p1[1] - p2[1]
        return -a, b, -c, y

    @staticmethod
    def get_distance(pt, a, b, c):
        d = (a*pt[0] + b*pt[1] + c) / math.sqrt(b*b + a*a)
        return d

    @staticmethod
    def get_angle(a, b):
        angle = math.atan(-a/b) * 180/math.pi
        if angle > 180:
            angle -= 360

        return angle

    def get_ratio(self, landmarks):
        face_length = self.dist(landmarks[98], landmarks[36])
        face_width = self.dist(landmarks[27], landmarks[46])
        return face_length / face_width

    def get_pts(self, heatmaps):
        coords_argmax = []
        for i in range(100):
            if i == 20:
                continue
            pt_32 = self.get_locs_from_hmap(heatmaps[:, :, i])
            pt_argmax = self.soft_argmax(heatmaps[:, :, i], pt_32)
            coords_argmax.append(pt_argmax)
        return coords_argmax

    def predict_shape(self, image_path):
        eye_brow=[]
        lip = []
        eye = []
        image_0 = cv2.imread(image_path)
        #image_2 = cv2.imread(image_path)
        #w, h, _ = image_2.shape
        image_ = cv2.resize(image_0, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        image_ = image_.astype(np.float32)
        heatmaps = self.sess.run(self.output, feed_dict={self.image: [image_]})
        heatmaps = np.squeeze(heatmaps)
        coords_argmax = self.get_pts(heatmaps)
        slopes = self.get_slopes(coords_argmax)
        slopes = np.float16(slopes)
        chin = slopes[11] + slopes[12] + slopes[13]
        jaws = slopes[3] + slopes[4] + slopes[5] + slopes[18] + slopes[19] + slopes[20]

        if chin < 24:
            jaw_class = 0
        elif jaws <55:
            jaw_class = 2
        else:
            jaw_class = 1
        '''
        for index, i in enumerate(range(24, 48)):
            pt = coords_argmax[i]
            slope= int(slopes[index])
            slope = str(slope)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_2, slope, (int(pt[0]* h/512), int(pt[1]*w/512)), font, 0.5, (200, 0, 0), 1, cv2.LINE_AA)
        for pt in coords_argmax:
            cv2.circle(image_2, (int(pt[0] * h/512), int(pt[1] *w/512)), 5, (0,0,255))

        cv2.imshow('cra', image_2)
        '''
        self.eye_width = self.dist(coords_argmax[88], coords_argmax[92])
        eye = self.get_eye_shape(coords_argmax)
        lip = self.get_lip_shape(coords_argmax)
        eye_brow = self.get_eye_brow_shape(coords_argmax)
        ratio = self.get_ratio(coords_argmax)


        if ratio > 1.45:
            return self.shapes[3], eye_brow, lip, eye

        if ratio > 1.2 and ratio < 1.45 and jaw_class == 2:
            return self.shapes[4], eye_brow, lip, eye

        return self.shapes[jaw_class], eye_brow, lip, eye


    def get_jaw_shape(self, landmarks):
        landmarks=np.asarray(landmarks) / 512
        landmarks = self.process_landmarks(landmarks)
        landmarks = np.asarray(landmarks)
        landmarks = np.reshape(landmarks, [-1,1])
        _, i = self.em.predict2(landmarks)[0]
        return int(i)

    def get_eye_brow_shape(self, landmarks):
        eye_brow_pts = []
        for item in self.eye_brow_landmarks:
            eye_brow_pts.append(landmarks[item])
        #Getting relevant params
        a, b, c, y_dst = self.get_slope(eye_brow_pts[0], eye_brow_pts[4])
        dist = self.get_distance(eye_brow_pts[3], a, b, c)
        dist_68 = self.get_distance(eye_brow_pts[2], a, b, c)

        shape = 'CURVED'
        if dist - dist_68 < -1:
            shape="CURVED"
        elif dist > 10:
            shape = 'HIGH'
        elif dist < 10 and dist > 6:
            shape = 'SOFT'
        elif y_dst < 10:
            shape = "STRAIGHT"
        else:
            shape = 'UPWARD'

        thickness = self.dist(landmarks[68], landmarks[63])
        length = self.dist(landmarks[70], landmarks[66])


        if length/self.eye_width > 1.6:
            len_type = "LONG"
        elif length/self.eye_width < 1.3:
            len_type = "SHORT"
        else:
            len_type = "REGULAR"

        if thickness/self.eye_width > .48:
            AC_data = "THICK"
        elif thickness/self.eye_width < 0.4:
            AC_data = "THIN"
        else:
            AC_data = 'REGULAR'

        return shape, len_type, AC_data
        pass

    def get_lip_shape(self, landmarks):
        Upper_lip_dst = self.dist(landmarks[2+1], landmarks[13+1])
        Lower_lip_dst = self.dist(landmarks[9], landmarks[18])
        Lip_Width = self.dist(landmarks[0], landmarks[16])
        ratio = Upper_lip_dst / Lower_lip_dst


        if ratio > 0.7:
            shape="TOP HEAVY"
        elif ratio <0.45:
            shape = "BOTTOM HEAVY"
        else:
            if (Upper_lip_dst + Lower_lip_dst)/self.eye_width > 0.54:
                shape = "FULL"
            elif (Upper_lip_dst + Lower_lip_dst)/self.eye_width < 0.39:
                shape = "THIN"
            else:
                shape = 'REGULAR'

        if Lip_Width/ self.eye_width > 1.8:
            lip_len = "LONG"
        elif Lip_Width/ self.eye_width < 1.5:
            lip_len = "SHORT"
        else:
            lip_len = "REGULAR"

        return shape, lip_len

        pass

    def get_eye_shape(self, landmarks):
        Eye_width = self.dist(landmarks[88], landmarks[92])
        Eye_height = self.dist(landmarks[90], landmarks[94])
        Face_width = self.dist(landmarks[27], landmarks[46])
        IPD = self.dist(landmarks[88], landmarks[80])
        a, b, c, y_dst = self.get_slope([landmarks[88][0], -landmarks[88][1]], [landmarks[92][0], -landmarks[92][1]])
        base_a, base_b, _, __ = self.get_slope([landmarks[84][0], -landmarks[84][1]], [landmarks[88][0], -landmarks[88][1]])
        base_angle = self.get_angle(base_a,base_b)
        eye_angle = self.get_angle(a,b)
        ratio = Eye_width/Eye_height


        if ratio >2.6:
            shape='ALMOND'
        else:
            shape='ROUND'

        if eye_angle-base_angle > 3.5:
            turn_shape = 'UPTURNED'
        elif eye_angle-base_angle < -2.6:
            turn_shape = 'DOWNTURN'
        else:
            turn_shape = 'REGULAR'

        if IPD/Face_width > 0.535:
            set_shape = 'WIDESET'
        elif IPD/Face_width < 0.495:
            set_shape = 'CLOSESET'
        else:
            set_shape = 'REGULAR'

        return shape, turn_shape, set_shape

    def get_slopes(self, landmarks):
        slopes = []
        for i in range(24,48):
            a,b,c,y = self.get_slope(landmarks[i], landmarks[i+1])
            slopes.append(math.atan(a/b) * 180/math.pi)

        slopes = np.asarray(slopes)
        slopes = np.gradient(slopes)
        return(slopes)





Net = Face_classifier()

images = os.listdir('/home/dhruv/Projects/Datasets/Groomyfy_27k/Source/Menpo512_25/hair')
for ima in images:
    image_full_path = os.path.join('/home/dhruv/Projects/Datasets/Groomyfy_27k/Source/Menpo512_25/hair'
                                   , ima)
    image_output_path = '/home/dhruv/Projects/Datasets/Groomyfy_27k/Source/Menpo512_25/shapes/'
    fsh = Net.predict_shape(image_full_path) #Face shape props here

    print('FACE:(\'',fsh[0], '\') EYEBROW:', fsh[1], ' LIP:',fsh[2],' EYE:', fsh[3])
    image_output_path_full = image_output_path + ima
    shutil.copy(image_full_path, image_output_path_full)
    im_crap = cv2.imread(image_full_path)
    cv2.imshow('image', im_crap)
    cv2.waitKey(0)