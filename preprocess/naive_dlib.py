import dlib
import numpy as np
from preprocess.crop_util import crop_rotate

class NaiveDlib:

    def __init__(self, facePredictor = None):
        """Initialize the dlib-based alignment."""
        self.detector = dlib.get_frontal_face_detector()
        if facePredictor != None:
            self.predictor = dlib.shape_predictor(facePredictor)
        else:
            self.predictor = None 

    def getAllFaceBoundingBoxes(self, img):
        faces = self.detector(np.array(img), 1)
        if len(faces) > 0:
            return faces
        else:
            return None

    def getLargestFaceBoundingBox(self, img):    #process only one face pertime
        faces = self.detector(np.array(img), 1)
        if len(faces) > 0:
            return max(faces, key=lambda rect: rect.area())
        else:
            return None


    def align(self, img, bb):
        points = self.predictor(np.array(img), bb)
        return list(map(lambda p: (p.x, p.y), points.parts()))
                
    def prepocessImg(self, img, bb, extend=0.33, dst_size=(96, 112)):
        """
        the image is load by PIL image directly
        bb is rct object of dlib
        """
        if self.predictor == None:
            raise Exception("Error: method affine should initial with an facepredictor.")
        alignPoints = self.align(img, bb)
        left_eye_l = alignPoints[36]
        left_eye_r = alignPoints[39]
        left_eye = (np.array(left_eye_l)+np.array(left_eye_r))/2
        right_eye_l = alignPoints[42]
        right_eye_r = alignPoints[45]
        right_eye = (np.array(right_eye_l)+np.array(right_eye_r))/2
        crop_img = crop_rotate(img, left_eye, right_eye, bb.width(),
                               extend=extend, dest_sz=dst_size)
        return crop_img