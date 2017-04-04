"""
this script is use landmark to do similiar transform for detected face
"""
import math
import cv2
import numpy as np


def Distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx*dx+dy*dy)

def cal_mid(p1, p2):
    m_x = (p2[0] + p1[0]) / 2.0
    m_y = (p2[1] + p1[1]) / 2.0
    return m_x, m_y

def rotateImage(image, angle, center=None, scale=1.0):
    h, w = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    result = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_CUBIC)
    return result


def crop_simi(img, eye_left, eye_right,
              extend=0.3, dest_sz=(96, 96)):
    # calculate offsets in output image
    offset_w = math.floor(float(extend)*dest_sz[0])  # w means width
    offset_h = math.floor(float(extend)*dest_sz[1])  # h means height
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians,the horizental anggle
    rotation = math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    rotation = rotation * 180 / np.pi
    # distance between them
    dist = Distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0*offset_w
    # scale factor
    # 1/scale is how the final image change
    scale = float(dist)/float(reference)  # the length between two eye is fixed
    # rotate original around the left eye
    # just rotate the image without scale
    im_height, im_width = img.shape[:2]
    img = rotateImage(img, center=tuple(eye_left), angle=rotation)
    # crop the rotated image
    # the coodinate of left eye is fixed

    left_x = int(max(eye_left[0] - scale*offset_w, 0))
    up_y = int(max(eye_left[1] - scale*offset_h, 0))
    right_x = int(min(left_x + dest_sz[0]*scale, im_width))
    down_y = int(min(up_y + dest_sz[1]*scale, im_height))

    crop_image = img[up_y:down_y, left_x:right_x]
    crop_image = cv2.resize(crop_image, dest_sz, interpolation=cv2.INTER_CUBIC)
    return crop_image


def crop_rotate(img, eye_left, eye_right, bb_width,
                extend=0.33, dest_sz=(96, 112)):
    """
    this function is used to do similarity transformation,
    put the eye into horizontal direction
    and make the eye in the same position
    """
    # calculate mid_point of two eyes
    mid_eye = cal_mid(eye_left, eye_right)
    # calculate the rotation angel
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians,the horizental anggle
    rotation = math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    rotation = rotation * 180 / np.pi
    # scale factor 1/scale is how the final image change
    im_height, im_width = img.shape[:2]
    scale = float(bb_width*1.02) / float(dest_sz[0]) # the width of bbox keeps unchange
    # rotate original around the left eye, rotate the image without scale
    img = rotateImage(img, center=tuple(eye_left), angle=rotation)
    # calculate aspect ratio
    a_ratio = 1.0 * dest_sz[1] / dest_sz[0]
    # crop the rotated image
    # the coodinate of left eye is fixed
    left_x = int(max(mid_eye[0] - bb_width / 2.0, 0))
    up_y = int(max(mid_eye[1] - bb_width * a_ratio * extend, 0))
    right_x = int(min(left_x + dest_sz[0]*scale, im_width))
    down_y = int(min(up_y + dest_sz[1]*scale, im_height))

    crop_image = img[up_y:down_y, left_x:right_x]
    crop_image = cv2.resize(crop_image, dest_sz, interpolation=cv2.INTER_CUBIC)
    return crop_image

def crop_only(img, bbox, extend=0.1, dest_sz=(160, 160)):
    left_x, up_y, right_x, down_y = bbox
    im_height, im_width = img.shape[:2]
    center_x = (left_x + right_x) / 2
    center_y = (up_y + down_y) / 2
    height = down_y - up_y
    width = right_x - left_x
    left_x = int(max(center_x - round((1+extend)*width/2), 0))
    up_y = int(max(center_y - round((1+extend)*height/2), 0))
    right_x = int(min(center_x + round((1+extend)*width/2), im_width))
    down_y = int(min(center_y + round((1+extend)*height/2), im_height))

    crop_image = img[up_y:down_y, left_x:right_x]
    crop_image = cv2.resize(crop_image, dest_sz, interpolation=cv2.INTER_CUBIC)
    return crop_image

if __name__ == '__main__':
    from detector import Detector
    model_path = './model/mtcnn'
    im = cv2.imread('test.jpg')
    detector = Detector(model_path, gpu_fraction=0.5)
    results = detector.detect_face(im, debug=True)
    for idx, result in enumerate(results):
        crop_im = crop_only(im, result['bbox'], extend=0, dest_sz=(160, 160))
        cv2.imwrite('crop_only_{}.jpg'.format(idx), crop_im)
        crop_im = crop_simi(im, result['left_eye'], result['right_eye'], extend=0.32, dest_sz=(160, 160))
        cv2.imwrite('crop_simi_{}.jpg'.format(idx), crop_im)
        crop_im = crop_rotate(im, result['left_eye'], result['right_eye'], result['width'])
        cv2.imwrite('crop_rotate_{}.jpg'.format(idx), crop_im)



