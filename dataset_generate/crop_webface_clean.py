import os
import cv2
from preprocess.detector import Detector
from preprocess.crop_util import crop_rotate
from preprocess.naive_dlib import NaiveDlib


rootdir = '/media/teddy/data/casia-maxpy-clean'
savedir = '/media/teddy/data/casia_crop'
model_path = '../preprocess/model/mtcnn'
dlib_model_path = '../preprocess/model/dlib/shape_predictor_68_face_landmarks.dat'
id_label_dic = {}
label_count_dic = {}
label_count = 0
all_image_count = 0
detector = Detector(model_path, gpu_fraction=1.0)
detector_dlib = NaiveDlib(dlib_model_path)

if not os.path.exists(savedir):
    os.mkdir(savedir)

def preprocess(img):
    results_0 = detector.detect_face(img)
    if len(results_0) != 0:
        result_0 = max(results_0, key=lambda r: r['area'])
    else:
        result_0 = None
    result_1 = detector_dlib.getLargestFaceBoundingBox(img)

    if result_0 is not None and result_1 is not None:
        if result_1.area() * 0.6 > result_0['area']:
            crop_img = detector_dlib.prepocessImg(img, result_1)
        else:
            crop_img = crop_rotate(img, result_0['left_eye'], result_0['right_eye'], result_0['width'])
    elif result_0 is not None and result_1 is None:
        crop_img = crop_rotate(img, result_0['left_eye'], result_0['right_eye'], result_0['width'])
    elif result_0 is None and result_1 is not None:
        crop_img = detector_dlib.prepocessImg(img, result_1)
    else:
        crop_img = img[70:210, 65:185, :]
        crop_img = cv2.resize(crop_img, (96, 112), interpolation=cv2.INTER_CUBIC)
    return crop_img


for parent, _, filenames in os.walk(rootdir, topdown=False):
    id_name = parent.rsplit('/', 1)[-1].strip()
    img_filenames = [i for i in filenames if i.find('.jpg')]
    if len(img_filenames) > 0:
        if id_name not in id_label_dic:
            id_label_dic[id_name] = label_count
            print('process the {} id'.format(label_count))
            im_count = 0
            os.mkdir(os.path.join(savedir, str(label_count)))
            for img_name in img_filenames:
                img_path = os.path.join(parent, img_name)
                im_arr = cv2.imread(img_path)
                crop_img = preprocess(im_arr)
                save_path = os.path.join(savedir, str(label_count), img_name)
                cv2.imwrite(save_path, crop_img)
                im_count += 1
            all_image_count += im_count
            label_count_dic[label_count] = im_count
            label_count += 1
        else:
            raise Exception("repeat id name")

txt_name = 'casia_disc.txt'
with open(txt_name, 'w') as f:
    in_str = 'image_num ' + str(all_image_count) + '\n'
    f.write(in_str)
    in_str = 'id_num ' + str(len(id_label_dic)) + '\n'
    f.write(in_str)
    in_str = 'id_name label img_num\n'
    f.write(in_str)
    for key in id_label_dic:
        in_str = str(key) + ' ' + str(id_label_dic[key]) + ' ' + str(label_count_dic[id_label_dic[key]]) + '\n'
        f.write(in_str)





