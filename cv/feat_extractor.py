import cv2
import numpy as np
import torch
from PIL import Image
from mtcnn.mtcnn import MTCNN
from torchvision.transforms import transforms

from cv.net_sphere import SphereFaceNet


def detect_face(img_path):
    """
    detect face with MTCNN
    :param img_path:
    :return:
    """
    img = cv2.imread(img_path)
    detector = MTCNN()
    mtcnn_result = detector.detect_faces(img)

    return mtcnn_result


def ext_feats(sphere_face, img_path, pretrained_model='model/sphere20a.pth'):
    if sphere_face is None:
        sphere_face = SphereFaceNet()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sphere_face.load_state_dict(torch.load(pretrained_model))
    sphere_face = sphere_face.to(device)

    img = cv2.imread(img_path)
    mtcnn_result = detect_face(img_path)

    print(mtcnn_result)

    if len(mtcnn_result) > 0:
        bbox = mtcnn_result[0]['box']

        margin_pixel = 10
        face_region = img[bbox[0] - margin_pixel: bbox[0] + bbox[2] + margin_pixel,
                      bbox[1] - margin_pixel: bbox[1] + bbox[3] + margin_pixel]

        ratio = max(face_region.shape[0], face_region.shape[1]) / min(face_region.shape[0], face_region.shape[1])
        if face_region.shape[0] < face_region.shape[1]:
            face_region = cv2.resize(face_region, (int(ratio * 64), 64))
            face_region = face_region[:,
                          int((face_region.shape[0] - 64) / 2): int((face_region.shape[0] - 64) / 2) + 64]
        else:
            face_region = cv2.resize(face_region, (64, int(ratio * 64)))
            face_region = face_region[int((face_region.shape[1] - 64) / 2): int((face_region.shape[1] - 64) / 2) +
                                                                            64, :]

        face_region = Image.fromarray(face_region.astype(np.uint8))
        preprocess = transforms.Compose([
            transforms.Resize((96, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        face_region = preprocess(face_region)
        face_region.unsqueeze_(0)
        face_region = face_region.to(device)

        x = face_region.to(device)
        x = sphere_face.forward(x)

        return {
            'status': 0,
            'message': 'extracted feature',
            'feature': x.to("cpu").detach().numpy().flatten()
        }

    else:
        return {
            'status': 0,
            'message': 'No face detected!',
            'feature': None
        }


if __name__ == "__main__":
    print(ext_feats(sphere_face=SphereFaceNet(feature=True), img_path="C:/Users/29140/Desktop/SCUT-FBP-1.jpg"))
