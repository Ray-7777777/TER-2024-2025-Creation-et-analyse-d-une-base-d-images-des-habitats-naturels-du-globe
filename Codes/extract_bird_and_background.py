import os
import sys
import cv2
import torch
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'yolov5'))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.datasets import LoadImages
from utils.torch_utils import select_device

def extract_bird_and_background(weights='yolov5s.pt', source='../yolov5/runs/detect/exp4', output_dir='output', img_size=640, conf_thres=0.25):
    device = select_device('')  # GPU ou CPU
    model = DetectMultiBackend(weights, device=device)
    model.eval()

    dataset = LoadImages(source, img_size=img_size)  # Chargement des images
    os.makedirs(output_dir, exist_ok=True)

    # Parcourir les images du dossier
    for path, img, im0s, _ in dataset:
        img_tensor = torch.from_numpy(img).to(device).float()
        img_tensor /= 255.0  # Normalisation entre 0 et 1
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Détection
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=0.45)

        for det in pred:  # Pour chaque détection
            if len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0s.shape).round()

                bird_img = im0s.copy()  # Image contenant uniquement l'oiseau
                background_img = im0s.copy()  # Image sans l'oiseau

                # Pour chaque détection (xyxy), on isole la région de l'oiseau
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)  # Coordonnées de l'oiseau détecté

                    # Extraire l'oiseau
                    bird = im0s[y1:y2, x1:x2]
                    bird_path = os.path.join(output_dir, f"bird_{Path(path).stem}.jpg")
                    cv2.imwrite(bird_path, bird)  # Enregistrer l'oiseau extrait

                    # Supprimer l'oiseau de l'image d'origine (en remplissant avec un fond noir)
                    background_img[y1:y2, x1:x2] = [0, 0, 0]  # Masque noir sur l'oiseau

                # Enregistrer l'image sans les oiseaux
                background_path = os.path.join(output_dir, f"background_{Path(path).stem}.jpg")
                cv2.imwrite(background_path, background_img)

                print(f"Images enregistrées pour {Path(path).name} dans {output_dir}")

if __name__ == '__main__':
    extract_bird_and_background(weights='yolov5s.pt', source='runs/detect/exp', output_dir='output')
