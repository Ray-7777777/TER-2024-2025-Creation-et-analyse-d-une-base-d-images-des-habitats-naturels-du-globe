import sys
sys.path.append('../yolov5')  # Chemin vers yolov5

import os
import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.datasets import LoadImages
from utils.torch_utils import select_device

def generate_bird_annotations_by_species(weights='yolov5s.pt', 
                                          source='../Données/birds_dataset/images/train',  # Modifiez ce chemin si nécessaire
                                          output_dir='../Données/birds_dataset/labels/train',  # Modifiez ce chemin si nécessaire
                                          img_size=640, 
                                          conf_thres=0.25):
    device = select_device('')  # GPU ou CPU
    model = DetectMultiBackend(weights, device=device)
    model.eval()

    # Obtenir la liste des espèces (chaque sous-dossier du dossier source représente une espèce)
    species_dirs = sorted([d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))])
    
    # Créer un dictionnaire pour associer chaque espèce à un ID unique
    species_to_id = {species: idx for idx, species in enumerate(species_dirs)}
    
    # Créer les dossiers de sortie si besoin
    os.makedirs(output_dir, exist_ok=True)

    # Parcourir chaque espèce (sous-dossier)
    for species in species_dirs:
        species_path = os.path.join(source, species)  # Chemin vers le dossier de l'espèce
        species_id = species_to_id[species]  # Obtenir l'ID de la classe pour cette espèce
        species_output_dir = os.path.join(output_dir, species)
        os.makedirs(species_output_dir, exist_ok=True)

        # Chargement des images pour cette espèce
        dataset = LoadImages(species_path, img_size=img_size)  # Assurez-vous que ce chemin est correct
        
        # Parcourir les images du dossier de l'espèce
        for path, img, im0s, _ in dataset:
            img_tensor = torch.from_numpy(img).to(device).float()
            img_tensor /= 255.0  # Normalisation entre 0 et 1
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)

            # Détection des oiseaux
            pred = model(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=0.45)

            # Créer un fichier d'annotation pour chaque image
            label_file = os.path.join(species_output_dir, Path(path).stem + ".txt")  # Crée le fichier .txt pour l'image

            # Si des objets sont détectés (ici des oiseaux), on les enregistre
            with open(label_file, 'w') as f:
                for det in pred:  # Pour chaque détection
                    if len(det):
                        # Mettez à jour cette ligne si nécessaire, selon vos besoins
                        det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], im0s.shape).round()

                        # Écriture des annotations dans le fichier .txt
                        for *xyxy, conf, cls in det:
                            # Convertir de xyxy en xywh (format YOLO)
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / torch.tensor([im0s.shape[1], im0s.shape[0], im0s.shape[1], im0s.shape[0]])).view(-1).tolist()
                            # Utiliser species_id pour l'espèce
                            line = f"{species_id} {' '.join([f'{x:.6f}' for x in xywh])}\n"
                            f.write(line)

            print(f"Annotations générées pour {Path(path).name} dans {species_output_dir}")

if __name__ == '__main__':
    generate_bird_annotations_by_species(weights='yolov5s.pt', 
                                          source='../Données/birds_dataset/images/train', 
                                          output_dir='../Données/birds_dataset/labels/train')  # Modifiez ce chemin si nécessaire
