#codes à mettre dans votre yolov5 cloner dans utils: yolov5/utils

import os
import cv2
import numpy as np
import torch
from pathlib import Path

class LoadImages:
    def __init__(self, path, img_size=640, auto=True):
        self.path = path
        self.img_size = img_size
        self.auto = auto

        #Les fichiers sont en .jpg et .jpeg
        self.files = sorted(Path(path).rglob('*.jpg')) + sorted(Path(path).rglob('*.jpeg'))
        self.nf = len(self.files)  # Nombre total de fichiers
        self.index = 0  # Initialiser l'index

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.nf:  # Si tous les fichiers ont été lus
            raise StopIteration  # Terminer l'itération

        # Charger l'image
        img_path = self.files[self.index]
        img0 = cv2.imread(str(img_path))  # Lire l'image
        self.index += 1  # Incrémenter l'index

        # Vérification de la dimension de l'image
        if img0 is None:
            print(f"Erreur de chargement de l'image: {img_path}")
            return None, None, None, None

        # Redimensionnement de l'image
        img = cv2.resize(img0, (self.img_size, self.img_size))  # Redimensionnement à img_size
        img = img / 255.0  # Normaliser les pixels entre 0 et 1
        
        # **Modification : Créer une copie contiguë de l'array**
        img = np.ascontiguousarray(img.copy())  # Ajoutez .copy() pour créer une copie contiguë

        # Vérifiez la dimension de l'image
        if img.ndim == 2:  # Si l'image est 2D (niveaux de gris)
            img = np.stack((img,)*3, axis=-1)  # Convertir en image RGB

        img = img[:, :, ::-1]  # BGR à RGB
        img = img.transpose(2, 0, 1)  # 3xHxW

        return img_path, img, img0, None  # Retourner le chemin, image normalisée, image originale et None

# Autres fonctions pour le dataset (si nécessaire)
