import os
import random
import shutil
import yaml

def split_dataset(images_folder, labels_folder, train_folder, val_folder, val_split=0.2):
    # Crée les dossiers train et val pour les images et les labels
    os.makedirs(os.path.join(train_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_folder, 'labels'), exist_ok=True)

    # Liste les fichiers d'images dans le dossier 'images'
    images = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Mélange les images pour la séparation aléatoire
    random.shuffle(images)

    # Calcul du nombre d'images pour l'ensemble de validation
    val_count = int(len(images) * val_split)
    train_images = images[val_count:]
    val_images = images[:val_count]

    # Déplace les fichiers images et labels dans train et val
    for image_file in train_images:
        label_file = os.path.splitext(image_file)[0] + '.txt'
        shutil.move(os.path.join(images_folder, image_file), os.path.join(train_folder, 'images', image_file))
        shutil.move(os.path.join(labels_folder, label_file), os.path.join(train_folder, 'labels', label_file))

    for image_file in val_images:
        label_file = os.path.splitext(image_file)[0] + '.txt'
        shutil.move(os.path.join(images_folder, image_file), os.path.join(val_folder, 'images', image_file))
        shutil.move(os.path.join(labels_folder, label_file), os.path.join(val_folder, 'labels', label_file))

def create_yaml_file(train_folder, val_folder, yaml_file, nc):
    # Crée le fichier YAML pour YOLO
    data = {
        'train': os.path.abspath(os.path.join(train_folder, 'images')),
        'val': os.path.abspath(os.path.join(val_folder, 'images')),
        'nc': nc,  # Nombre de classes
        'names': []  # Liste des noms des classes à remplir
    }

    with open(yaml_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

if __name__ == "__main__":
    images_folder = 'images'  # Dossier des images
    labels_folder = 'labels'  # Dossier des labels
    train_folder = 'train'  # Dossier de sortie pour l'ensemble train
    val_folder = 'val'  # Dossier de sortie pour l'ensemble val
    yaml_file = 'data.yaml'  # Fichier YAML à créer
    nc = 1  # Nombre de classes (à ajuster en fonction de tes classes)

    # Sépare le dataset en train et val
    split_dataset(images_folder, labels_folder, train_folder, val_folder, val_split=0.2)
    
    # Crée le fichier YAML
    create_yaml_file(train_folder, val_folder, yaml_file, nc)

