import os
import subprocess
import sys

# Fonction pour exécuter la détection
def run_detection(source_dir, weights='yolov5s.pt', img_size=640, conf_thres=0.25, output_dir="../Données/ birds_dataset/labels", save_crop=True, save_txt=True):
    """
    Exécute la détection d'objets avec YOLOv5 sur un répertoire donné.

    :param source_dir: Chemin du répertoire contenant les images
    :param weights: Poids du modèle YOLOv5 (par défaut: yolov5s.pt)
    :param img_size: Taille des images redimensionnées (par défaut: 640)
    :param conf_thres: Seuil de confiance (par défaut: 0.25)
    :param output_dir: Répertoire pour sauvegarder les résultats (par défaut: runs/detect)
    :param save_crop: Si True, enregistre les objets détectés en tant qu'images croppées
    :param save_txt: Si True, enregistre les résultats des objets détectés dans un fichier .txt
    """
    # Assurez-vous que le chemin du répertoire source existe
    if not os.path.exists(source_dir):
        print(f"Le répertoire source {source_dir} n'existe pas.")
        sys.exit(1)

    # Vérifier si le répertoire de sortie existe, sinon le créer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Le répertoire de sortie {output_dir} a été créé.")

    # Chemin absolu vers detect.py
    detect_script = r"C:\Users\alvin\Documents\M1 MIASHS\S1\TER\yolov5\detect.py"

    # Commande pour exécuter detect.py avec les options spécifiées
    command = [
        'python', detect_script,  # Chemin complet vers detect.py
        '--source', source_dir,  # Répertoire contenant les images
        '--weights', weights,  # Poids du modèle YOLOv5
        '--img', str(img_size),  # Taille des images
        '--conf', str(conf_thres),  # Seuil de confiance
        '--project', output_dir,  # Répertoire pour sauvegarder les résultats
        '--name', 'exp',  # Nom de l'expérience
        '--exist-ok'  # Permet d'écraser les résultats existants
    ]

    # Ajouter l'option --save-crop si demandée
    if save_crop:
        command.append('--save-crop')

    # Ajouter l'option --save-txt si demandée
    if save_txt:
        command.append('--save-txt')

    # Exécution de la commande
    try:
        subprocess.run(command, check=True)
        print(f"Détection terminée pour les images dans {source_dir}. Les résultats sont sauvegardés dans {output_dir}.")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de la détection : {e}")

# Parcourir tous les sous-dossiers dans 'birds_dataset' et exécuter la détection pour chaque dossier
if __name__ == "__main__":
    # Le chemin du dossier parent contenant tous les sous-dossiers d'espèces
    parent_directory = r'../Données/ birds_dataset'  # Le répertoire qui contient tous les sous-dossiers pour chaque espèce

    # Vérifier si le dossier parent existe
    if not os.path.exists(parent_directory):
        print(f"Le répertoire parent {parent_directory} n'existe pas.")
        sys.exit(1)

    # Parcourir tous les sous-dossiers dans 'birds_dataset'
    for species_dir in os.listdir(parent_directory):
        full_species_dir = os.path.join(parent_directory, species_dir)

        # Vérifier que c'est un répertoire
        if os.path.isdir(full_species_dir):
            print(f"Détection en cours pour {species_dir}...")

            # Définir le répertoire de sortie pour chaque espèce
            output_dir_for_species = os.path.join(parent_directory, species_dir, 'labels')

            # Exécuter la détection pour ce sous-dossier (espèce)
            run_detection(full_species_dir, output_dir=output_dir_for_species)
