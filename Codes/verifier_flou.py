import cv2
import os
import shutil


def is_blurry(image_path, threshold=100):
    """
    Vérifie si une image est floue en utilisant la variance de la Laplacienne.

    :param image_path: Chemin vers l'image à vérifier.
    :param threshold: Seuil en dessous duquel l'image est considérée comme floue.
    :return: Booléen indiquant si l'image est floue ou non, et la variance de la Laplacienne.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Impossible de charger l'image {image_path}")
        return False, 0

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()

    return variance < threshold, variance


def check_and_copy_net_images(parent_folder, output_folder, threshold=100):
    """
    Parcourt les sous-dossiers d'un dossier parent pour vérifier si les images sont floues,
    et copie les images nettes dans un nouveau dossier en conservant la structure de dossiers.

    :param parent_folder: Chemin vers le dossier parent contenant les sous-dossiers d'images.
    :param output_folder: Chemin vers le dossier de sortie pour les images nettes.
    :param threshold: Seuil de flou.
    """
    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                blurry, variance = is_blurry(image_path, threshold)

                if not blurry:
                    # Créer le chemin de destination correspondant dans le dossier de sortie
                    relative_path = os.path.relpath(root, parent_folder)
                    output_dir = os.path.join(output_folder, relative_path)
                    os.makedirs(output_dir, exist_ok=True)

                    # Copier l'image nette vers le dossier de sortie
                    output_image_path = os.path.join(output_dir, file)  #pour reproduire la même structure de sous dossiers
                    shutil.copy2(image_path, output_image_path)
                    print(f"[COPIÉE] {image_path} -> {output_image_path} - Variance: {variance:.2f}")
                else:
                    print(f"[FLOUE] {image_path} - Variance: {variance:.2f}")


# Dossier d'origine et dossier de sortie
parent_folder = "../Donnees/ birds_dataset"
output_folder = "../Donnees/data_nettoye"

# Exécuter la fonction
check_and_copy_net_images(parent_folder, output_folder, threshold=100)



