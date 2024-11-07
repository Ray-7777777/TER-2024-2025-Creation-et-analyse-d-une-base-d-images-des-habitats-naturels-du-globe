"""import cv2
import os
import shutil


def is_blurry(image_path, threshold=100):
    ""
    Vérifie si une image est floue en utilisant la variance de la Laplacienne.

    :param image_path: Chemin vers l'image à vérifier.
    :param threshold: Seuil en dessous duquel l'image est considérée comme floue.
    :return: Booléen indiquant si l'image est floue ou non, et la variance de la Laplacienne.
    ""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Impossible de charger l'image {image_path}")
        return False, 0

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()

    return variance < threshold, variance


def check_and_copy_net_images(parent_folder, output_folder, threshold=100):
    ""
    Parcourt les sous-dossiers d'un dossier parent pour vérifier si les images sont floues,
    et copie les images nettes dans un nouveau dossier en conservant la structure de dossiers.

    :param parent_folder: Chemin vers le dossier parent contenant les sous-dossiers d'images.
    :param output_folder: Chemin vers le dossier de sortie pour les images nettes.
    :param threshold: Seuil de flou.
    ""
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


"""














































import os
import cv2
import shutil

def is_blurry(image_path, threshold=40):
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

def classify_images(parent_folder, sharp_folder, blurry_folder, threshold=40):
    """
    Parcourt le dossier parent pour classifier les images comme floues ou nettes.

    :param parent_folder: Dossier parent contenant les images.
    :param sharp_folder: Dossier où sauvegarder les images nettes.
    :param blurry_folder: Dossier où sauvegarder les images floues.
    :param threshold: Seuil de flou.
    """
    os.makedirs(sharp_folder, exist_ok=True)
    os.makedirs(blurry_folder, exist_ok=True)

    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                blurry, variance = is_blurry(image_path, threshold)

                # Afficher les résultats et déplacer les images
                if blurry:
                    print(f"[FLU] {image_path} - Variance: {variance:.2f}")
                    shutil.copy(image_path, blurry_folder)  # Copier l'image floue
                else:
                    print(f"[NETTE] {image_path} - Variance: {variance:.2f}")
                    shutil.copy(image_path, sharp_folder)  # Copier l'image nette

# Exécuter la classification d'images
parent_folder = "../Donnees/ birds_dataset"  # Chemin vers le dossier contenant les images
sharp_folder = "../Donnees/sharp_images"  # Dossier pour les images nettes
blurry_folder = "../Donnees/blurry_images"  # Dossier pour les images floues

classify_images(parent_folder, sharp_folder, blurry_folder, threshold=9)
