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




import os
import cv2
import shutil
import numpy as np
import logging

# Configurer le logging pour afficher dans un fichier
logging.basicConfig(filename='image_classification.log', level=logging.INFO, format='%(asctime)s - %(message)s')


def calculate_entropy(image_path):
    ""
    Calcule l'entropie d'une image en niveau de gris.

    :param image_path: Chemin vers l'image à analyser.
    :return: L'entropie de l'image.
    ""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logging.error(f"Impossible de charger l'image {image_path}")
        return 0

    # Calculer l'histogramme de l'image
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Normaliser l'histogramme

    # Calculer l'entropie
    entropy = -np.sum(hist * np.log2(hist + 1e-6))  # Ajouter un petit epsilon pour éviter les problèmes de log(0)

    return entropy


def is_blurry(image_path, threshold=5):
    ""
    Vérifie si une image est floue en utilisant l'entropie.

    :param image_path: Chemin vers l'image à vérifier.
    :param threshold: Seuil en dessous duquel l'image est considérée comme floue.
    :return: Booléen indiquant si l'image est floue ou non, et l'entropie de l'image.
    ""
    entropy = calculate_entropy(image_path)

    return entropy < threshold, entropy


def classify_images(parent_folder, sharp_folder, blurry_folder, threshold=5):
    ""
    Parcourt le dossier parent pour classifier les images comme floues ou nettes.

    :param parent_folder: Dossier parent contenant les images.
    :param sharp_folder: Dossier où sauvegarder les images nettes.
    :param blurry_folder: Dossier où sauvegarder les images floues.
    :param threshold: Seuil d'entropie pour déterminer si l'image est floue.
    ""
    os.makedirs(sharp_folder, exist_ok=True)
    os.makedirs(blurry_folder, exist_ok=True)

    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                blurry, entropy = is_blurry(image_path, threshold)

                # Enregistrer les résultats dans le fichier de log et déplacer les images
                if blurry:
                    logging.info(f"[FLU] {image_path} - Entropie: {entropy:.2f}")
                    shutil.copy(image_path, blurry_folder)  # Copier l'image floue
                else:
                    logging.info(f"[NETTE] {image_path} - Entropie: {entropy:.2f}")
                    shutil.copy(image_path, sharp_folder)  # Copier l'image nette


# Exécuter la classification d'images
parent_folder = r"../Donnees/birds_dataset"  # Chemin vers le dossier contenant les images
sharp_folder = r"../Donnees/sharp_images"  # Dossier pour les images nettes
blurry_folder = r"../Donnees/blurry_images"  # Dossier pour les images floues

classify_images(parent_folder, sharp_folder, blurry_folder, threshold=5)
"""


import cv2
import numpy as np
import os
import shutil

def variance_of_laplacian(image):
    """
    Calcule la variance du Laplacien pour détecter la netteté de l'image.
    Une faible variance signifie que l'image est floue.
    """
    # Appliquer l'opérateur de Laplace et calculer la variance
    return cv2.Laplacian(image, cv2.CV_64F).var()

def calculate_fft(image):
    """
    Applique la transformée de Fourier pour mesurer le flou à partir du spectre de fréquence.
    Une image floue aura un spectre de fréquence plus faible.
    """
    # Appliquer la transformée de Fourier
    fft_image = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_image)  # Déplacer les fréquences basses au centre
    magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)  # Calculer le spectre de magnitude
    blur_metric = np.mean(magnitude_spectrum)  # Moyenne des magnitudes
    return blur_metric

def is_blurry(image_path, threshold_laplacian=20, threshold_fft=3):
    """
    Détecte si une image est floue en utilisant les méthodes Laplacien et FFT.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Impossible de charger l'image {image_path}")
        return False, 0, 0

    # Calcul de la variance du Laplacien
    laplacian_variance = variance_of_laplacian(image)

    # Calcul de la métrique de flou FFT
    fft_blur_metric = calculate_fft(image)

    # Vérification du flou basé sur les seuils
    is_blurry_laplacian = laplacian_variance < threshold_laplacian
    is_blurry_fft = fft_blur_metric < threshold_fft

    return is_blurry_laplacian or is_blurry_fft, laplacian_variance, fft_blur_metric


def add_text_with_black_background(image, laplacian_variance, fft_blur_metric, position=(10, 30), font_scale=0.5,
                                   thickness=1):
    """
    Ajoute le score de flou (variance du Laplacien et métrique FFT) sur l'image.
    """
    text = f'Laplacian: {laplacian_variance:.2f}, FFT: {fft_blur_metric:.2f}'

    # Assurer que 'thickness' est un entier
    thickness = int(thickness)

    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(image, (position[0], position[1] - text_height - 10),
                  (position[0] + text_width + 5, position[1] + 10), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, text, (position[0], position[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                thickness)


def classify_images(parent_folder, sharp_folder, blurry_folder, threshold_laplacian=30, threshold_fft=5):
    """
    Parcourt un dossier d'images, et classe les images comme floues ou nettes en fonction des seuils.
    La structure des dossiers inclut des sous-dossiers pour chaque espèce sous `nom_espèce/train`.
    """
    os.makedirs(sharp_folder, exist_ok=True)
    os.makedirs(blurry_folder, exist_ok=True)

    # Parcourir chaque espèce dans le dossier parent
    for species_folder in os.listdir(parent_folder):
        species_path = os.path.join(parent_folder, species_folder, 'train')  # Chemin vers le dossier `train` de chaque espèce
        if os.path.isdir(species_path):
            for root, _, files in os.walk(species_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(root, file)
                        blurry, laplacian_variance, fft_blur_metric = is_blurry(image_path, threshold_laplacian, threshold_fft)

                        # Charger l'image pour l'annoter et afficher les résultats
                        image = cv2.imread(image_path)

                        # Ajouter le texte avec les mesures de flou
                        add_text_with_black_background(image, laplacian_variance, fft_blur_metric)

                        # Afficher les résultats et déplacer l'image
                        if blurry:
                            print(f"[FLU] {image_path} - Laplacian Variance: {laplacian_variance:.2f}, FFT Metric: {fft_blur_metric:.2f}")
                            shutil.copy(image_path, blurry_folder)  # Copier l'image floue
                        else:
                            print(f"[NETTE] {image_path} - Laplacian Variance: {laplacian_variance:.2f}, FFT Metric: {fft_blur_metric:.2f}")
                            shutil.copy(image_path, sharp_folder)  # Copier l'image nette

                        # Optionnel : Afficher l'image avec les annotations
                        cv2.imshow('Image with Blur Metrics', image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

# Exécution de la classification d'images
parent_folder = r"..\Donnees\birds_dataset2"  # Dossier contenant les images
sharp_folder = r"..\Donnees\sharp_images"  # Dossier pour les images nettes
blurry_folder = r"..\Donnees\blurry_images"  # Dossier pour les images floues

classify_images(parent_folder, sharp_folder, blurry_folder, threshold_laplacian=15, threshold_fft=7)
