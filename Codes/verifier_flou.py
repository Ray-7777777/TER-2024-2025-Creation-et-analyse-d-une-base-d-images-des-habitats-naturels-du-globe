import cv2
import numpy as np
import os
import shutil

def sobel_variance(image):
    """
    Calcule la variance du filtre de Sobel pour détecter la netteté de l'image.
    Une faible variance signifie que l'image est floue.
    """
    # Appliquer les filtres Sobel pour détecter les bords horizontaux, verticaux et diagonaux
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient horizontal (Gx)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient vertical (Gy)

    # Filtre Sobel pour détecter les bords diagonaux (45° et 135°)
    sobel_diag_45 = cv2.filter2D(image, cv2.CV_64F, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))  # 45°
    sobel_diag_135 = cv2.filter2D(image, cv2.CV_64F, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))  # 135°

    # Calculer la magnitude des gradients dans chaque direction
    gradient_magnitude_x = cv2.magnitude(sobel_x, sobel_y)  # Magnitude des gradients horizontaux et verticaux
    gradient_magnitude_diag = cv2.magnitude(sobel_diag_45, sobel_diag_135)  # Magnitude des gradients diagonaux

    # Calculer la variance des magnitudes
    variance_x = gradient_magnitude_x.var()
    variance_diag = gradient_magnitude_diag.var()

    # Retourner la moyenne des variances pour un meilleur aperçu global
    return (variance_x + variance_diag) / 2

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

def is_blurry(image_path, threshold_sobel=100, threshold_fft=3):
    """
    Détecte si une image est floue en utilisant les méthodes Sobel et FFT.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Impossible de charger l'image {image_path}")
        return False, 0, 0

    # Convertir l'image en niveaux de gris (prétraitement pour optimiser les calculs)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calcul de la variance du filtre de Sobel
    sobel_variance_value = sobel_variance(image_gray)

    # Calcul de la métrique de flou FFT
    fft_blur_metric = calculate_fft(image_gray)

    # Vérification du flou basé sur les seuils
    is_blurry_sobel = sobel_variance_value < threshold_sobel
    is_blurry_fft = fft_blur_metric < threshold_fft

    return is_blurry_sobel and is_blurry_fft, sobel_variance_value, fft_blur_metric

def add_text_with_black_background(image, sobel_variance_value, fft_blur_metric, position=(10, 30), font_scale=1,
                                   thickness=1):
    """
    Ajoute le score de flou (variance du Sobel et métrique FFT) sur l'image.
    """
    text = f'Sobel Variance: {sobel_variance_value:.2f}, FFT: {fft_blur_metric:.2f}'

    # Assurer que 'thickness' est un entier
    thickness = int(thickness)

    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(image, (position[0], position[1] - text_height - 10),
                  (position[0] + text_width + 5, position[1] + 10), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, text, (position[0], position[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                thickness)

def classify_images(parent_folder, sharp_folder, blurry_folder, threshold_sobel=100, threshold_fft=3):
    """
    Parcourt un dossier d'images, et classe les images comme floues ou nettes en fonction des seuils.
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
                        blurry, sobel_variance_value, fft_blur_metric = is_blurry(image_path, threshold_sobel, threshold_fft)

                        # Charger l'image pour l'annoter et afficher les résultats
                        image = cv2.imread(image_path)

                        # Ajouter le texte avec les mesures de flou
                        add_text_with_black_background(image, sobel_variance_value, fft_blur_metric)

                        # Afficher l'image avec les annotations
                        """cv2.imshow(f"Image avec annotations - {file}", image)
                        cv2.waitKey(0)  # Attendre que l'utilisateur appuie sur une touche pour passer à l'image suivante
                        cv2.destroyAllWindows()"""

                        # Afficher les résultats et déplacer l'image
                        if blurry:
                            print(f"[FLU] {image_path} - Sobel Variance: {sobel_variance_value:.2f}, FFT Metric: {fft_blur_metric:.2f}")
                            shutil.copy(image_path, blurry_folder)  # Copier l'image floue
                        else:
                            print(f"[NETTE] {image_path} - Sobel Variance: {sobel_variance_value:.2f}, FFT Metric: {fft_blur_metric:.2f}")
                            shutil.copy(image_path, sharp_folder)  # Copier l'image nette

                        # Sauvegarder l'image annotée
                        annotated_image_path = os.path.join(blurry_folder if blurry else sharp_folder, file)
                        cv2.imwrite(annotated_image_path, image)

# Exécution de la classification d'images
parent_folder = r"..\..\Copie_TER\Donnees\birds_dataset2"  # Dossier contenant les images
sharp_folder = r"..\Donnees\sharp_images_sobel_fft"  # Dossier pour les images nettes
blurry_folder = r"..\Donnees\blurry_images_sobel_fft"  # Dossier pour les images floues

classify_images(parent_folder, sharp_folder, blurry_folder, threshold_sobel=2300, threshold_fft=7)
