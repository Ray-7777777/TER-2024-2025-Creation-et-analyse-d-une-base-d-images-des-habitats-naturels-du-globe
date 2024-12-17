import os
import cv2
import numpy as np
from skimage import feature #necessite d'installer skimage
from keras.applications import VGG16
from keras.preprocessing import image as keras_image
from keras.models import Model

# Charger le modèle VGG16 pré-entraîné sans la couche de classification
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.inputs, outputs=base_model.layers[-1].output)

def compute_color_histogram(image):
    """
    Calcule l'histogramme de couleur pour une image.
    """
    # Convertir l'image en HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calculer l'histogramme pour chaque canal
    h_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
    return h_hist.flatten(), s_hist.flatten(), v_hist.flatten()  # Aplatir les histogrammes

def compute_lbp(image):
    """
    Calcule les descripteurs LBP pour une image.
    """
    lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= hist.sum()  # Normaliser
    return hist

def extract_features_from_image(image_path):
    """
    Extrait les features d'une image, y compris l'histogramme de couleur, LBP et features CNN.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : Impossible de charger l'image {image_path}")
        return None

    # Redimensionner l'image pour le modèle CNN
    resized_image = cv2.resize(image, (224, 224))
    resized_image = keras_image.img_to_array(resized_image)
    resized_image = np.expand_dims(resized_image, axis=0)  # Ajouter une dimension pour le batch
    resized_image = resized_image / 255.0  # Normaliser

    # Extraire les features avec le modèle CNN
    cnn_features = model.predict(resized_image)
    cnn_features = cnn_features.flatten()  # Aplatir pour obtenir un vecteur de features

    # Calculer l'histogramme de couleur
    h_hist, s_hist, v_hist = compute_color_histogram(image)

    # Calculer LBP
    lbp_features = compute_lbp(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    return {
        'cnn_features': cnn_features,
        'h_hist': h_hist,
        's_hist': s_hist,
        'v_hist': v_hist,
        'lbp_features': lbp_features
    }

def extract_features_from_margins(margins_folder, output_file):
    """
    Parcourt le dossier des marges et extrait les features de chaque image.
    """
    features_list = []

    for species in os.listdir(margins_folder):
        species_path = os.path.join(margins_folder, species)

        if os.path.isdir(species_path):
            for file in os.listdir(species_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(species_path, file)
                    features = extract_features_from_image(image_path)
                    if features:
                        features_list.append({
                            'species': species,
                            'filename': file,
                            **features
                        })

    # Sauvegarder les features dans un fichier (par exemple, CSV ou JSON)
    # Ceci est un exemple d'enregistrement des features dans un fichier texte
    with open(output_file, 'w') as f:
        for feature in features_list:
            f.write(f"{feature['species']},{feature['filename']},"
                    f"{','.join(map(str, feature['h_hist']))},"
                    f"{','.join(map(str, feature['s_hist']))},"
                    f"{','.join(map(str, feature['v_hist']))},"
                    f"{','.join(map(str, feature['lbp_features']))},"
                    f"{','.join(map(str, feature['cnn_features']))}\n")

    print("Extraction des features terminée.")

# Utilisation de la fonction
margins_folder = r"../Donnees/birds_dataset/Background_margins"  # Chemin vers le dossier des marges
output_file = r"features_extracted.csv"  # Fichier de sortie pour les features

extract_features_from_margins(margins_folder, output_file)
