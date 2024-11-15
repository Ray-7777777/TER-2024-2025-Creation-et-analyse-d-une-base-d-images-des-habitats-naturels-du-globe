import os

def delete_unlabeled_images(images_folder, labels_folder):
    # Parcours toutes les images dans le dossier 'images'
    for image_file in os.listdir(images_folder):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):  # Extensions d'images courantes
            # Vérifie si le fichier correspondant existe dans le dossier 'labels'
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_folder, label_file)

            # Si le fichier label n'existe pas, supprimer l'image
            if not os.path.exists(label_path):
                print(f"Le fichier label pour {image_file} n'existe pas. Suppression de l'image.")
                image_path = os.path.join(images_folder, image_file)
                os.remove(image_path)  # Supprimer l'image

if __name__ == "__main__":
    images_folder = 'images'  # Dossier des images
    labels_folder = 'labels'  # Dossier des labels
    delete_unlabeled_images(images_folder, labels_folder)
