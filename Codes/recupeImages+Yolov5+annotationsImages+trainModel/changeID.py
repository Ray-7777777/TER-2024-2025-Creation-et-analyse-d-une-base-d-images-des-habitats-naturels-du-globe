import os

def replace_class_id_in_txt_files(directory):
    # Parcours tous les fichiers dans le répertoire donné et ses sous-dossiers
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                # Chemin du fichier .txt
                file_path = os.path.join(root, file)
                
                # Ouvre le fichier en lecture
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                # Remplacer le class_id par 0 dans chaque ligne
                with open(file_path, 'w') as f:
                    for line in lines:
                        parts = line.split()
                        if parts:  # Si la ligne n'est pas vide
                            parts[0] = '0'  # Remplacer le class_id par 0
                        f.write(' '.join(parts) + '\n')  # Écrire la ligne modifiée

# Utilise le répertoire actuel (ou remplace par un répertoire spécifique si tu préfères)
if __name__ == "__main__":
    current_directory = os.getcwd()  # Utilise le répertoire actuel
    replace_class_id_in_txt_files(current_directory)
