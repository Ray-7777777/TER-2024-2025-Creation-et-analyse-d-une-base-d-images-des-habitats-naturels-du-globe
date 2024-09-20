import gzip
import pandas as pd
import subprocess


path_bash = 'C:\\Program Files\\Git\\bin\\bash.exe' #Il sera peut-être nécessaire de changer le chemin de bash suivant votre installation
n_photos = 500 #Nombre de photos à télécharger


# Chemins vers les fichiers
gz_file_path_taxa = "../donnees/taxa.csv.gz"
gz_file_path_observations = "../donnees/observations.csv.gz"
photos_file_path = "../donnees/photos.csv.gz"

# 1. Lire le fichier taxa et arrêter après avoir trouvé 1 taxon_id
taxon_ids = []
taxon_name = []
with gzip.open(gz_file_path_taxa, 'rt') as f:
    for chunk in pd.read_csv(f, sep='\t', chunksize=1000):
        for index, row in chunk.iterrows():
            if len(str(row['taxon_id'])) == 4:
                taxon_ids.append(row['taxon_id'])
                taxon_name.append(row["name"])

# 2. Lire le fichier observations et s'arrêter après avoir trouvé n observation_uuid
observation_ids = []
with gzip.open(gz_file_path_observations, 'rt') as f:
    for chunk in pd.read_csv(f, sep='\t', chunksize=1000):
        for index, row in chunk.iterrows():
            if row['taxon_id'] in taxon_ids:
                observation_ids.append(row['observation_uuid'])
                if len(observation_ids) >= 20:
                    break  # Stop dès qu'on a n observations
        if len(observation_ids) >= 20:  # Stop si on a trouvé n observation_uuid
            break

# 3. Lire le fichier photos et s'arrêter après avoir trouvé n photo_id
photo_ids = []
photos_ext = []
df_photos_chunks = pd.read_csv(photos_file_path, chunksize=1000, sep="\t")

for chunk in df_photos_chunks:
    for index, row in chunk.iterrows():
        if row['observation_uuid'] in observation_ids:
            photo_ids.append(row['photo_id'])
            photos_ext.append(row['extension'])
            if len(photo_ids) >= n_photos:
                break
    if len(photo_ids) >= n_photos:  # Stop si on a trouvé n photo_id
        break

with open('download_photos.sh', 'w') as f:
    f.write("#!/bin/bash\n\n")

    for photo_id, ext in zip(photo_ids, photos_ext):
        # Créer la commande `aws s3 cp` pour chaque photo_id avec son extension
        s3_command = f"aws s3 cp s3://inaturalist-open-data/photos/{photo_id}/original.{ext} ../photos/{photo_id}.{ext} --no-sign-request\n"
        f.write(s3_command)

subprocess.run([path_bash, './download_photos.sh'], check=True)