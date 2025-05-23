import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, cdist, pdist

# 1) Lecture des données
df_birds = pd.read_csv('../../Donnees/birds_conv2_features.csv')

def test_one_image(image_name, df, n_bg=9, n_null=1000):
    emb0 = df.loc[df.image_name == image_name, df.columns[1:]].values.astype(float)
    emb0 = emb0.reshape(1, -1)

    species = image_name.split('_')[0]
    same = df[(df.image_name.str.startswith(species + '_')) & (df.image_name != image_name)]
    # ici on sait que same.shape[0] >= n_bg
    emb_same = same.sample(n_bg, random_state=0).iloc[:,1:].values
    d_intra = cdist(emb0, emb_same, 'euclidean').mean()

    others = df[~df.image_name.str.startswith(species + '_')]
    emb_others = others.iloc[:,1:].values
    null_means = np.zeros(n_null)
    for i in range(n_null):
        idx = np.random.choice(len(emb_others), size=n_bg, replace=False)
        null_means[i] = cdist(emb0, emb_others[idx,:], 'euclidean').mean()

    return d_intra, null_means

# 2) Ne garder que les espèces ayant au moins n_bg+1 images
n_bg = 9
species_counts = df_birds['image_name'].apply(lambda x: x.split('_')[0]).value_counts()
valid_species = species_counts[species_counts >= n_bg+1].index.tolist()[:10]

# 3) Tracer la grille 2×5 pour ces 10 espèces
fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)

for ax, sp in zip(axes.flat, valid_species):
    img0 = df_birds[df_birds.image_name.str.startswith(sp + '_')]['image_name'].iloc[0]
    d_intra, null_dist = test_one_image(img0, df_birds, n_bg=n_bg, n_null=1000)

    ax.hist(null_dist, bins=30)
    ax.axvline(d_intra, color='red', linestyle='--', label=f'intra={d_intra:.2f}')
    ax.set_title(f"{sp} ({img0})")
    ax.set_xlabel("Distance moyenne")
    ax.set_ylabel("Fréquence")
    ax.legend()

plt.savefig('distributions_grid.png', dpi=300)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
import textwrap

# (reprise du calcul de null_dists et intra_distance...)
# → supposons que `scenes` contient vos 10 orig_images

fig, axes = plt.subplots(2, 5, figsize=(22, 10), constrained_layout=True)

for ax, orig in zip(axes.flat, scenes):
    # histogramme de la distribution nulle
    ax.hist(null_dists, bins=50, alpha=0.7)

    # distance intra
    d_intra = intra_distance(orig, df_bg, feature_cols)
    if d_intra is not None:
        ax.axvline(d_intra, color='red', linestyle='--',
                   label=f'intra={d_intra:.2f}')
        ax.legend(fontsize='small')

    # on ne garde que l'espèce en titre, en 2 lignes max
    species = orig.split('_')[0]
    wrapped = textwrap.fill(species, width=12)
    ax.set_title(wrapped, fontsize=10)

    # on place l'ID complet en sous-titre (plus discret, police plus petite)
    ax.text(0.5, 0.94, orig, transform=ax.transAxes,
            fontsize=7, va='top', ha='center', color='gray')

    ax.set_xlabel("Distance euclidienne", fontsize=8)
    ax.set_ylabel("Fréquence", fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=7)

plt.savefig('bg_distributions_diversite_especes.png', dpi=300)
plt.show()

# 0) Paramètres
CSV_BG = '../../Donnees/backgrounds_conv2_features.csv'

# 1) Chargement et extraction de l’espèce et de l’image d’origine
df = pd.read_csv(CSV_BG)
tokens = df['image_name'].str.split('_')
df['species']    = tokens.str[:2].str.join('_')
df['orig_image'] = df['image_name'].str.split(pat='_bg_', n=1).str[0]

# 2) Matrice d’embeddings et vecteurs de labels
feat_cols = [c for c in df.columns if c.startswith('f')]
X      = df[feat_cols].values         # (N, D)
sp     = df['species'].values         # (N,)
orig   = df['orig_image'].values      # (N,)

# 3) Calcul de toutes les distances pair-à-pair
all_dists = pdist(X, metric='euclidean')  # longueur M = N*(N-1)/2

# 4) Construction des masques sur le triangle supérieur (i<j)
N = len(df)
i, j = np.triu_indices(N, k=1)

same_species = (sp[:, None] == sp[None, :])[i, j]
same_image   = (orig[:, None] == orig[None, :])[i, j]

# 5) Séparation intra-espèce vs inter-espèce
#   - intra-espèce : même espèce mais pas même image d’origine
mask_intra_sp   = same_species & (~same_image)
dists_intra_sp  = all_dists[mask_intra_sp]

#   - inter-espèce : espèces différentes
mask_inter_sp   = ~same_species
dists_inter_sp  = all_dists[mask_inter_sp]

# 6) Statistiques
print(f"Intra-espèce : {len(dists_intra_sp)} paires")
print(f"Inter-espèce: {len(dists_inter_sp)} paires")
print(f"Intra-espèce – min/max/mean = "
      f"{dists_intra_sp.min():.2f}/"
      f"{dists_intra_sp.max():.2f}/"
      f"{dists_intra_sp.mean():.2f}")
print(f"Inter-espèce – min/max/mean = "
      f"{dists_inter_sp.min():.2f}/"
      f"{dists_inter_sp.max():.2f}/"
      f"{dists_inter_sp.mean():.2f}")

# 7) Sous-échantillonnage optionnel pour équilibrer les effectifs
rng = np.random.default_rng(42)
sampled_inter_sp = rng.choice(
    dists_inter_sp,
    size=len(dists_intra_sp),
    replace=False
)

# 8) Tracé comparatif
plt.figure(figsize=(10,4))

# a) histogrammes normalisés (densités)
plt.subplot(1,2,1)
plt.hist(dists_intra_sp, bins=50, density=True,
         alpha=0.6, label='Intra-espèce')
plt.hist(dists_inter_sp, bins=50, density=True,
         alpha=0.6, label='Inter-espèce')
plt.title("Histogramme (densités)")
plt.xlabel("Distance euclidienne")
plt.ylabel("Densité")
plt.legend()

# b) histogrammes avec effectifs égalisés
plt.subplot(1,2,2)
plt.hist(dists_intra_sp, bins=50, alpha=0.6,
         label='Intra-espèce')
plt.hist(sampled_inter_sp, bins=50, alpha=0.6,
         label='Inter-espèce (échant.)')
plt.title("Histogramme (effectifs égalisés)")
plt.xlabel("Distance euclidienne")
plt.ylabel("Nombre de paires")
plt.legend()

plt.tight_layout()
plt.savefig('intra_vs_inter_espece.png', dpi=300, bbox_inches='tight')
plt.show()


