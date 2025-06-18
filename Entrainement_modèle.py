import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from ctgan import CTGAN
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from sdmetrics.reports.single_table import QualityReport
import warnings
warnings.filterwarnings("ignore")

#  chemins 
base_dir = r"C:\Users\VictorAudoux\Desktop\synthetic_health_data\data"
fichier_entree = os.path.join(base_dir, "effectifs.csv")
fichier_sortie_fusion = os.path.join(base_dir, "donnees_synthetiques_fusion.csv")

print("Chargement du fichier CSV...")
df = pd.read_csv(fichier_entree, sep=";")
print("Fichier chargé avec succès.")

colonnes_utiles = ["annee", "patho_niv1", "cla_age_5", "sexe", "region", "dept", "Npop", "prev"]
df = df[colonnes_utiles]
# Réduction des classes rares dans 'patho_niv1'
top_classes = df['patho_niv1'].value_counts().nlargest(10).index
df = df[df['patho_niv1'].isin(top_classes)]
# Fonction d'équilibrage (à inclure si elle n'est pas encore dans le script)
def resample_grouped_data(df, group_cols, max_samples=500):
    return (
        df.groupby(group_cols, group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), max_samples), random_state=42))
        .reset_index(drop=True)
    )

df = resample_grouped_data(df, group_cols=["annee", "dept", "patho_niv1"], max_samples=300)

df["prev"] = df["prev"].astype(str).str.replace(",", ".").astype(float)
df["Npop"] = pd.to_numeric(df["Npop"], errors="coerce")
df = df[(df["prev"] >= 0) & (df["prev"] <= 100)]
df = df[(df["Npop"] >= 1000) & (df["Npop"] <= 200000)]
df.dropna(inplace=True)
print(f"\u2705 Données filtrées ({len(df)} lignes).")

for col in ["annee", "patho_niv1", "cla_age_5", "sexe", "region", "dept"]:
    df[col] = df[col].astype(str)

df["Npop_log"] = np.log1p(df["Npop"])

df = df.sample(n=50000, random_state=42)


df_train = df.drop(columns=["Npop"])         
df_eval = df.drop(columns=["Npop_log"])       

colonnes_categorielle = ["annee", "patho_niv1", "cla_age_5", "sexe", "region", "dept"]


# === 4. CTGAN ===
print(" Entraînement CTGAN...")
ctgan = CTGAN(epochs=50)
ctgan.fit(df_train, discrete_columns=colonnes_categorielle)
synth_ctgan = ctgan.sample(10000)
synth_ctgan["Npop"] = np.expm1(synth_ctgan["Npop_log"]).clip(1000, 200000)  # Re-transforme
synth_ctgan["prev"] = synth_ctgan["prev"].clip(0, 100)
synth_ctgan.drop(columns=["Npop_log"], inplace=True)
print("CTGAN terminé.")

print("Entraînement TVAE...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_eval) 
tvae = TVAESynthesizer(metadata)
tvae.fit(df_eval)
# Génération TVAE
synth_tvae = tvae.sample(10000)
synth_tvae["Npop"] = synth_tvae["Npop"].clip(1000, 200000)
synth_tvae["prev"] = synth_tvae["prev"].clip(0, 100)
print("TVAE terminé.")

#  Fusion moyenne pondérée
print("Fusion moyenne pondérée des données CTGAN (70%) + TVAE (30%)...")

# Échantillonnage aligné pour une moyenne ligne à ligne
synth_ctgan_sample = synth_ctgan.sample(n=1000, random_state=42).reset_index(drop=True)
synth_tvae_sample = synth_tvae.sample(n=1000, random_state=42).reset_index(drop=True)

# Définir les colonnes
colonnes_continues = ["prev", "Npop"]
colonnes_categorielle = ["annee", "patho_niv1", "cla_age_5", "sexe", "region", "dept"]

# moyenne pondérée : 50% CTGAN + 50% TVAE
fusion_ponderee = pd.DataFrame()
fusion_ponderee[colonnes_categorielle] = synth_ctgan_sample[colonnes_categorielle]
for col in colonnes_continues:
    fusion_ponderee[col] = (
        0.5 * synth_ctgan_sample[col] + 0.5 * synth_tvae_sample[col]
    )

def recalibrer_categorielle(df_source, df_synth, colonnes, n_final):
    df_intermediaire = df_synth.copy()

    for col in colonnes:
        print(f"Recalibrage de la colonne : {col}")
        distrib_reelle = df_source[col].value_counts(normalize=True)
        fusion_temp = []

        for categorie, poids in distrib_reelle.items():
            n_categorie = int(poids * n_final)
            subset = df_intermediaire[df_intermediaire[col] == categorie]

            if len(subset) == 0:
                print(f" Catégorie absente dans les données synthétiques : {categorie}")
                continue

            if len(subset) >= n_categorie:
                echantillon = subset.sample(n=n_categorie, replace=False, random_state=42)
            else:
                echantillon = subset.sample(n=n_categorie, replace=True, random_state=42)

            fusion_temp.append(echantillon)

        # Fusionne les sous-échantillons
        df_intermediaire = pd.concat(fusion_temp, ignore_index=True)

    print(f"ℹ️ Total lignes après concaténation : {len(df_intermediaire)} (vs attendu : {n_final})")
    if len(df_intermediaire) >= n_final:
        df_final = df_intermediaire.sample(n=n_final, replace=False, random_state=42)
    else:
        df_final = df_intermediaire.sample(n=n_final, replace=True, random_state=42)

    return df_final.reset_index(drop=True)



# appliquer recalibrage sur les colonnes catégorielles
colonnes_categorielle = ["annee", "patho_niv1", "cla_age_5", "sexe", "region", "dept"]
synth_fusion_recalibree = recalibrer_categorielle(
    df_eval,
    fusion_ponderee,
    colonnes_categorielle,
    n_final=1000,
    poids_renforce={"region": 2.0, "dept": 2.5}  #  augmente pour renforcer
)

# sauvegarde
synth_fusion_recalibree.to_csv(fichier_sortie_fusion, index=False)
print("✅ Synthétiques recalibrées et enregistrées.")

# nettoyage post-fusion
synth_fusion_recalibree.drop_duplicates(inplace=True)

def supprimer_outliers(df, col, quantile_inf=0.01, quantile_sup=0.99):
    borne_inf = df[col].quantile(quantile_inf)
    borne_sup = df[col].quantile(quantile_sup)
    return df[(df[col] >= borne_inf) & (df[col] <= borne_sup)]

synth_fusion_recalibree = supprimer_outliers(synth_fusion_recalibree, "prev")
synth_fusion_recalibree = supprimer_outliers(synth_fusion_recalibree, "Npop")
synth_fusion_recalibree.reset_index(drop=True, inplace=True)

synth_fusion = synth_fusion_recalibree 


# visualisation 
def comparer_distributions(colonne, titre):
    plt.figure(figsize=(10, 5))
    plt.hist(df_eval[colonne], bins=50, alpha=0.5, label='Réelles', density=True)
    plt.hist(synth_fusion[colonne], bins=50, alpha=0.5, label='Synthétiques fusionnées', density=True)
    plt.title(f"Distribution de '{titre}'")
    plt.xlabel(colonne)
    plt.ylabel("Densité")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

comparer_distributions("prev", "Prévalence")
comparer_distributions("Npop", "Population")

# évaluation statistique
print(" ÉVALUATION STATISTIQUE")
for col in ["prev", "Npop"]:
    stat, p = ks_2samp(df_eval[col], synth_fusion[col])
    print(f"KS test '{col}': stat={stat:.3f}, p={p:.3f} {'bien' if p > 0.05 else 'nul'}")

def js_divergence(col):
    dist_real = df_eval[col].value_counts(normalize=True).sort_index()
    dist_synth = synth_fusion[col].value_counts(normalize=True).reindex(dist_real.index, fill_value=0)
    return jensenshannon(dist_real, dist_synth)

for col in colonnes_categorielle:
    jsd = js_divergence(col)
    print(f"JSD '{col}': {jsd:.3f} {'bien' if jsd < 0.1 else 'pas ouf'}")

# rapport qualité SDMetrics
print("\nRapport Qualité SDMetrics...")
quality_report = QualityReport()
quality_report.generate(df_eval, synth_fusion, metadata.to_dict())
print(f"\nScore global SDV : {quality_report.get_score():.2f} / 100")
print("\n Détail :")
print(quality_report.get_score_breakdown()[["Metric", "Score"]])
