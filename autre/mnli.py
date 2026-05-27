import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import pipeline
from typing import List, Dict

# =====================================================================
# 1. OUTILS ET MÉTRIQUES
# =====================================================================

def calculate_metrics(gold_ids: List[int], pred_ids: List[int]) -> Dict[str, float]:
    """Calcule les performances pour une liste d'IDs prédits."""
    gold_set = set(gold_ids)
    pred_set = set(pred_ids)
    
    correct = gold_set.intersection(pred_set)
    
    p = len(correct) / len(pred_set) if pred_set else 0.0
    r = len(correct) / len(gold_set) if gold_set else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    em = 1.0 if gold_set == pred_set else 0.0
    
    return {"precision": p, "recall": r, "f1": f1, "accuracy": em}

# =====================================================================
# 2. LOGIQUE NLI (L'INFÉRENCE UNIQUE PAR BART)
# =====================================================================

def get_all_entailment_scores(df: pd.DataFrame, nli_model) -> List[Dict]:
    """
    Passe tout le dataset dans BART une seule fois.
    Retourne une liste contenant, pour chaque sample, les scores de chaque chunk.
    """
    print(f"⏳ Phase 1/2 : Inférence BART sur {len(df)} samples (Batch processing)...")
    all_sample_scores = []
    
    for index, row in df.iterrows():
        # Extraction sécurisée des faits atomiques
        atomic_facts = row["atomic_claims"]
        if isinstance(atomic_facts, str):
            atomic_facts = [f.strip("- *").strip() for f in atomic_facts.split('\n') if f.strip()]
            
        contexts = row["contexts"]
        
        # Stockage des scores bruts pour CE sample (clé: chunk_id, valeur: score max trouvé)
        chunk_entailment_scores = {chunk['id']: 0.0 for chunk in contexts}
        
        if not atomic_facts:
            all_sample_scores.append(chunk_entailment_scores)
            continue
            
        # Création de TOUTES les paires [Chunk, Fait] pour ce sample
        pairs = []
        pair_mapping = [] # Pour retenir quel chunk_id correspond à quelle paire
        
        for fact in atomic_facts:
            for chunk in contexts:
                pairs.append({"text": chunk['text'], "text_pair": fact})
                pair_mapping.append(int(chunk['id']))
                
        # Inférence rapide en une passe
        results = nli_model(pairs, top_k=None, batch_size=32)
        
        # Extraction du score 'ENTAILMENT'
        for chunk_id, chunk_scores in zip(pair_mapping, results):
            for score_dict in chunk_scores:
                if 'ENTAIL' in str(score_dict['label']).upper():
                    entail_prob = score_dict['score']
                    # Un chunk peut valider plusieurs faits. On garde son score d'implication le plus fort.
                    if entail_prob > chunk_entailment_scores[chunk_id]:
                        chunk_entailment_scores[chunk_id] = entail_prob
                    break
                    
        all_sample_scores.append(chunk_entailment_scores)
        
    return all_sample_scores

# =====================================================================
# 3. LE MOTEUR DE GRID SEARCH NLI
# =====================================================================

def run_nli_grid_search(parquet_path: str, threshold_list=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]):
    """Applique les différents seuils de probabilité sur les scores pré-calculés."""
    df_saved = pd.read_parquet(parquet_path)
    
    # Initialisation du modèle BART sur le GPU
    print("🤖 Chargement de facebook/bart-large-mnli en mémoire...")
    nli_judge = pipeline("text-classification", model="facebook/bart-large-mnli", device=0)
    
    # Étape 1 : On calcule tous les scores bruts une bonne fois pour toutes
    cached_scores = get_all_entailment_scores(df_saved, nli_judge)
    
    # Nettoyage de la VRAM (le modèle a fini son travail)
    del nli_judge
    torch.cuda.empty_cache()
    
    print("📈 Phase 2/2 : Calcul ultra-rapide de la grille des seuils...")
    grid_results = []
    
    # Étape 2 : On teste nos seuils mathématiquement
    for t in threshold_list:
        all_metrics = []
        
        for index, row in df_saved.iterrows():
            scores_dict = cached_scores[index]
            gold_ids = row["gold_ids"]
            
            # Un ID est prédit si son score d'entailment est supérieur ou égal au seuil t
            pred_ids = [chunk_id for chunk_id, score in scores_dict.items() if score >= t]
            
            metrics = calculate_metrics(gold_ids, sorted(pred_ids))
            all_metrics.append(metrics)
            
        # Moyennes MACRO pour ce seuil
        grid_results.append({
            "threshold": t,
            "precision": np.mean([m['precision'] for m in all_metrics]),
            "recall": np.mean([m['recall'] for m in all_metrics]),
            "f1": np.mean([m['f1'] for m in all_metrics]),
            "accuracy": np.mean([m['accuracy'] for m in all_metrics])
        })
        
    df_grid = pd.DataFrame(grid_results)
    
    # --- AFFICHAGE DES DEUX MEILLEURS COMPROMIS ---
    best_f1_row = df_grid.loc[df_grid['f1'].idxmax()]
    best_acc_row = df_grid.loc[df_grid['accuracy'].idxmax()]
    
    print("\n" + "=" * 22 + " CONFIGURATIONS OPTIMALES BART-MNLI " + "=" * 22)
    print(f"🏆 [MEILLEUR F1-SCORE]")
    print(f"   Seuil (t) de confiance : {best_f1_row['threshold']}")
    print(f"   F1: {best_f1_row['f1']:.2%} | Exact Match: {best_f1_row['accuracy']:.2%}")
    print(f"   (P: {best_f1_row['precision']:.2%}, R: {best_f1_row['recall']:.2%})")
    print(r"  " + "-"*40)
    print(f"🎯 [MEILLEUR EXACT MATCH]")
    print(f"   Seuil (t) de confiance : {best_acc_row['threshold']}")
    print(f"   Exact Match: {best_acc_row['accuracy']:.2%} | F1: {best_acc_row['f1']:.2%}")
    print(f"   (P: {best_acc_row['precision']:.2%}, R: {best_acc_row['recall']:.2%})")
    print("=" * 79 + "\n")
    
    return df_grid

# =====================================================================
# 4. GÉNÉRATION DU GRAPHIQUE (VISUALISATION SCIENTIFIQUE)
# =====================================================================

def plot_nli_results(df_grid: pd.DataFrame):
    """Génère un graphique de performance en fonction de la probabilité exigée."""
    df_grid = df_grid.sort_values(by='threshold')
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(df_grid['threshold'], df_grid['precision'], label='Précision', marker='o', color='#2ca02c', lw=2)
    plt.plot(df_grid['threshold'], df_grid['recall'], label='Rappel', marker='x', color='#d62728', lw=2)
    plt.plot(df_grid['threshold'], df_grid['f1'], label='F1-Score', linestyle='--', color='#7f7f7f', lw=2)
    plt.plot(df_grid['threshold'], df_grid['accuracy'], label='Exact Match (Acc)', marker='s', color='#1f77b4', lw=2)
    
    plt.title("Performance du juge BART-MNLI selon le seuil de probabilité", fontsize=14, fontweight='bold')
    plt.xlabel("Seuil de Confiance 'Entailment' (t)", fontsize=12)
    plt.ylabel("Score (0.0 à 1.0)", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xlim(0.35, 1.0)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower left', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# --- COMMENT LANCER LE CODE ---
# df_resultats_nli = run_nli_grid_search("ton_dossier/model_output.parquet")
# plot_nli_results(df_resultats_nli)