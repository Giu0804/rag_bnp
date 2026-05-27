import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Set

# =====================================================================
# 1. OUTILS DE NETTOYAGE ET MÉTRIQUES
# =====================================================================

def _citefix_clean_tokens(text: str) -> Set[str]:
    """Nettoyage strict : minuscules, sans ponctuation, sans mots vides."""
    if not text:
        return set()
    text = text.lower()
    text = re.sub(r'[^\w\s\-]', ' ', text)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were',
        'le', 'la', 'les', 'de', 'des', 'un', 'une', 'en', 'que', 'est', 'et', 'a', 'dans', 'pour', 'par', 'sur', 'qui'
    }
    return {w for w in text.split() if w and w not in stop_words}

def calculate_metrics(gold_ids: List[int], pred_ids: List[int]) -> Dict[str, float]:
    """Calcule la Précision, le Rappel, le F1 et l'Exact Match pour un sample."""
    gold_set = set(gold_ids)
    pred_set = set(pred_ids)
    
    correct = gold_set.intersection(pred_set)
    
    p = len(correct) / len(pred_set) if pred_set else 0.0
    r = len(correct) / len(gold_set) if gold_set else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    em = 1.0 if gold_set == pred_set else 0.0
    
    return {"precision": p, "recall": r, "f1": f1, "accuracy": em}

# =====================================================================
# 2. LOGIQUE MATHÉMATIQUE KSC (CITEFIX)
# =====================================================================

def calculate_ksc_score(claim_tokens: Set[str], chunk_tokens: Set[str], question_tokens: Set[str], alpha: float) -> float:
    """Implémente la formule d'overlap asymétrique pondérée par alpha."""
    # 1. Jaccard entre le Fait Atomique et le Chunk
    intersection_xc = claim_tokens.intersection(chunk_tokens)
    union_xc = claim_tokens.union(chunk_tokens)
    sim_lexique = len(intersection_xc) / len(union_xc) if union_xc else 0.0
    
    # 2. Jaccard entre la Question et le Chunk
    if question_tokens:
        intersection_qc = question_tokens.intersection(chunk_tokens)
        union_qc = question_tokens.union(chunk_tokens)
        sim_retriever = len(intersection_qc) / len(union_qc) if union_qc else 0.0
    else:
        sim_retriever = 0.0
        
    # 3. L'équation finale
    return alpha * sim_lexique + (1 - alpha) * sim_retriever

def run_citefix_ksc(claims: List[str], contexts: List[Dict], question: str, alpha: float, threshold: float) -> List[int]:
    """Scanne tous les faits contre tous les chunks, et retient ceux >= threshold."""
    predicted_ids = set()
    question_tokens = _citefix_clean_tokens(question)
    
    for fact in claims:
        fact_tokens = _citefix_clean_tokens(fact)
        if not fact_tokens:
            continue
            
        for chunk in contexts:
            chunk_tokens = _citefix_clean_tokens(chunk['text'])
            score = calculate_ksc_score(fact_tokens, chunk_tokens, question_tokens, alpha)
            
            if score >= threshold:
                predicted_ids.add(int(chunk['id']))
                
    return sorted(list(predicted_ids))

# =====================================================================
# 3. LE MOTEUR DE RECHERCHE SUR GRILLE (GRID SEARCH)
# =====================================================================

def run_ksc_grid_search(parquet_path: str, alpha_list=[0.2, 0.5, 0.8, 1.0], threshold_list=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]):
    """Teste toutes les combinaisons sur ton jeu de données Parquet."""
    df_saved = pd.read_parquet(parquet_path)
    grid_results = []
    
    print(f"🔬 Démarrage de l'analyse sur {len(df_saved)} échantillons...")
    
    for alpha in alpha_list:
        for t in threshold_list:
            all_metrics = []
            
            for index, row in df_saved.iterrows():
                # Gestion souple de la colonne atomic_claims
                atomic_facts = row["atomic_claims"]
                if isinstance(atomic_facts, str):
                    atomic_facts = [f.strip("- *").strip() for f in atomic_facts.split('\n') if f.strip()]
                
                # Exécution et évaluation
                pred_ids = run_citefix_ksc(atomic_facts, row["contexts"], row["question"], alpha, t)
                metrics = calculate_metrics(row["gold_ids"], pred_ids)
                all_metrics.append(metrics)
                
            # Moyennes MACRO pour la combinaison actuelle
            grid_results.append({
                "alpha": alpha,
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
    
    print("\n" + "=" * 22 + " CONFIGURATIONS OPTIMALES TROUVÉES " + "=" * 22)
    print(f"🏆 [MEILLEUR F1-SCORE]")
    print(f"   Alpha: {best_f1_row['alpha']} | Seuil: {best_f1_row['threshold']}")
    print(f"   F1: {best_f1_row['f1']:.2%} | Exact Match: {best_f1_row['accuracy']:.2%}")
    print(f"   (P: {best_f1_row['precision']:.2%}, R: {best_f1_row['recall']:.2%})")
    print(r"  " + "-"*40)
    print(f"🎯 [MEILLEUR EXACT MATCH]")
    print(f"   Alpha: {best_acc_row['alpha']} | Seuil: {best_acc_row['threshold']}")
    print(f"   Exact Match: {best_acc_row['accuracy']:.2%} | F1: {best_acc_row['f1']:.2%}")
    print(f"   (P: {best_acc_row['precision']:.2%}, R: {best_acc_row['recall']:.2%})")
    print("=" * 79 + "\n")
    
    return df_grid

# =====================================================================
# 4. GÉNÉRATION DES GRAPHIQUES (VISUALISATION SCIENTIFIQUE)
# =====================================================================

def plot_ksc_results(df_grid: pd.DataFrame):
    """Génère les graphiques pour voir l'impact du seuil t."""
    alphas = df_grid['alpha'].unique()
    plt.figure(figsize=(12, 4 * len(alphas)))
    
    for i, alpha in enumerate(alphas):
        df_sub = df_grid[df_grid['alpha'] == alpha].sort_values(by='threshold')
        
        plt.subplot(len(alphas), 1, i + 1)
        plt.plot(df_sub['threshold'], df_sub['precision'], label='Précision', marker='o', color='#2ca02c', lw=2)
        plt.plot(df_sub['threshold'], df_sub['recall'], label='Rappel', marker='x', color='#d62728', lw=2)
        plt.plot(df_sub['threshold'], df_sub['f1'], label='F1-Score', linestyle='--', color='#7f7f7f', lw=2)
        
        plt.title(f"Performance pour Alpha = {alpha}", fontweight='bold')
        plt.xlabel("Seuil critique (t)")
        plt.ylabel("Score")
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(loc='lower left')
        
    plt.tight_layout()
    plt.show()

# --- COMMENT LANCER LE CODE ---
# df_resultats = run_ksc_grid_search("ton_dossier/model_output.parquet")
# plot_ksc_results(df_resultats)




def plot_ksc_alpha_impact(df_grid: pd.DataFrame):
    """
    Génère un tableau de bord 2x2 montrant l'impact d'Alpha (en abscisse) 
    pour chaque seuil critique testé.
    """
    # On récupère tous les seuils testés et on les trie
    thresholds = sorted(df_grid['threshold'].unique())
    
    # Création d'une grille de 4 graphiques (2 lignes, 2 colonnes)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten() # Pour itérer facilement sur les 4 cases
    
    # Nos 4 métriques à afficher
    metrics = [
        ('precision', 'Précision Moyenne'),
        ('recall', 'Rappel Moyen'),
        ('f1', 'F1-Score'),
        ('accuracy', 'Exact Match (Accuracy)')
    ]
    
    # Création d'une palette de couleurs dynamique (du violet au jaune) 
    # pour bien distinguer les différentes lignes de seuil
    colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
    
    for i, (col, title) in enumerate(metrics):
        for j, t in enumerate(thresholds):
            # On isole les données d'un seul seuil, et on les trie par Alpha
            df_sub = df_grid[df_grid['threshold'] == t].sort_values(by='alpha')
            
            # On trace la courbe : X = Alpha, Y = La métrique
            axes[i].plot(
                df_sub['alpha'], 
                df_sub[col], 
                label=f'Seuil (t) = {t}', 
                marker='o', 
                linewidth=2, 
                color=colors[j]
            )
            
        axes[i].set_title(f"Impact d'Alpha sur : {title}", fontweight='bold', fontsize=12)
        axes[i].set_xlabel("Paramètre Alpha (α)", fontsize=10)
        axes[i].set_ylabel("Score (0.0 à 1.0)", fontsize=10)
        axes[i].set_ylim(0, 1.05)
        axes[i].grid(True, linestyle=':', alpha=0.6)
        
        # On place la légende intelligemment
        axes[i].legend(title="Seuils critiques", fontsize='small', loc='best')

    plt.tight_layout()
    plt.show()












def plot_ksc_alpha_impact(df_grid: pd.DataFrame):
    """
    Génère un graphique par Seuil (t).
    Sur chaque graphique, toutes les métriques évoluent en fonction d'Alpha (X).
    """
    # On récupère la liste des seuils testés
    thresholds = sorted(df_grid['threshold'].unique())
    
    # On crée une figure verticale (un bloc par seuil)
    plt.figure(figsize=(10, 5 * len(thresholds)))
    
    for i, t in enumerate(thresholds):
        # On filtre les données pour ne garder que le seuil actuel et on trie par Alpha
        df_sub = df_grid[df_grid['threshold'] == t].sort_values(by='alpha')
        
        plt.subplot(len(thresholds), 1, i + 1)
        
        # On trace TOUTES les métriques sur le MÊME graphique
        plt.plot(df_sub['alpha'], df_sub['precision'], label='Précision', marker='o', color='#2ca02c', lw=2)
        plt.plot(df_sub['alpha'], df_sub['recall'], label='Rappel', marker='x', color='#d62728', lw=2)
        plt.plot(df_sub['alpha'], df_sub['f1'], label='F1-Score', linestyle='--', color='#7f7f7f', lw=2)
        plt.plot(df_sub['alpha'], df_sub['accuracy'], label='Exact Match (Acc)', marker='s', color='#1f77b4', lw=2)
        
        plt.title(f"Impact d'Alpha pour le Seuil constant (t) = {t}", fontweight='bold')
        plt.xlabel("Paramètre Alpha (α)")
        plt.ylabel("Score")
        plt.ylim(0, 1.05)
        plt.xlim(df_grid['alpha'].min() - 0.05, df_grid['alpha'].max() + 0.05)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(loc='lower left')
        
    plt.tight_layout()
    plt.show()