def run_and_save_atomic_generation(dataset, sys_prompt, user_prompt_template, model, tokenizer, sample_size=None, output_parquet_path=None):
    """
    Exécute l'inférence sur le dataset avec le prompt atomique et sauvegarde 
    le dataset d'origine enrichi des réponses textuelles au format Parquet.
    """
    generation_results = []
    df_sample = dataset.head(sample_size) if sample_size is not None else dataset

    print(f"🚀 Démarrage de la génération sur {len(df_sample)} échantillons...")

    for index, row in df_sample.iterrows():
        # 1. Préparation du contexte textuel pour le prompt User
        context_text = "\n".join([f"ID: {p['id']} | Paragraph: {p['text']}" for p in row['contexts']])
        formatted_user_prompt = user_prompt_template.format(
            context_text=context_text,
            question=row['question']
        )

        # 2. Inférence locale (via ta fonction generate_llm_response)
        raw_output = generate_llm_response(model, tokenizer, sys_prompt, formatted_user_prompt)

        # 3. Extraction exclusive de la réponse textuelle
        predicted_answer = extract_answer(raw_output)

        # 4. Stockage des données d'origine + la réponse générée
        generation_results.append({
            "id": row['id'],
            "question": row['question'],
            "contexts": row['contexts'],  # Conserve la structure originale [{"id":..., "text":...}]
            "gold_ids": row['gold_ids'],  # Conserve la vérité terrain d'origine
            "generated_answer": predicted_answer  # Notre matière première textuelle atomique
        })

        print(f"✅ Échantillon {index+1}/{len(df_sample)} (ID: {row['id']}) généré.")

    # Transformation en DataFrame conforme au format d'origine + answer
    df_generated = pd.DataFrame(generation_results)

    # Sauvegarde au format Parquet pour conserver parfaitement les listes et dicts
    if output_parquet_path:
        df_generated.to_parquet(output_parquet_path, index=False)
        print(f"\n💾 Dataset d'inférence sauvegardé avec succès dans : {output_parquet_path}")

    return df_generated
















import re
import pandas as pd
import numpy as np
from typing import List, Dict, Set

def _citefix_clean_tokens(text: str) -> Set[str]:
    """
    Nettoyage standard selon la méthodologie du papier :
    Passage en minuscules, suppression de la ponctuation et des stop-words.
    """
    if not text:
        return set()
    text = text.lower()
    # Remplacement de la ponctuation par des espaces
    text = re.sub(r'[^\w\s\-]', ' ', text)
    
    # Liste de stop-words standard (Anglais/Français) pour isoler les mots-clés
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were',
        'le', 'la', 'les', 'de', 'des', 'un', 'une', 'en', 'que', 'est', 'et', 'a', 'dans', 'pour', 'par', 'sur', 'qui'
    }
    return {w for w in text.split() if w and w not in stop_words}

def segment_answer_into_claims(answer_text: str) -> List[str]:
    """
    Étape 1 de la méthodologie : Découpe la réponse A en un ensemble 
    de propositions atomiques {x_1, x_2, ..., x_n}.
    """
    if not answer_text or "Format Error" in answer_text:
        return []
    
    # Extraction des lignes de tirets générées par ton prompt adaptatif
    lines = answer_text.split('\n')
    claims = []
    for line in lines:
        line_clean = line.strip()
        if line_clean.startswith('-'):
            # On retire le tiret pour isoler le texte du claim x_i
            claim = line_clean.lstrip('-').strip()
            if claim:
                claims.append(claim)
                
    # Sécurité : si le format de tiret a échoué, on se replie sur un découpage par phrase
    if not claims:
        import nltk
        claims = nltk.sent_tokenize(answer_text, language='english')
        
    return claims



















def calculate_ksc_score(claim_tokens: Set[str], chunk_tokens: Set[str], question_tokens: Set[str], alpha: float) -> float:
    """
    Calcule l'équation KSC officielle :
    Score = alpha * Sim_lexique(x_i, c_j) + (1 - alpha) * Sim_retriever(q, c_j)
    """
    # 1. Force lexicale : Jaccard Similarity entre le claim x_i et le chunk c_j
    intersection_xc = claim_tokens.intersection(chunk_tokens)
    union_xc = claim_tokens.union(chunk_tokens)
    sim_lexique = len(intersection_xc) / len(union_xc) if union_xc else 0.0
    
    # 2. Force sémantique : Alignement sémantique/lexical entre la Question q et le chunk c_j
    if question_tokens:
        intersection_qc = question_tokens.intersection(chunk_tokens)
        union_qc = question_tokens.union(chunk_tokens)
        sim_retriever = len(intersection_qc) / len(union_qc) if union_qc else 0.0
    else:
        sim_retriever = 0.0
        
    # Équation finale CiteFix
    return alpha * sim_lexique + (1 - alpha) * sim_retriever

def run_citefix_ksc_pipeline(claims: List[str], contexts: List[Dict], question: str, alpha: float, threshold: float) -> List[int]:
    """
    Parcourt l'ensemble des claims et des documents pour appliquer le filtrage par seuil t.
    """
    predicted_citations = set()
    question_tokens = _citefix_clean_tokens(question)
    
    for x_i in claims:
        x_i_tokens = _citefix_clean_tokens(x_i)
        if not x_i_tokens:
            continue
            
        for chunk in contexts:
            c_j_text = chunk['text']
            c_j_id = int(chunk['id'])
            c_j_tokens = _citefix_clean_tokens(c_j_text)
            
            # Calcul de la note hybride
            score = calculate_ksc_score(x_i_tokens, c_j_tokens, question_tokens, alpha)
            
            # Étape 4 de la méthodologie : Filtrage par seuil critique t
            if score >= threshold:
                predicted_citations.add(c_j_id)
                
    return sorted(list(predicted_citations))

















def run_citefix_fbs_pipeline(claims: List[str], contexts: List[Dict], threshold: float) -> List[int]:
    """
    Variante FBS : Utilise un modèle de re-ranking de type BERT Score local 
    pour prédire la probabilité d'implication logique de la paire (c_j, x_i).
    """
    try:
        from sentence_transformers import CrossEncoder
        # Modèle BERT compact (~200MB) spécialisé dans le scoring d'implication
        model = CrossEncoder('BAAI/bge-reranker-base')
    except ImportError:
        # Renvoie une liste vide si sentence-transformers n'est pas installé
        return []

    predicted_citations = set()
    
    for x_i in claims:
        # Construction des paires officielles (Document, Claim)
        pairs = [[chunk['text'], x_i] for chunk in contexts]
        if not pairs:
            continue
            
        # Le modèle BERT calcule la note d'implication
        scores = model.predict(pairs)
        
        for idx, score in enumerate(scores):
            c_j_id = int(contexts[idx]['id'])
            # Filtrage par seuil t appliqué à la sortie du modèle
            if score >= threshold:
                predicted_citations.add(c_j_id)
                
    return sorted(list(predicted_citations))















def execute_citefix_laboratory(parquet_input_path: str, alpha_grid: List[float] = [0.2, 0.5, 0.8, 1.0], threshold_grid: List[float] = [0.05, 0.1, 0.15, 0.2]):
    """
    Fait tourner la réplication exacte des expériences CiteFix en faisant 
    varier les hyperparamètres alpha et t sur l'intégralité du dataset stocké.
    """
    df_lab = pd.read_parquet(parquet_input_path)
    experiment_logs = []

    print(f"🔬 Analyse de {len(df_lab)} échantillons de réponses atomiques...")

    for index, row in df_lab.iterrows():
        # Étape 1 : Obtenir les propositions {x_i}
        claims = segment_answer_into_claims(row['generated_answer'])
        gold_ids = row['gold_ids']
        
        # --- RECONSTRUCTION EXPÉRIENCE KSC ---
        for alpha in alpha_grid:
            for t in threshold_grid:
                pred_ids_ksc = run_citefix_ksc_pipeline(
                    claims=claims,
                    contexts=row['contexts'],
                    question=row['question'],
                    alpha=alpha,
                    threshold=t
                )
                
                # Calcul des métriques via ton utilitaire fétiche
                metrics = calculate_metrics(gold_ids, pred_ids_ksc)
                
                experiment_logs.append({
                    "method": "KSC",
                    "alpha": alpha,
                    "threshold": t,
                    "precision": metrics['precision'],
                    "recall": metrics['recall'],
                    "f1": metrics['f1'],
                    "accuracy": metrics['accuracy']
                })

        # --- RECONSTRUCTION EXPÉRIENCE FBS ---
        # Seuil empirique ajusté pour les sorties logit de BGE Reranker
        for t_fbs in [0.0, 0.3, 0.5]:
            pred_ids_fbs = run_citefix_fbs_pipeline(claims=claims, contexts=row['contexts'], threshold=t_fbs)
            if pred_ids_fbs: # N'évalue que si sentence-transformers est présent
                metrics_fbs = calculate_metrics(gold_ids, pred_ids_fbs)
                experiment_logs.append({
                    "method": "FBS",
                    "alpha": np.nan,
                    "threshold": t_fbs,
                    "precision": metrics_fbs['precision'],
                    "recall": metrics_fbs['recall'],
                    "f1": metrics_fbs['f1'],
                    "accuracy": metrics_fbs['accuracy']
                })

    # Agrégation des résultats finaux sous forme de tableau macro (Mean Level Accuracy)
    df_results = pd.DataFrame(experiment_logs)
    macro_summary = df_results.groupby(["method", "alpha", "threshold"], dropna=False).mean()
    
    print("\n" + "="*20 + " TABLEAU MACRO DES EXPÉRIENCES CITEFIX " + "="*20)
    print(macro_summary.to_string())
    print("="*79 + "\n")
    
    return macro_summary










# exemple

summary_table = execute_citefix_laboratory("mes_resultats.parquet")











# NLI 

import re
import pandas as pd
import torch
from transformers import pipeline

# ==========================================
# 1. INITIALISATION DU NLI JUDGE (BART)
# ==========================================
nli_judge = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    device=0
)

def scan_context_with_nli_atomic(atomic_facts_list, context_list, threshold=0.6):
    """
    Vérifie si les chunks impliquent (entail) les faits atomiques.
    """
    predicted_ids = set()

    if not atomic_facts_list:
        return []

    for fact in atomic_facts_list:
        # On crée les paires [Chunk, Fait Atomique]
        pairs = [{"text": chunk['text'], "text_pair": fact} for chunk in context_list]

        # Inférence BART (on ajoute batch_size pour que ce soit très rapide)
        results = nli_judge(pairs, top_k=None, batch_size=32)

        for chunk, chunk_scores in zip(context_list, results):
            for score_dict in chunk_scores:
                if 'ENTAIL' in str(score_dict['label']).upper() and score_dict['score'] >= threshold:
                    predicted_ids.add(chunk['id'])
                    break 

    return list(predicted_ids)

# ==========================================
# 2. CHARGEMENT DE TON DATASET PARQUET
# ==========================================
# On charge le fichier généré par l'étape d'inférence précédente
df_saved = pd.read_parquet("ton_dataset_genere.parquet")

results_nli = []

# ==========================================
# 3. BOUCLE D'ÉVALUATION UNIQUEMENT
# ==========================================
for index, row in df_saved.iterrows():

    # --- ÉTAPE A : EXTRACTION DES FAITS ATOMIQUES (SANS LLM) ---
    # On découpe simplement le texte stocké grâce à tes tirets "-"
    raw_answer = row["generated_answer"]
    atomic_facts = [fact.strip("- *").strip() for fact in raw_answer.split('\n') if fact.strip() and fact.strip().startswith('-')]

    # Sécurité si une ligne n'a pas de tiret
    if not atomic_facts:
        atomic_facts = [fact.strip() for fact in raw_answer.split('\n') if fact.strip()]

    # --- ÉTAPE B : ÉVALUATION NLI AVEC BART ---
    pred_ids = scan_context_with_nli_atomic(atomic_facts, row["contexts"], threshold=0.6)

    # --- ÉTAPE C : METRICS (Ton code exact) ---
    gold_set = set(row["gold_ids"])
    pred_set = set(pred_ids)

    correct_ids = pred_set.intersection(gold_set)

    p = len(correct_ids) / len(pred_set) if pred_set else 0
    r = len(correct_ids) / len(gold_set) if gold_set else 0
    em = 1 if pred_set == gold_set else 0

    results_nli.append({"p": p, "r": r, "em": em})

    # --- AFFICHAGE ---
    print(f"--- SAMPLE {index + 1} ---")
    print(f"Q: {row['question']}")
    print(f"A (Stockée): {raw_answer}")
    print("Faits Atomiques extraits :")
    for f in atomic_facts:
        print(f"  - {f}")
    print(f"Gold IDs: {row['gold_ids']}")
    print(f"NLI Pred IDs: {pred_ids}")
    print(f"Verdict: {'✅' if em else '❌'} (P: {p:.2f}, R: {r:.2f})")
    print("-" * 30)

# ==========================================
# 4. BILAN FINAL (Ton code exact)
# ==========================================
total = len(results_nli)
print("\n" + "=" * 40)
print(f"BILAN FINAL NLI-SCAN ATOMIQUE BART ({total} SAMPLES)")
print("=" * 40)
print(f"Précision moyenne : {sum(m['p'] for m in results_nli) / total:.2%}")
print(f"Rappel moyen      : {sum(m['r'] for m in results_nli) / total:.2%}")
print(f"Exact Match Total : {sum(m['em'] for m in results_nli) / total:.2%}")