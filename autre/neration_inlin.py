import re
import pandas as pd

# ==============================================================================
# 1. CONFIGURATION DES PROMPTS (VARIABLES GLOBALES)
# ==============================================================================

SYS_PROMPT = (
    "Tu es un assistant de recherche rigoureux. Tu dois répondre à la question "
    "de l'utilisateur en te basant UNIQUEMENT sur les documents fournis.\n\n"
    "Règle de rédaction stricte :\n"
    "Tu dois rédiger ta réponse en utilisant exclusivement des phrases simples, "
    "courtes et directes. Chaque phrase ne doit contenir qu'un seul fait ou "
    "un seul élément de réponse (proposition atomique). Ne fais pas de phrases complexes."
)

USER_PROMPT = (
    "Voici les documents à ta disposition :\n{contexts}\n\n"
    "Question : {question}\n\n"
    "Consigne de formatage obligatoire :\n"
    "Ta réponse complète doit être obligatoirement entourée par les balises <answer> et </answer>.\n"
    "À la fin de CHAQUE phrase simple, juste avant le point final, insère "
    "l'identifiant du document qui prouve le fait énoncé sous la forme <chunk_id>X</chunk_id> (où X est le numéro du document).\n\n"
    "Exemple attendu :\n"
    "<answer>Le chiffrement RSA est un système asymétrique <chunk_id>14</chunk_id>. Cette méthode repose sur les nombres premiers <chunk_id>14</chunk_id>.</answer>\n\n"
    "Réponse :"
)

# ==============================================================================
# 2. FONCTIONS DE PARSING ET DE CALCUL DE MÉTRIQUES
# ==============================================================================

def parse_inline_atomic(full_generation):
    """
    Analyse la génération brute pour extraire la raw_output, clean_output, 
    les pred_ids uniques et les atomic_claims.
    """
    # Extraction du contenu à l'intérieur de <answer>...</answer>
    answer_match = re.search(r'<answer>(.*?)</answer>', full_generation, re.DOTALL)
    raw_output = answer_match.group(1).strip() if answer_match else full_generation.strip()
        
    # Découpage et extraction des propositions atomiques
    raw_claims = re.findall(r'(.*?<chunk_id>\d+</chunk_id>\.?)', raw_output)
    
    atomic_claims_list = []
    pred_ids_all = []
    
    for claim in raw_claims:
        id_match = re.search(r'<chunk_id>(\d+)</chunk_id>', claim)
        if id_match:
            pred_ids_all.append(int(id_match.group(1)))
            
        clean_claim = re.sub(r'<chunk_id>\d+</chunk_id>', '', claim).strip().strip('.')
        if clean_claim:
            atomic_claims_list.append(f"- {clean_claim}")
            
    atomic_claims_str = "\n".join(atomic_claims_list)
    
    # Suppression des doublons d'IDs en préservant leur ordre d'apparition
    pred_ids = list(dict.fromkeys(pred_ids_all))
    
    # Nettoyage de la réponse texte (clean_output)
    clean_answer = re.sub(r'\s*<chunk_id>\d+</chunk_id>', '', raw_output)
    clean_answer = re.sub(r'\s+\.', '.', clean_answer)
    
    return {
        "raw_output": raw_output,
        "clean_output": clean_answer,
        "pred_ids": pred_ids,
        "atomic_claims": atomic_claims_str
    }

def compute_metrics(gold_ids, pred_ids):
    """Calcule la précision, le rappel et le score F1 basés sur les listes d'IDs uniques."""
    gold_set = set(gold_ids)
    pred_set = set(pred_ids)
    
    if not pred_set:
        return 0.0, 0.0, 0.0
        
    tp = len(gold_set.intersection(pred_set))
    precision = tp / len(pred_set)
    recall = tp / len(gold_set) if len(gold_set) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

# ==============================================================================
# 3. FONCTION DE GESTION DE LA STRATÉGIE ET DE L'ÉVALUATION
# ==============================================================================

def evaluate_strategy(dataset, model_name, generate_ll):
    """
    Parcourt le dataset, appelle le LLM avec SYS_PROMPT et USER_PROMPT en paramètres,
    puis parse, affiche et sauvegarde l'évaluation en format DataFrame Parquet.
    """
    processed_records = []
    metrics_accumulator = []
    
    for idx, sample in enumerate(dataset):
        # Remplissage dynamique du patron du prompt utilisateur
        user_prompt_formated = USER_PROMPT.format(
            contexts=sample["contexts"], 
            question=sample["question"]
        )
        
        # Appel de ta fonction générique avec les prompts passés en paramètres
        full_generation = generate_ll(
            model_name=model_name,
            system_prompt=SYS_PROMPT,
            user_prompt=user_prompt_formated
        )
        
        gold_ids = sample["gold_ids"]
        
        # Traitement et extraction des données de la sortie LLM
        parsed_data = parse_inline_atomic(full_generation)
        pred_ids = parsed_data["pred_ids"]
        
        # Calcul des scores de l'échantillon courant
        prec, rec, f1 = compute_metrics(gold_ids, pred_ids)
        metrics_accumulator.append({"precision": prec, "recall": rec, "f1": f1})
        
        # Construction de l'enregistrement pour le fichier de sauvegarde final
        processed_records.append({
            "id": sample.get("id", idx),
            "question": sample["question"],
            "contexts": sample["contexts"],
            "gold_ids": gold_ids,
            "raw_output": parsed_data["raw_output"],
            "clean_output": parsed_data["clean_output"],
            "pred_ids": pred_ids,
            "atomic_claims": parsed_data["atomic_claims"]
        })
        
        # --- LE CLASSICO : AFFICHAGE ÉCHANTILLON PAR ÉCHANTILLON ---
        print(f"\n================================================================================")
        print(f"[ÉCHANTILLON {idx}] - MODÈLE : {model_name}")
        print(f"================================================================================")
        print(f"QUESTION      : {sample['question']}")
        print(f"--------------------------------------------------------------------------------")
        print(f"GOLD IDs      : {gold_ids}")
        print(f"PRED IDs      : {pred_ids}")
        print(f"MÉTRIQUES     : Précision: {prec:.2f} | Rappel: {rec:.2f} | F1: {f1:.2f}")
        print(f"--------------------------------------------------------------------------------")
        print(f"RAW OUTPUT    : {parsed_data['raw_output']}")
        print(f"--------------------------------------------------------------------------------")
        print(f"CLEAN OUTPUT  : {parsed_data['clean_output']}")
        print(f"--------------------------------------------------------------------------------")
        print(f"ATOMIC CLAIMS :\n{parsed_data['atomic_claims']}")
        print(f"================================================================================\n")
        
    # --- AFFICHAGE DE LA MOYENNE GLOBALE DES RÉSULTATS ---
    df_metrics = pd.DataFrame(metrics_accumulator)
    print("\n" + "#"*40)
    print(f"   MÉTRIQUES FINALES GLOBALES - {model_name}")
    print("#"*40)
    print(f"Précision Moyenne : {df_metrics['precision'].mean():.4f}")
    print(f"Rappel Moyen      : {df_metrics['recall'].mean():.4f}")
    print(f"Score F1 Moyen    : {df_metrics['f1'].mean():.4f}")
    print("#"*40 + "\n")
    
    # --- CRÉATION DU DATAFRAME ET SAUVEGARDE EN .PARQUET ---
    df_final = pd.DataFrame(processed_records)
    output_filename = f"rag_eval_{model_name}_inline_atomic.parquet"
    df_final.to_parquet(output_filename, index=False)
    print(f"DataFrame exporté avec succès vers le fichier : {output_filename}")
    
    return df_final


















def parse_inline_atomic(full_generation):
    """
    Analyse la génération brute pour extraire la raw_output, clean_output, 
    les pred_ids uniques et les atomic_claims (gère l'absence totale de chunk_id).
    """
    # 1. Isolation de la balise <answer>
    answer_match = re.search(r'<answer>(.*?)</answer>', full_generation, re.DOTALL)
    raw_output = answer_match.group(1).strip() if answer_match else full_generation.strip()
        
    # 2. Vérification de la présence de balises chunk_id
    has_chunks = bool(re.search(r'<chunk_id>\d+</chunk_id>', raw_output))
    
    atomic_claims_list = []
    pred_ids_all = []
    
    if has_chunks:
        # Découpage classique par balise(s)
        raw_claims = re.findall(r'(.*?(?:<chunk_id>\d+</chunk_id>)+\.?)', raw_output)
        for claim in raw_claims:
            ids_in_claim = re.findall(r'<chunk_id>(\d+)</chunk_id>', claim)
            for chunk_id in ids_in_claim:
                pred_ids_all.append(int(chunk_id))
                
            clean_claim = re.sub(r'<chunk_id>\d+</chunk_id>', '', claim).strip().strip('.')
            if clean_claim:
                atomic_claims_list.append(f"- {clean_claim}")
    else:
        # CAS SANS AUCUN CHUNK_ID : Découpage par les points finaux (.)
        # On sépare le texte à chaque phrase pour garder la structure atomique
        sentences = re.split(r'\.(?=\s|$)', raw_output)
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if clean_sentence:
                atomic_claims_list.append(f"- {clean_sentence}")
                
    atomic_claims_str = "\n".join(atomic_claims_list)
    
    # 3. Suppression des doublons d'IDs en gardant l'ordre
    pred_ids = list(dict.fromkeys(pred_ids_all))
    
    # 4. Nettoyage de la réponse texte globale (clean_output)
    clean_answer = re.sub(r'\s*<chunk_id>\d+</chunk_id>', '', raw_output)
    clean_answer = re.sub(r'\s+\.', '.', clean_answer)
    
    return {
        "raw_output": raw_output,
        "clean_output": clean_answer,
        "pred_ids": pred_ids,
        "atomic_claims": atomic_claims_str
    }




















def parse_inline_atomic(full_generation):
    """
    Analyse la génération brute pour extraire la raw_output, clean_output, 
    les pred_ids uniques et les atomic_claims (gère le format liste <chunk_id>X, Y</chunk_id>).
    """
    # 1. Isolation de la balise <answer>
    answer_match = re.search(r'<answer>(.*?)</answer>', full_generation, re.DOTALL)
    raw_output = answer_match.group(1).strip() if answer_match else full_generation.strip()
        
    # 2. Vérification de la présence de balises chunk_id
    has_chunks = bool(re.search(r'<chunk_id>.*?</chunk_id>', raw_output))
    
    atomic_claims_list = []
    pred_ids_all = []
    
    if has_chunks:
        # Découpage : cherche des morceaux de texte se terminant par une balise chunk_id
        raw_claims = re.findall(r'(.*?<chunk_id>.*?</chunk_id>\.?)', raw_output)
        for claim in raw_claims:
            # Extraction de la chaîne de caractères à l'intérieur des balises (ex: "14, 2")
            id_string_match = re.search(r'<chunk_id>(.*?)</chunk_id>', claim)
            if id_string_match:
                # On extrait tous les nombres présents dans cette chaîne
                ids_in_claim = re.findall(r'\d+', id_string_match.group(1))
                for chunk_id in ids_in_claim:
                    pred_ids_all.append(int(chunk_id))
                    
            # Nettoyage du claim pour enlever la balise chunk_id de la phrase
            clean_claim = re.sub(r'<chunk_id>.*?</chunk_id>', '', claim).strip().strip('.')
            if clean_claim:
                atomic_claims_list.append(f"- {clean_claim}")
    else:
        # CAS SANS AUCUN CHUNK_ID : Découpage classique par les points finaux (.)
        sentences = re.split(r'\.(?=\s|$)', raw_output)
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if clean_sentence:
                atomic_claims_list.append(f"- {clean_sentence}")
                
    atomic_claims_str = "\n".join(atomic_claims_list)
    
    # 3. Suppression des doublons d'IDs globaux en préservant l'ordre d'apparition
    pred_ids = list(dict.fromkeys(pred_ids_all))
    
    # 4. Nettoyage de la réponse texte globale (clean_output)
    clean_answer = re.sub(r'\s*<chunk_id>.*?</chunk_id>', '', raw_output)
    clean_answer = re.sub(r'\s+\.', '.', clean_answer)