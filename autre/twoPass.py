import re
import gc
import torch
import pandas as pd

# ==========================================
# 1. UTILITAIRES TECHNIQUES
# ==========================================

def clean_memory():
    """Vide le cache mémoire GPU et force le Garbage Collector."""

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def generate_llm_response(
    model,
    tokenizer,
    system_prompt,
    user_prompt,
    max_tokens=350
):
    """Génère la réponse en séparant bien le System et le User."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            do_sample=False
        )

    raw_output = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    del inputs, outputs
    clean_memory()

    return raw_output


# ==========================================
# 2. PARSEURS ET MÉTRIQUES
# ==========================================

def extract_ids(raw_output):
    """Extrait uniquement la liste des IDs depuis la balise <chunks_id>."""

    chunks_match = re.search(
        r"<chunks_id>(.*?)</chunks_id>",
        raw_output,
        re.DOTALL
    )

    if chunks_match:
        return [int(n) for n in re.findall(r"\d+", chunks_match.group(1))]

    return []


def extract_answer(raw_output):
    """Extrait uniquement le texte depuis la balise <answer>."""

    answer_match = re.search(
        r"<answer>(.*?)</answer>",
        raw_output,
        re.DOTALL
    )

    return (
        answer_match.group(1).strip()
        if answer_match
        else "Format Error (Answer)"
    )


def calculate_metrics(gold_ids, predicted_ids):
    """Calcule Précision, Rappel, F1 et Accuracy."""

    gold_set = set(gold_ids)
    pred_set = set(predicted_ids)

    correct_ids = pred_set.intersection(gold_set)

    p = len(correct_ids) / len(pred_set) if len(pred_set) > 0 else 0.0
    r = len(correct_ids) / len(gold_set) if len(gold_set) > 0 else 0.0

    f1 = (
        2 * (p * r) / (p + r)
        if (p + r) > 0
        else 0.0
    )

    accuracy = 1.0 if pred_set == gold_set else 0.0

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "accuracy": accuracy
    }


# ==========================================
# =========================================================
#           CITE -> THEN -> ANSWER
# =========================================================
# ==========================================

# =========================================================
# PASS 1 : CHUNK SELECTION
# =========================================================

def predict_relevant_chunks(
    dataset,
    sys_prompt,
    user_prompt_template,
    model,
    tokenizer,
    sample_size=None,
    output_filename=None
):
    """
    PASS 1 :
    Sélection des chunks pertinents.
    """

    results = []

    df_sample = (
        dataset.head(sample_size)
        if sample_size is not None
        else dataset
    )

    for index, row in df_sample.iterrows():

        # ==========================================
        # Construction du contexte complet
        # ==========================================

        context_text = "\n".join([
            f"ID: {p['id']} | Paragraph: {p['text']}"
            for p in row["contexts"]
        ])

        formatted_user_prompt = user_prompt_template.format(
            context_text=context_text,
            question=row["question"]
        )

        # ==========================================
        # Inférence
        # ==========================================

        raw_output = generate_llm_response(
            model=model,
            tokenizer=tokenizer,
            system_prompt=sys_prompt,
            user_prompt=formatted_user_prompt
        )

        # ==========================================
        # Parsing
        # ==========================================

        predicted_ids = extract_ids(raw_output)

        # ==========================================
        # Évaluation
        # ==========================================

        metrics = calculate_metrics(
            row["gold_ids"],
            predicted_ids
        )

        # ==========================================
        # Sauvegarde
        # ==========================================

        results.append({
            "sample_id": row["id"],
            "question": row["question"],
            "contexts": row["contexts"],
            "gold_ids": row["gold_ids"],
            "predicted_ids": predicted_ids,
            **metrics
        })

        # ==========================================
        # PRINTS EXACTEMENT COMME TON CODE
        # ==========================================

        print(f"--- SAMPLE {index+1} (ID: {row['id']}) ---")
        print(f"Question : {row['question']}")
        print(f"Pred IDs : {predicted_ids} | Gold IDs : {row['gold_ids']}")
        print(f"Verdict  : {'✅ Exact' if metrics['accuracy'] else '❌ Différent'}")

        print(
            f"Métriques: "
            f"Précision: {metrics['precision']:.2f} | "
            f"Rappel: {metrics['recall']:.2f} | "
            f"F1: {metrics['f1']:.2f}"
        )

        print("-" * 50)

    # ==========================================
    # Agrégation finale EXACTEMENT COMME TON CODE
    # ==========================================

    df_results = pd.DataFrame(results)

    avg_metrics = df_results[
        ["precision", "recall", "f1", "accuracy"]
    ].mean().to_dict()

    print("\n" + "="*20 + " RÉSULTATS GLOBAUX " + "="*20)

    for k, v in avg_metrics.items():
        print(f"Moyenne {k.capitalize()} : {v:.4f}")

    print("="*59 + "\n")

    # ==========================================
    # Sauvegarde
    # ==========================================

    if output_filename:

        df_results.to_csv(
            output_filename,
            index=False,
            encoding="utf-8"
        )

        print(f"💾 Sauvegardé dans : {output_filename}")

    return df_results, avg_metrics


# =========================================================
# BUILD DATASET FILTRÉ
# =========================================================

def build_filtered_context_dataset(prediction_df):
    """
    Construit un dataset contenant UNIQUEMENT
    les chunks sélectionnés.
    """

    filtered_rows = []

    for _, row in prediction_df.iterrows():

        predicted_ids = set(row["predicted_ids"])

        filtered_contexts = [
            ctx
            for ctx in row["contexts"]
            if ctx["id"] in predicted_ids
        ]

        filtered_rows.append({
            "sample_id": row["sample_id"],
            "question": row["question"],
            "selected_contexts": filtered_contexts,
            "predicted_ids": row["predicted_ids"],
            "gold_ids": row["gold_ids"]
        })

    return pd.DataFrame(filtered_rows)


# =========================================================
# PASS 2 : ANSWER GENERATION
# =========================================================

def generate_answers_from_selected_contexts(
    filtered_dataset,
    sys_prompt,
    user_prompt_template,
    model,
    tokenizer,
    sample_size=None,
    output_filename=None
):
    """
    PASS 2 :
    Génération de réponse UNIQUEMENT
    avec les chunks sélectionnés.
    """

    results = []

    df_sample = (
        filtered_dataset.head(sample_size)
        if sample_size is not None
        else filtered_dataset
    )

    for index, row in df_sample.iterrows():

        # ==========================================
        # Construction du contexte filtré
        # ==========================================

        context_text = "\n".join([
            f"ID: {p['id']} | Paragraph: {p['text']}"
            for p in row["selected_contexts"]
        ])

        formatted_user_prompt = user_prompt_template.format(
            context_text=context_text,
            question=row["question"]
        )

        # ==========================================
        # Inférence
        # ==========================================

        raw_output = generate_llm_response(
            model=model,
            tokenizer=tokenizer,
            system_prompt=sys_prompt,
            user_prompt=formatted_user_prompt
        )

        # ==========================================
        # Parsing
        # ==========================================

        predicted_answer = extract_answer(raw_output)

        # ==========================================
        # Sauvegarde
        # ==========================================

        results.append({
            "sample_id": row["sample_id"],
            "question": row["question"],
            "predicted_ids": row["predicted_ids"],
            "selected_contexts": row["selected_contexts"],
            "generated_answer": predicted_answer
        })

        # ==========================================
        # PRINTS
        # ==========================================

        print(f"--- SAMPLE {index+1} (ID: {row['sample_id']}) ---")
        print(f"Question : {row['question']}")
        print(f"Réponse  : {predicted_answer}")
        print("-" * 50)

    # ==========================================
    # DataFrame final
    # ==========================================

    df_results = pd.DataFrame(results)

    # ==========================================
    # Sauvegarde
    # ==========================================

    if output_filename:

        df_results.to_csv(
            output_filename,
            index=False,
            encoding="utf-8"
        )

        print(f"💾 Sauvegardé dans : {output_filename}")

    return df_results


# ==========================================
# =========================================================
#           ANSWER -> THEN -> CITE
# =========================================================
# ==========================================

# =========================================================
# PASS 1 : ANSWER GENERATION
# =========================================================

def generate_answers_full_context(
    dataset,
    sys_prompt,
    user_prompt_template,
    model,
    tokenizer,
    sample_size=None,
    output_filename=None
):
    """
    PASS 1 :
    Génération de réponse avec contexte complet.
    """

    results = []

    df_sample = (
        dataset.head(sample_size)
        if sample_size is not None
        else dataset
    )

    for index, row in df_sample.iterrows():

        # ==========================================
        # Construction du contexte complet
        # ==========================================

        context_text = "\n".join([
            f"ID: {p['id']} | Paragraph: {p['text']}"
            for p in row["contexts"]
        ])

        formatted_user_prompt = user_prompt_template.format(
            context_text=context_text,
            question=row["question"]
        )

        # ==========================================
        # Inférence
        # ==========================================

        raw_output = generate_llm_response(
            model=model,
            tokenizer=tokenizer,
            system_prompt=sys_prompt,
            user_prompt=formatted_user_prompt
        )

        # ==========================================
        # Parsing
        # ==========================================

        predicted_answer = extract_answer(raw_output)

        # ==========================================
        # Sauvegarde
        # ==========================================

        results.append({
            "sample_id": row["id"],
            "question": row["question"],
            "contexts": row["contexts"],
            "gold_ids": row["gold_ids"],
            "generated_answer": predicted_answer
        })

        # ==========================================
        # PRINTS
        # ==========================================

        print(f"--- SAMPLE {index+1} (ID: {row['id']}) ---")
        print(f"Question : {row['question']}")
        print(f"Réponse  : {predicted_answer}")
        print("-" * 50)

    # ==========================================
    # DataFrame final
    # ==========================================

    df_results = pd.DataFrame(results)

    # ==========================================
    # Sauvegarde
    # ==========================================

    if output_filename:

        df_results.to_csv(
            output_filename,
            index=False,
            encoding="utf-8"
        )

        print(f"💾 Sauvegardé dans : {output_filename}")

    return df_results


# =========================================================
# PASS 2 : CITATION
# =========================================================

def cite_generated_answers(
    dataset_with_answers,
    sys_prompt,
    user_prompt_template,
    model,
    tokenizer,
    sample_size=None,
    output_filename=None
):
    """
    PASS 2 :
    Le modèle reçoit :
    - question
    - contexte complet
    - réponse générée

    Puis prédit les citations.
    """

    results = []

    df_sample = (
        dataset_with_answers.head(sample_size)
        if sample_size is not None
        else dataset_with_answers
    )

    for index, row in df_sample.iterrows():

        # ==========================================
        # Construction du contexte complet
        # ==========================================

        context_text = "\n".join([
            f"ID: {p['id']} | Paragraph: {p['text']}"
            for p in row["contexts"]
        ])

        formatted_user_prompt = user_prompt_template.format(
            context_text=context_text,
            question=row["question"],
            generated_answer=row["generated_answer"]
        )

        # ==========================================
        # Inférence
        # ==========================================

        raw_output = generate_llm_response(
            model=model,
            tokenizer=tokenizer,
            system_prompt=sys_prompt,
            user_prompt=formatted_user_prompt
        )

        # ==========================================
        # Parsing
        # ==========================================

        predicted_ids = extract_ids(raw_output)

        # ==========================================
        # Évaluation
        # ==========================================

        metrics = calculate_metrics(
            row["gold_ids"],
            predicted_ids
        )

        # ==========================================
        # Sauvegarde
        # ==========================================

        results.append({
            "sample_id": row["sample_id"],
            "question": row["question"],
            "generated_answer": row["generated_answer"],
            "gold_ids": row["gold_ids"],
            "predicted_ids": predicted_ids,
            **metrics
        })

        # ==========================================
        # PRINTS EXACTEMENT COMME TON CODE
        # ==========================================

        print(f"--- SAMPLE {index+1} (ID: {row['sample_id']}) ---")
        print(f"Question : {row['question']}")
        print(f"Réponse  : {row['generated_answer']}")
        print(f"Pred IDs : {predicted_ids} | Gold IDs : {row['gold_ids']}")
        print(f"Verdict  : {'✅ Exact' if metrics['accuracy'] else '❌ Différent'}")

        print(
            f"Métriques: "
            f"Précision: {metrics['precision']:.2f} | "
            f"Rappel: {metrics['recall']:.2f} | "
            f"F1: {metrics['f1']:.2f}"
        )

        print("-" * 50)

    # ==========================================
    # Agrégation finale EXACTEMENT COMME TON CODE
    # ==========================================

    df_results = pd.DataFrame(results)

    avg_metrics = df_results[
        ["precision", "recall", "f1", "accuracy"]
    ].mean().to_dict()

    print("\n" + "="*20 + " RÉSULTATS GLOBAUX " + "="*20)

    for k, v in avg_metrics.items():
        print(f"Moyenne {k.capitalize()} : {v:.4f}")

    print("="*59 + "\n")

    # ==========================================
    # Sauvegarde
    # ==========================================

    if output_filename:

        df_results.to_csv(
            output_filename,
            index=False,
            encoding="utf-8"
        )

        print(f"💾 Sauvegardé dans : {output_filename}")

    return df_results, avg_metrics


# ==========================================
# EXEMPLES D'UTILISATION
# ==========================================

# =========================================================
# CITE -> THEN -> ANSWER
# =========================================================

"""
# ==========================================
# PASS 1 : CHUNK SELECTION
# ==========================================

df_citations, metrics_citations = predict_relevant_chunks(
    dataset=ds_rag,
    sys_prompt=sys_prompt_citation,
    user_prompt_template=user_prompt_citation,
    model=citation_model,
    tokenizer=citation_tokenizer,
    sample_size=10,
    output_filename="pass1_chunk_selection.csv"
)

# ==========================================
# BUILD FILTERED DATASET
# ==========================================

df_filtered = build_filtered_context_dataset(
    df_citations
)

# ==========================================
# PASS 2 : ANSWER GENERATION
# ==========================================

df_answers = generate_answers_from_selected_contexts(
    filtered_dataset=df_filtered,
    sys_prompt=sys_prompt_answer,
    user_prompt_template=user_prompt_answer,
    model=answer_model,
    tokenizer=answer_tokenizer,
    output_filename="pass2_answers.csv"
)
"""


# =========================================================
# ANSWER -> THEN -> CITE
# =========================================================

"""
# ==========================================
# PASS 1 : ANSWER GENERATION
# ==========================================

df_generated_answers = generate_answers_full_context(
    dataset=ds_rag,
    sys_prompt=sys_prompt_answer,
    user_prompt_template=user_prompt_answer,
    model=answer_model,
    tokenizer=answer_tokenizer,
    sample_size=10,
    output_filename="pass1_generated_answers.csv"
)

# ==========================================
# PASS 2 : CITATION
# ==========================================

df_citations, metrics_citations = cite_generated_answers(
    dataset_with_answers=df_generated_answers,
    sys_prompt=sys_prompt_citation,
    user_prompt_template=user_prompt_citation_after_answer,
    model=citation_model,
    tokenizer=citation_tokenizer,
    output_filename="pass2_citations.csv"
)
"""