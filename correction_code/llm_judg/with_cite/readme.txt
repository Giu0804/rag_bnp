# LLM-as-a-Judge: Context Attribution Evaluation (Inline Citations)

## Overview & Goal
This module evaluates the attribution and contextual fidelity of LLM-generated answers that contain strict inline citations (e.g., `[1, 2]`). Unlike the standard evaluation, this pipeline operates at the **sentence level (snippet granularity)**. 
The goal is to automatically parse the generated text into individual statements, query an ensemble of three reasoning LLMs to judge whether each specific statement is truly supported by the cited context chunks, and finally apply a Majority Vote consensus to validate the citations.

## Architecture
The architecture is designed to handle hierarchical data (sentences within answers) through a 3-step process:
1. **Regex Parsing**: Before any LLM evaluation, a utility function (`parse_inline_citations`) splits the raw text into distinct snippets using regular expressions, creating the `inline_processed` column.
2. **Snippet-Level Asynchronous Judging**: The `LLMJudge` engine iterates through every single snippet independently. It sends highly concurrent requests (`asyncio.Semaphore`) to three distinct LLM backends (vLLM, LLM-as-a-Service, and Qwen with `xhigh` reasoning) to predict which context IDs actually support the snippet.
3. **Snippet-Level Consensus**: The post-processing engine extracts the raw predictions, safely converts them into Python arrays using AST, and applies a Majority Vote (at least 2 out of 3 judges must agree) per snippet to determine the final accepted Ground Truth (`GT_cited`).

## File Structure
- `utils.py`: Contains the `parse_inline_citations` Regex function to split text into dictionaries containing `snippet_id`, `text`, and original `citations`.
- `llm_judge.py`: The core asynchronous execution engine. Contains the `LLMJudge` class and `run_judge_pipeline` which loops over the parsed snippets to evaluate them independently without losing the original row structure.
- `preannotation.py`: The consensus engine. Contains `convert_predictions_int_multi` for safe nested dictionary parsing, and `pre_annotation` for applying the majority vote logic on the snippet level.
- `notebook_judge_inline.ipynb`: The orchestrator notebook. It loads the dataset, applies the Regex parser, triggers the three judges sequentially, merges the nested predictions, and computes the final consensus.

## Requirements
Create a `requirements.txt` file in this directory with the following content:
```text
pandas
numpy
openai
requests
pyarrow