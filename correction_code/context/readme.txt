# RAG Context Retriever Simulation

## Overview & Goal
This module simulates an advanced Retrieval-Augmented Generation (RAG) retriever. Its primary goal is to build customized context windows for LLM evaluation by mixing "Ground Truth" (GT) chunks with "Noise" (irrelevant or distracting) chunks. By strictly controlling the ratios of GT vs. Noise, this pipeline allows us to evaluate how well an LLM handles varying levels of context quality. 
It supports two distinct workflows: 
1. **Bot Data**: Un-scored raw chunks that require on-the-fly relevance scoring using a local Reranker model (Qwen).
2. **Public Data**: Pre-scored chunks that only require filtering and mathematical re-sampling.

## Architecture
The system is built around a centralized scoring and composition logic. For unscored data, it loads a local HuggingFace Reranker model to compute relevance scores, using manual batching to prevent GPU Out-Of-Memory (OOM) errors. Once scores are obtained (or if they are already present), the `ContextComposer` applies probabilistic rules (using `random` and `math`) to select the exact proportion of Ground Truth vs. Distractor chunks required by the user parameters.

## File Structure
- `bot/scorers.py`: Wrapper for the Qwen Reranker model. Handles tokenization, OOM-safe batch processing, and GPU memory management.
- `bot/context_composer.py`: The core mathematical engine. Contains the `ContextComposer` class that applies strategies (`PARTIAL_V2`, `RELEVANT`, `IRRELEVANT`) to mix chunks based on target ratios.
- `bot/utils.py`: Utility functions for flattening nested lists (`get_texts2`) and forcing GPU cache clearance.
- `bot/run_context_bot.py`: The pipeline for unscored data. It loads a `.pkl` corpus, extracts texts, scores them, and composes the contexts.
- `public_data/run_context_pub.py`: The pipeline for pre-scored data. Skips the LLM scoring step and directly applies the filtering/mixing logic.
- `notebook_context.ipynb`: The orchestrator notebook that runs both pipelines sequentially and saves the output to `.parquet` files.

## Requirements
Create a `requirements.txt` file in this directory with the following content:
```text
pandas
pydantic
transformers
torch
tqdm
pyarrow