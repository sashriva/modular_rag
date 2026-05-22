# Evaluation Harness

This folder contains a small starter evaluation setup for the current RAG pipeline.

## Contents

- `golden_set.json`: a small golden dataset with three samples
- `sample_docs/`: starter documents that match the golden dataset
- `results/`: saved evaluation reports

## Intended workflow

1. Ingest the sample docs first:

```powershell
python scripts/ingest_docs.py --docs eval/sample_docs
```

2. Run the evaluation harness:

```powershell
python scripts/run_eval.py
```

## Metrics

- `faithfulness`: LLM judge score for whether the answer is grounded in retrieved context
- `answer_relevancy`: LLM judge score for whether the answer addresses the user query
- `context_recall`: deterministic score based on whether expected source files were retrieved

## Notes

- The harness uses the same configured Groq model as the app for judging.
- Replace `golden_set.json` with your own curated examples once you index real documents.
