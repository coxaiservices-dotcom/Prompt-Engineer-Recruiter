
# Prompt Engineering Demo (Generic)

This repo is a **neutral, employer-agnostic** demo you can show in interviews. It demonstrates:
1) **Prompt Template Library** usage (render + discuss patterns)
2) **Multi-Turn Conversation** with simple session memory
3) **Tiny RAG** grounded on a small, generic corpus (no employer/JD references)

## Run (CLI)
```bash
python3 demo.py
```

## Notebook
```bash
jupyter notebook interview_demo.ipynb
```

- Runs offline by default using a Mock LLM (so it never breaks).
- You can wire in any provider later; code is intentionally minimal.

## Folders
- `data/context/` – generic policy + small knowledge base
- `data/templates/` – 16-pattern prompt library (JSON)
- `docs/` – your resume + Vanderbilt certificates

