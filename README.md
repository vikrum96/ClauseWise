# ClauseWise

An end-to-end NLP system for legal contract analysis. Upload a contract, ask a question, and get a grounded answer backed by clause classification, information extraction, and LLM reasoning.

<img width="1547" height="276" alt="Screenshot 2026-05-28 141452" src="https://github.com/user-attachments/assets/8333c7bc-baee-4741-921b-5f6c6073287a" />


---

## Architecture

ClauseWise is deployed on AWS ECS Fargate with a FastAPI serving layer, S3 document storage, and CloudWatch monitoring. CI/CD is handled via GitHub Actions.
<img width="1086" height="627" alt="Screenshot 2026-05-28 143726" src="https://github.com/user-attachments/assets/2d881d26-6593-4ed8-b54f-a64130b3f98c" />


**Request flow:**
1. User uploads a contract (`.txt` or `.pdf`) and a question via the `/analyze` endpoint
2. Contract is saved to S3 and segmented into clauses via rule-based parsing
3. Clauses are encoded into dense embeddings and indexed with FAISS for semantic retrieval
4. Relevant clauses are selected and classified by LegalBERT (33-label multi-label)
5. FLAN-T5 + LoRA performs information extraction on the selected clauses
6. A reasoning prompt is assembled and sent to Llama 3.3 70B via Groq API
7. The grounded answer is returned to the user

---

## Models

| Model | Task | Hub ID |
|---|---|---|
| LegalBERT (fine-tuned) | 33-label multi-label clause classification | `ClauseWise/legalbert-clause-classifier` |
| FLAN-T5-base + LoRA (fine-tuned) | Clause information extraction | `ClauseWise/flan-t5-cuad-clause-extractor-lora` |
| all-MiniLM-L6-v2 | Semantic clause retrieval (FAISS) | `sentence-transformers/all-MiniLM-L6-v2` |
| Llama 3.3 70B (Groq) | Legal reasoning and answer generation | via Groq API |

**Training data:** [CUAD dataset](https://huggingface.co/datasets/theatticusproject/cuad) — 510 commercial contracts with 13,000+ expert annotations across 41 clause types.

**Evaluation results (held-out test set):**

| Model | Metric | Score |
|---|---|---|
| LegalBERT | Micro-F1 | 0.72 |
| FLAN-T5 + LoRA | ROUGE-L | 0.67 |

---

## Infrastructure

**CloudWatch Logs**
<img width="1637" height="246" alt="aws-3" src="https://github.com/user-attachments/assets/92910b28-f228-4cbc-9a93-40e28d4e1197" />

**CloudWatch Alarms**
<img width="1558" height="672" alt="aws-2" src="https://github.com/user-attachments/assets/866bb3b2-649a-432c-977d-d97812b1523f" />

**ECS Running Task**
<img width="1423" height="180" alt="aws-4" src="https://github.com/user-attachments/assets/4448065d-3371-4424-8a89-05ce2bdcfb89" />

**GitHub Actions**
<img width="1458" height="107" alt="aws-1" src="https://github.com/user-attachments/assets/8fc42d99-c1bc-49da-a639-e79b04151231" />


| Component | Service |
|---|---|
| Container registry | AWS ECR |
| Container orchestration | AWS ECS Fargate |
| Document storage | AWS S3 |
| Monitoring & logging | AWS CloudWatch |
| CI/CD | GitHub Actions |

**CI/CD:** Every push to `main` triggers an automated rebuild and zero-downtime redeployment via GitHub Actions. The workflow builds the Docker image, pushes it to ECR, and updates the ECS service.

**Monitoring:** CloudWatch alarms are configured for error rate thresholds and unexpected task termination, with SNS email notifications.

---

## Stack

**ML / AI:** PyTorch, HuggingFace Transformers, PEFT/LoRA, FAISS, sentence-transformers

**Backend:** FastAPI, Python, Docker

**Cloud / MLOps:** AWS ECS Fargate, ECR, S3, CloudWatch, GitHub Actions

**Data:** CUAD dataset, LegalBERT, FLAN-T5-base

---

## Running Locally

**Prerequisites:** Python 3.10.6, a [Groq API key](https://console.groq.com) (free)

```bash
git clone https://github.com/vikrum96/ClauseWise.git
cd ClauseWise
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Start the server:
```bash
python app.py
```

Test the endpoint:
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@samples/sample_contract.txt" \
  -F "question=What are the red flags in this contract?"
```

The interactive API docs are available at `http://localhost:8000/docs`.

---

## Reproducing Training

Model training requires the CUAD dataset. Download `master_clauses.csv` from the [Atticus Project CUAD dataset](https://huggingface.co/datasets/theatticusproject/cuad/tree/main/CUAD_v1) and place it in the `data/` directory (gitignored).

Open `notebooks/ClauseWise.ipynb` in Google Colab and run cells top-to-bottom. The notebook covers data preprocessing, LegalBERT fine-tuning, FLAN-T5 + LoRA fine-tuning, evaluation, and pushing trained weights to HuggingFace Hub.

---

## Project Structure

```
ClauseWise/
├── src/
│   ├── classifier.py      # LegalBERT multi-label clause classification
│   ├── extractor.py       # FLAN-T5 + LoRA clause information extraction
│   ├── pipeline.py        # Query routing and inference orchestration
│   ├── reasoner.py        # Groq API reasoning (Llama 3.3 70B)
│   ├── retriever.py       # FAISS semantic clause retrieval
│   └── segmenter.py       # Rule-based contract segmentation
├── notebooks/
│   └── ClauseWise.ipynb   # Training notebook
├── samples/
│   └── sample_contract.txt
├── app.py                 # FastAPI entry point
├── Dockerfile
└── requirements.txt
```


---

## Known Limitations
- Retrieval quality degrades on long contracts where the most important clauses appear in later sections.
  - FAISS returns semantically similar clauses but the segmenter indexes all 600+ clauses equally, so early introductory clauses can crowd out substantive ones.