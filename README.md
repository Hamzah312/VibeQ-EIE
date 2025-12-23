# Emotion Intelligence Engine (EIE)

## Overview

**Emotion Intelligence Engine (EIE)** is a research-driven system for **emotion understanding in first-person journaling text**.  
The project focuses on building a **high-quality emotion classifier** through **multi-stage training, domain adaptation, and teacher–student distillation**.

EIE is designed to serve as a **core emotional understanding layer** for downstream systems such as coping-action recommendation (via RAG), safety filtering, and emotional intervention logic.

> **Core principle:** robust emotion understanding — *not* text generation.

---

## Goals

1. Build a **strong base emotion classifier** with high macro and micro F1.
2. Adapt the model from generic emotion datasets to **journaling-style language**.
3. Apply **teacher–student distillation** using state-of-the-art LLMs.
4. Preserve generalization on benchmark datasets while improving domain performance.
5. Enable reliable downstream usage (RAG, safety filtering, intervention suggestions).

---

## Target Emotion Space

The system operates on **9 core emotions**:

- anger  
- anticipation  
- caring  
- disgust  
- fear  
- joy  
- neutral  
- sadness  
- surprise  

This reduced emotion space was selected to balance:

- Psychological validity  
- Model stability  
- Downstream usability  

---

## Datasets

### 1. GoEmotions (Google)

- Large-scale, high-quality human-annotated dataset  
- Used as the **anchor dataset**  
- Provides strong general emotion grounding  

---

### 2. ISEAR

- Psychologically grounded emotion reports  
- Original labels mapped into the 9-emotion space  
- Relabeled using an LLM to produce:
  - Soft emotion distributions  
  - Valence / Arousal / Dominance (VAD) scores  
- Used for domain adaptation toward **introspective emotional text**

---

### 3. Synthetic Journaling Data

- Generated using a SOTA LLM (Qwen)
- Emotion-controlled generation
- Labeled by a second LLM acting as a **teacher**
- Each sample includes:
  - Primary emotion
  - Secondary emotions
  - Full probability distribution over emotions
  - Valence / Arousal / Dominance (VAD)
  - Teacher confidence score

---

## Project Structure

### Core Notebooks

#### Emotion Classifier Experiments

- `01_emotion_classifier_exp1.ipynb`
- `01_emotion_classifier_exp2.ipynb`
- `01_emotion_classifier_exp3.ipynb`

Baseline and improved classifiers trained on **GoEmotions**, using:

- DeBERTa-v3-large
- Multi-label and single-label heads
- Cross-entropy loss
- Focal loss (for class imbalance)

---

#### Domain Adaptation

- `Domain_adaptation_exp1.ipynb`
- `Domain_adaptation_exp2.ipynb`

Experiments combining:

- GoEmotions
- ISEAR
- Synthetic journaling data

These experiments revealed **performance regression caused by equal-trust training**, motivating the final distillation-based approach.

---

#### Evaluation

- `Eval_domain_adapt_exp1_1.ipynb`
- `Eval_domain_adapt_exp1_2.ipynb`
- `Eval_domain_adapt_exp2_1.ipynb`
- `Eval_domain_adapt_exp2_2.ipynb`

Evaluation includes:

- Per-class F1 scores
- Confusion matrices
- Dataset-specific evaluation
- Regression detection on GoEmotions

---

#### Data Generation & Annotation

- `Journal_Generator_exp1.ipynb`
- `Journal_Generator_exp2.ipynb`
- `Journal_Generator_exp3.ipynb`
- `ISEAR_Annotator_exp1.ipynb`
- `ISEAR_Annotator_exp2.ipynb`

LLM-based generation and annotation with:

- Strict schema validation
- Emotion distribution enforcement
- VAD consistency checks

---

#### RAG
- `RAG_Qdrant_Index.ipynb` – preparation for downstream action retrieval
---

#### Final pipeline with Langraph
- `Pipeline.ipynb` – end-to-end experiment orchestration


## Training Strategy

### Phase 1 – Anchor Training

- Train the classifier on **GoEmotions only**
- Establish strong general emotion understanding
- Serves as the performance baseline

---

### Phase 2 – Domain Exposure

- Introduce journaling-style data and ISEAR
- Analyze domain shift effects
- Identify performance regressions caused by naive mixing

---

### Phase 3 – Teacher–Student Distillation

- **Teacher:** SOTA LLM producing soft emotion distributions
- **Student:** DeBERTa-based classifier

**Loss components:**

- Cross-Entropy loss (hard labels, anchor data)
- KL-Divergence loss (soft labels, teacher data)
- Dataset-aware loss weighting (no equal trust)