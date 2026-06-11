# Project Improvement Plan
## Lecture Audio Summarization and Key Information Extraction

---

## Current State — What Works, What Doesn't

### What Works
- Whisper transcription pipeline (audio to text)
- spaCy text cleaning (stopword removal, lemmatization, POS tagging, NER)
- YAKE keyword extraction
- Basic Gradio UI

### What's Broken or Weak
| Issue | Location | Impact |
|---|---|---|
| `small_summary` is hardcoded string | `cell 18` | Completely fake output |
| Teacher Name / Due Date = `[Extracted Name]` | `cell 12, 17, 18` | Placeholders, never extracted |
| pyTextRank just reorders sentences | `cell 15–17` | No real understanding |
| T5 summary depends on pre-generated files | `cell 12` | Pipeline breaks without them |
| Single-direction output only | entire notebook | No user interactivity after report |

---

## Phase 1 — Fix the Broken Foundation
**Goal:** Replace placeholder logic and weak summarization with real NLP models.
**Estimated time:** 1–2 days

### 1.1 Abstractive Summarization with BART
**Replace:** `pyTextRank` extractive summarization
**Use:** `facebook/bart-large-cnn` from HuggingFace Transformers

```python
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
```

- Generates new sentences, not just ranked existing ones
- Understands context, produces coherent paragraph-level summaries
- Runs on free Colab T4 GPU

### 1.2 Structured Field Extraction with Flan-T5
**Replace:** All `[Extracted Name]`, `[Due Date]`, `[Quiz Date]` placeholders
**Use:** `google/flan-t5-base` with instruction prompts

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

def extract_field(transcript, question):
    prompt = f"Answer the question from the lecture transcript.\nTranscript: {transcript[:1000]}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

teacher_name = extract_field(text, "What is the name of the teacher or professor?")
quiz_date    = extract_field(text, "When is the next quiz or exam?")
due_date     = extract_field(text, "What is the assignment due date?")
```

- Finally extracts real values instead of placeholders
- This is the T5 step referenced in cell 12 but never actually implemented

### 1.3 Dynamic Small Summary
**Replace:** Hardcoded `small_summary` string
**Use:** BART summarizer with `max_length=60` for a one-sentence recap

### Deliverable for Phase 1
A notebook cell that takes a transcript and returns a fully filled structured report — no placeholders.

---

## Phase 2 — Add RAG (The Killer Feature)
**Goal:** Allow users to ask questions across all their lecture transcripts.
**Estimated time:** 2–3 days

### Why RAG Changes Everything
Instead of just producing a one-time report, the user can now interact:
- "What did the teacher say about mitosis?"
- "Which lecture covered calcium carbonate?"
- "List all assignments from all lectures"
- "Summarize everything from the November lectures"

### 2.1 Chunking
Split each transcript into overlapping chunks for better retrieval:

```python
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
```

### 2.2 Embeddings
**Use:** `sentence-transformers/all-MiniLM-L6-v2` — fast, lightweight, high quality

```python
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# For each chunk across all transcripts
embeddings = embed_model.encode(all_chunks)
```

### 2.3 Vector Store with FAISS
**Use:** FAISS — no server required, pure Python, free

```python
import faiss
import numpy as np

dimension = embeddings.shape[1]  # 384 for MiniLM
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index for reuse
faiss.write_index(index, "lecture_index.faiss")
```

### 2.4 Retrieval + Generation (RAG Loop)
```python
def ask_lectures(question, top_k=3):
    query_embedding = embed_model.encode([question])
    distances, indices = index.search(np.array(query_embedding), top_k)

    retrieved_chunks = [all_chunks[i] for i in indices[0]]
    context = " ".join(retrieved_chunks)

    prompt = f"Answer the question using the lecture notes below.\nLecture Notes: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 2.5 Source Attribution
Track which lecture each chunk came from so answers show their source:

```python
chunk_metadata = []  # list of {"chunk": "...", "source": "MarChem9_28edit.txt", "chunk_id": 3}
```

### Libraries Needed for Phase 2
```
pip install sentence-transformers faiss-cpu
```

### Deliverable for Phase 2
A `ask_lectures("your question")` function that retrieves relevant lecture content and generates a grounded answer with source attribution.

---

## Phase 3 — Polish the UI and Output
**Goal:** Make the project presentable, interactive, and demo-ready.
**Estimated time:** 1 day

### 3.1 Upgrade Gradio to Multi-Tab Layout

```python
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Tab("Upload & Transcribe"):
        audio_input = gr.Audio(type="filepath", label="Upload Lecture Audio (.mp3)")
        transcribe_btn = gr.Button("Transcribe")
        transcript_output = gr.Textbox(label="Transcript", lines=10)

    with gr.Tab("Generate Report"):
        subject_input = gr.Textbox(label="Subject Name")
        report_btn = gr.Button("Generate Structured Report")
        report_output = gr.Textbox(label="Structured Report", lines=20)

    with gr.Tab("Ask Your Lectures (RAG)"):
        question_input = gr.Textbox(label="Ask a question about any lecture")
        ask_btn = gr.Button("Ask")
        answer_output = gr.Textbox(label="Answer")
        source_output = gr.Textbox(label="Source Lecture")
```

### 3.2 Export Report as Markdown File
Add a download button in Gradio that lets the user save the structured report as a `.md` file.

```python
def save_report(report_text, filename="lecture_report.md"):
    with open(filename, "w") as f:
        f.write(report_text)
    return filename

gr.File(label="Download Report")
```

### 3.3 Visualizations to Add
- **Word cloud** from the transcript (replace or supplement the bar chart)
- **Topic timeline**: which topics appeared in which lecture (using lecture filenames as x-axis)
- **Keyword heatmap** across multiple lectures

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(cleaned_text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```

### 3.4 Speaker Diarization (Bonus — Advanced)
Identify teacher vs student speech segments before transcription using `pyannote.audio`:

```python
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
diarization = pipeline("lecture.mp3")
```

- Label segments as `TEACHER` or `STUDENT`
- Only summarize teacher segments
- This requires a free HuggingFace token

### Deliverable for Phase 3
A polished, demo-ready Gradio app with 3 tabs: Upload, Report, and Q&A — plus downloadable output.

---

## Full Stack Summary

| Component | Current | Upgraded |
|---|---|---|
| Audio Transcription | Whisper medium | Whisper medium (keep) |
| Text Cleaning | NLTK + spaCy sm | NLTK + spaCy lg (keep) |
| Keyword Extraction | YAKE | YAKE (keep) + semantic topics |
| Summarization | pyTextRank (extractive) | BART-large-CNN (abstractive) |
| Field Extraction | Regex + placeholders | Flan-T5 with prompts |
| Q&A | None | RAG: MiniLM + FAISS + Flan-T5 |
| UI | Single Gradio input/output | Multi-tab Gradio Blocks |
| Export | None | Markdown download |
| Visualization | Bar chart | Bar chart + Word cloud |

---

## Install Requirements (all phases)

```bash
pip install transformers torch accelerate
pip install sentence-transformers faiss-cpu
pip install wordcloud
pip install gradio>=4.0
# Already installed: whisper, spacy, yake, nltk, pytextrank
```

---

## Priority Order

1. **Phase 1 first** — fixes broken outputs, makes existing report credible
2. **Phase 2 second** — adds the RAG Q&A, the most impressive new feature
3. **Phase 3 last** — polish after the core is working

---

## Notes
- All models listed run on free Google Colab T4 GPU
- FAISS index can be saved to Drive and reloaded between sessions
- Flan-T5-base is small enough to run on Colab CPU if needed (slower)
- For better generation quality, swap Flan-T5 for `mistralai/Mistral-7B-Instruct-v0.2` on Colab Pro
