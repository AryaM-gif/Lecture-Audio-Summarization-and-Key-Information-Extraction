# Lecture Audio Summarization and Key Information Extraction

##  Project Goal

The main objective of this project is to process audio transcripts of class lectures and generate:

* Structured summaries
* Extracted key information (topics, assignments, quizzes)
* Brief, readable overviews

This is achieved through a combination of audio transcription, text preprocessing, NLP techniques, and visualization.

---

##  Workflow and Techniques

### 1. Mount Google Drive

* **Library**: `google.colab.drive`
* **Why**: Access audio files stored in Drive and store outputs (transcripts, summaries, reports) persistently.

### 2. Audio Transcription with Whisper

* **Model**: `whisper.load_model("medium")`
* **Technique**: Transcribes `.mp3` lecture recordings into raw text.
* **Why**: Forms the foundation for all downstream NLP tasks.

### 3. Text Cleaning and Annotation

**Libraries**: `NLTK`, `spaCy`

* **Steps**:

  * Lowercasing and punctuation removal
  * Tokenization
  * Stopword removal (NLTK)
  * Lemmatization (NLTK WordNet)
  * POS tagging (spaCy)
  * NER (spaCy)
* **Why**: Normalize the transcript and annotate linguistic features for better extraction and summarization.

### 4. Word Frequency Analysis

* **Library**: `collections.Counter`, `matplotlib`
* **Why**: Identify dominant themes or concepts via most frequent words in the cleaned transcript.

### 5. Keyword Extraction with YAKE

* **Library**: `yake`
* **Why**: Extract relevant keywords and keyphrases using unsupervised statistical features of words.

### 6. Structured Report Generation (with YAKE + Placeholder T5)

* **Technique**: Combine YAKE keywords and (optional) T5 summaries into a unified report format.
* **Why**: Create a readable and structured summary per lecture, including extracted topics.

### 7. Enhanced Report Generation (with pyTextRank + Heuristics)

* **Libraries**: `spaCy`, `pyTextRank`, `re`
* **Techniques**:

  * **Summarization**: Using `pyTextRank` (TextRank-based graph algorithm)
  * **Topic Extraction**: Noun chunks and entities + blacklist filtering
  * **Assignment & Quiz Extraction**: Regex-based sentence matching
* **Why**: Produce a robust structured report including:

  * Main topics
  * Summary
  * Assignment description & due date
  * Quiz details

### 8. Gradio Interface

* **Library**: `Gradio`
* **Why**: Simplifies usage by providing a web interface to input transcripts and generate reports on the fly.


###  Problem Statement

"Imagine you have hours of recorded lectures and no time to review. This project automatically transcribes, summarizes, and extracts key lecture information."

###  Solution

"An AI-based pipeline that turns lecture audio into structured summaries and highlights quizzes, assignments, and main topics."

###  Key Steps to Explain:

1. **Audio Transcription**: Using Whisper to convert voice into text.
2. **Text Cleaning**: Preprocessing to remove noise (punctuation, stopwords, etc.)
3. **NLP Analysis**: POS tagging, lemmatization, NER.
4. **Keyword Extraction**: YAKE and noun-chunk-based topic detection.
5. **Structured Summary Generation**:

   * Summarization with PyTextRank
   * Quiz/assignment detection using regular expressions
6. **Report Output**: A formatted report with all extracted fields.
7. **Gradio UI**: Easy interface for uploading/pasting text to get reports.

###  Techniques & Tools Used

* **Whisper (OpenAI)**: Audio-to-text
* **NLTK & spaCy**: Text cleaning, annotation
* **YAKE**: Keyword extraction
* **PyTextRank**: Extractive summarization
* **Regular Expressions**: Pattern-based extraction (assignments, quizzes)
* **Gradio**: Interactive UI

###  Benefits

* Saves hours of manual note-taking
* Highlights crucial academic content
* Easy-to-use with Gradio frontend

###  Show a Demo (Optional)

* Paste a transcript
* Click "Generate Report"
* Instantly get: Topics, Summary, Assignments, and Quiz Info
---

##  Future Improvements

* Integrate live T5 summarization
* Support multiple languages
* Add date detection using advanced NLP (e.g., SUTime)
* Export reports as PDF/Markdown

