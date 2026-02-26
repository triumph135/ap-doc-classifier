Here’s a clean, GitHub-ready **README.md** based directly on your outline:

---

```markdown
# Document Type Classifier

A lightweight machine learning document classifier built with:

- Tesseract OCR
- PyMuPDF
- TF-IDF (scikit-learn)
- Logistic Regression
- Streamlit

The system classifies operational business documents into defined categories such as:

- Bill of Lading
- Bill of Lading Invoice
- Purchase Order Receipt
- Purchase Order Invoice

---

## Overview

This project uses OCR and text extraction to convert documents into text, vectorizes that text using TF-IDF, and applies a Logistic Regression model to predict document type.

It is designed to be:

- Simple
- Interpretable
- Fast to retrain
- Effective with small datasets

---

## Workflow

### 1. Collect Example Documents

Add labeled examples to:

```

data/labels.csv

```

Format:

```

path,label
data/docs/doc1.pdf,Bill of Lading
data/docs/doc2.pdf,Bill of Lading Invoice

````

**Notes:**

- `path` = file location
- `label` = one of the defined document classes
- Labels must match spelling exactly

Save documents into your training folder (e.g., `data/docs/`).

---

### 2. Verify Training Paths

Ensure file paths listed in `labels.csv` exist.

Update any paths if documents were moved.

---

### 3. Train the Model

Run:

```bash
python train_model.py
````

#### Training Steps

**a) Text Extraction**

* Native PDF text (if available)
* Otherwise → render PDF → Tesseract OCR

**b) Feature Engineering**

Text is vectorized using:

```
TfidfVectorizer(ngram_range=(1,2))
```

This captures:

* Single words (unigrams)
* Two-word phrases (bigrams)

Examples:

* "invoice"
* "amount due"
* "bill of"
* "of lading"

Bigrams improve classification accuracy by preserving context.

---

**c) Classification Model**

Uses:

```
LogisticRegression(class_weight="balanced")
```

* Multinomial logistic regression
* Learns weights for each document class
* Outputs class probabilities

Balanced class weighting prevents majority-class bias.

---

**d) Evaluation Metrics**

Initial metrics include:

* Accuracy
* Precision / Recall / F1
* Confusion Matrix

Example initial run:

* Dataset size: 50 documents
* Train/Test split: 75% / 25%
* Test set: 13 documents
* Accuracy: 1.00

**Important:**
Perfect scores on small datasets are common but not necessarily indicative of real-world performance.

---

### 4. Model Output

Training produces:

```
doc_type_model.joblib
```

This serialized model is used by:

* Prediction scripts
* Streamlit UI

---

## Evaluating on Unseen Data (Holdout Validation)

To measure true generalization:

### 1. Create Validation Set

Place new documents into:

```
data/holdout_eval/
```

---

### 2. Label Validation Documents

Create:

```
data/holdout_eval_labels.csv
```

Format:

```
path,label
data/holdout_eval/new_doc1.pdf,Bill of Lading
```

---

### 3. Run Evaluation

```bash
python evaluate_holdout.py
```

Outputs:

* Accuracy on unseen documents
* Misclassifications
* Confidence scores
* `holdout_results.csv`

---

## Running the Streamlit App

Launch locally:

```bash
streamlit run classifier_app.py
```

Features:

* Upload PDF or image
* OCR / text extraction
* Prediction + confidence
* Optional manual review threshold

---

## Key Design Choices

| Component              | Reason                               |
| ---------------------- | ------------------------------------ |
| TF-IDF                 | Works well with small datasets       |
| ngram_range=(1,2)      | Captures meaningful phrases          |
| Logistic Regression    | Fast, interpretable, strong baseline |
| Balanced class weights | Prevents bias                        |
| OCR fallback           | Handles scanned docs                 |

---

## Limitations

* OCR quality affects accuracy
* Requires labeled examples
* Streamlit Cloud does not include Tesseract by default
* Layout understanding is text-based (not vision-layout aware)

---

## Future Improvements

* Add more labeled documents
* Introduce PO classes
* Confidence-based routing
* Active learning loop
* Layout-aware models (LayoutLM / Donut)

---

## Requirements

Minimal deployment dependencies:

```
streamlit
pymupdf
pillow
pytesseract
scikit-learn
pandas
joblib
```

---

## License


Just tell me 👍
```
