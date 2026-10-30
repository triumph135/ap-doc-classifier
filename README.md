# AP Document Classifier

Classifies incoming accounts payable documents (scanned or digital PDFs and images) into one of four types, and reports the matching Cargas Energy document type. Low confidence documents are sent to manual review instead of being guessed.

| Model class | Cargas Energy document type |
| --- | --- |
| BOL Invoice (fuel invoice) | BOL Invoice |
| BOL Purchase (fuel purchase, Bill of Lading) | Material Receipt |
| PO Receipt | Material Receipt |
| PO Invoice | PO Invoice |

BOL Purchase and PO Receipt are both Material Receipts in Cargas Energy, but the model tells them apart because later steps in the workflow treat them differently.

## How it works

1. Text comes from the PDF itself when it has embedded text, otherwise from Tesseract OCR (first two pages).
2. A TF IDF vectorizer turns the text into features: word and word pair features plus character features, which tolerate OCR typos. Numbers are collapsed so document numbers and dates do not add noise.
3. A logistic regression model predicts the class and a confidence. Below the review threshold (default 0.90, set in `docclf/config.py`) the answer is "Manual Review".

The method is deliberately simple, fast to retrain, and easy to explain. Training and prediction share one extraction module (`docclf/extraction.py`), so a document is always read the same way.

## Setup (Windows)

1. Install Python 3.12 or newer, Git, and Tesseract (the UB Mannheim Windows build). Put Tesseract on your PATH, or set the `TESSERACT_CMD` environment variable to the full path of `tesseract.exe`.
2. Clone the repo somewhere outside OneDrive, then:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Streamlit Cloud, `packages.txt` installs Tesseract.

## Retraining (repeat whenever there are new labeled documents)

All labels live in one master file, `data/labels.csv`, with columns `path,label,split,group`. Only `path` and `label` are required. Documents themselves live under `data/docs/` and are never committed.

```powershell
python list_unlabeled.py        # lists documents not yet in labels.csv, writes data/unlabeled.csv
```

1. Fill in the `label` column of `data/unlabeled.csv` with `BOL Invoice`, `BOL Purchase`, `PO Receipt` or `PO Invoice`, then append those rows to `data/labels.csv`.
2. One time only, reserve a fixed holdout set that never trains the model: `python train_model.py --init-holdout 0.15`
3. Train: `python train_model.py`
4. Read the report in `models/versions/<timestamp>/report.txt`. `cv_predictions.csv` in the same folder lists mistakes first, which is the best place to find mislabeled or confusing documents.
5. If the run says PROMOTED, commit `models/latest.joblib` and `models/latest.json`. The app loads those two files.

What a run does:

* Trains only the classes that have enough examples (default 20). A class with none yet is reported as not trained, and the app shows which classes the loaded model knows. Add examples of a new class and the next run picks it up with no code changes.
* OCR results are cached by file fingerprint in `data/text_cache/`. The first run is slow because scanned pages are OCR'd. Later runs only process new files, and an interrupted run resumes where it stopped.
* Exact duplicate files count once. Identical files with conflicting labels are excluded and reported.
* Evaluates with 5 fold cross validation and the holdout set, and saves a dated model with a metadata file (classes, counts, library versions, scores).
* Promotes the new model only if its holdout score is not worse than the current model on the same holdout documents.

Optional columns in `labels.csv`: `split` (`holdout` or blank) and `group` (whatever should be held out together, for example a vendor or a customer). When `group` is filled in, documents from the same group are kept in the same cross validation fold, which gives a more honest score on sources the model has not seen. The training report then shows that score next to the usual random fold score, plus accuracy per group.

## Using the model

```powershell
streamlit run classifier_app.py                  # upload one document
python predict_folder.py "C:\path\to\folder"     # classify a folder, writes predictions.csv
python evaluate_holdout.py                       # score the promoted model on the holdout set
python explain_prediction.py "C:\path\to\file.pdf"   # why the model scored one document that way
```

The app has a "Why this result?" section that shows how familiar the document's wording is to the model and which words pushed the result toward each class. A low familiarity share usually means a vendor or format that was not in the training data.

## Things to know

* **Class coverage.** A model can only answer with classes it was trained on. Until a class has examples, documents of that type will be sent to a different class or to manual review.
* **Scanned versus digital documents.** If one class is mostly scans and another mostly digital PDFs, a model can learn the scan type instead of the content. The reports show accuracy by text source and by class and source. Watch for large gaps, and consider `python train_model.py --mode ocr` (OCR every page for consistency, slower).
* **One document type per file.** Customers are instructed to send a BOL purchase or a BOL invoice on its own. A file that combines both is classified from its first pages and is best left out of the labels file.
* **Handwriting.** OCR cannot read handwritten documents. They come back with too little text and go to manual review, and they are left out of training automatically.
* **Text only.** The model reads words, not page layout. If two types share nearly identical wording, layout features or a layout aware model would be the next step.
* **Library versions.** A saved model only loads reliably with the scikit-learn version that trained it. `requirements.txt` pins it, and the app warns on a mismatch.
* **What is committed.** Documents, extracted text and zip exports are git ignored. The labels file lists file names, and the model file contains words from the training documents, so review those before publishing.

## Layout

```
docclf/               shared code: config, extraction, labels, evaluation, classify
train_model.py        train, evaluate, version and promote a model
evaluate_holdout.py   score a model on the holdout set
predict_folder.py     classify a folder of documents
explain_prediction.py explain one prediction (scores, key words, familiarity)
list_unlabeled.py     find documents that still need labels
classifier_app.py     Streamlit app
models/latest.*       the promoted model (committed); models/versions/ is local history
data/labels.csv       master labels (committed); data/docs/ holds the documents (not committed)
archive_unused/       old experiments kept for history, not used
```

Settings such as class names, label aliases, the review threshold and the minimum examples per class are in `docclf/config.py`.
