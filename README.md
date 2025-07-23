# Handwritten-Text-Recognition-OCR-using-CNN-BiLSTM-CTC
This project implements an end-to-end Optical Character Recognition (OCR) system for handwritten English words using the IAM dataset. It includes preprocessing, character encoding, model training using deep learning, and evaluation.
# Features
- Preprocessing of IAM handwritten word images
- Label encoding using a custom alphabet
- CNN + BiLSTM + CTC Loss architecture
- Parallelized image loading
- Character-level and word-level accuracy evaluation
- Model saving to `.h5` format
# Dataset
This project uses the [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database). You must download it manually and organize it as follows:
- `IAM_DIR` in the code refers to the root of the image directory.
- `WORDS_FILE` refers to the transcript metadata.
# Model Architecture
- Input: Grayscale word image (64×256)
- CNN Layers: Feature extraction (3 blocks)
- BiLSTM Layers: Sequential modeling (2 layers, bidirectional)
- Dense + Softmax: Character probability distribution
- CTC Loss: Used for unsegmented sequence training
# Requirements
Install the following packages:
```bash
pip install tensorflow opencv-python-headless numpy pandas scikit-learn tqdm

How to Run:
Ensure the dataset is properly structured and the paths are correct in the script.

Run the Python script:
```bash
python handwriting_ocr.py

The script will:
- Load and preprocess images and labels
- Train the model
- Evaluate on validation and test sets
- Save the model to handwriting_recognition_model.h5

Training and Evaluation:
- Uses the Adam optimizer (learning rate = 0.0001)
- Trains for 30 epochs
- Computes both character-level and word-level accuracy

Output:
- Trained model saved as: handwriting_recognition_model.h5
- Console prints training logs, model summary, validation, and test accuracy

Notes:
- Characters outside the defined alphabet are ignored
- Uses multiprocessing to speed up image preprocessing
- CTC decoding is used for final predictions

License
This project is for educational and research purposes. IAM dataset requires a license for use.
