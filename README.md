# Tumor Detection

### 📝 Project Description
This repository contains a deep learning pipeline designed to automate the detection of brain tumors from magnetic resonance imaging (MRI) scans. Utilizing a custom-built Convolutional Neural Network (CNN), the model processes raw axial brain images to extract high-level spatial features such as structural anomalies, tissue density variations, and irregular boundaries that indicate the presence of a malignancy.Manual analysis of MRI scans is time-consuming and highly dependent on radiologist availability. This project aims to provide a reliable, high accuracy computer vision tool to assist medical professionals by acting as a fast secondary screening mechanism, minimizing diagnostic latency and reducing human error in critical clinical environments.

---

### ✨ Key Features
* Automated Feature Extraction: Leverages multi-layered convolutional blocks to automatically capture complex spatial hierarchies and micro-patterns within brain tissue without relying on manual, hand-crafted feature engineering.
* End-to-End Image Pipeline: Includes a comprehensive preprocessing module that handles image resizing, grayscale normalization, contrast enhancement, and intensity scaling to ensure consistent model inputs across varying MRI scan qualities.
* Binary Classification Engine: Outputs precise probabilistic predictions regarding the presence or absence of a tumor, enabling definitive triage classification (Tumor Detected vs. Normal).
* Performance Metrics & Monitoring: Incorporates custom evaluation tracking for training/validation loss, classification accuracy, sensitivity (recall), and specificity to ensure the network minimizes dangerous false-negative predictions.

---

### 🪛 Tech Stack

**Core Language:** Python 3.12 (Can be used with newer versions as well)
\
\
**Libraries:**
* TensorFlow - for the NN
* FastAPI - for hosting the model
* uvicorn
* pydantic

---

### ⚙️ How to Run(Installation)

Once you have installed the repo, you can use the following commands,

* **Installing required libraries**
```bash
$ pip install - requirements.txt
```

* **Train the model and host it**

```bash 
$ python app.py retrain
```

you don't need to use the retrain flag after it has generated the "model.keras".
<br>
<br>
If you want to train the model with your own data then add the images to the "data/raw" folder. Make sure you have correctly seperated them into a "yes" and "no" folder.

```bash
$ python app.py retrain new
```
The "new" flag indicates that there is new training data available and the model needs to be trained on that instead of the the previously saved preprocessed data.

* **Changing the IP and port**

To change the ip and port open the app.py file and change the ip and port variable values.


```python
ip = "localhost"   #str
port = 8000        #int
```


**NOTE:** Python must be installed on your system.

---

### 🗺️ Project Structure

```
├── data/
│   ├── processed/
│   │   ├── x_test.npy
│   │   ├── x_train.npy
│   │   ├── y_test.npy
│   │   └── y_train.npy
│   └── raw/
│       ├── no/
│       │   └── download raw data and save here.txt
│       └── yes/
│           └── download raw data and save here.txt
├── model/
│   └── model will be saved here.txt
├── src/
│   ├── __init__.py
│   ├── data_preprocessor.py
│   ├── evaluation.py
│   └── train.py
├── templates/
│   └── index.html
├── app.py
├── Dockerfile.txt
├── mlflow.db
├── mlops.yaml
├── README.md
└── requirements.txt
```