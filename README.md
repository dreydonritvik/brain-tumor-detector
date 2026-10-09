# Brain Tumor Detector

A convolutional neural network that classifies brain MRI slices into four classes: **glioma**, **meningioma**, **pituitary** tumor, and **no tumor**. Training, evaluation, and inference all live in [`detector.ipynb`](detector.ipynb), and the trained model is included as `brain_tumor_expert_model.keras`.

> **Not a medical device.** This is a learning project. Do not use its predictions for diagnosis or any clinical decision.

## Repository contents

| Path | Description |
| --- | --- |
| `detector.ipynb` | Data loading, model definition, training, evaluation, and inference |
| `brain_tumor_expert_model.keras` | Trained model (~134 MB, stored with Git LFS) |
| `images/` | Sample images for running inference |
| `dataset/` | Training and test data. Not included in the repo; see [Dataset](#dataset) |

## Getting started

The model file is stored with [Git LFS](https://git-lfs.com/). Install Git LFS before cloning, or the `.keras` file will be a small pointer file instead of the model:

```bash
git lfs install
git clone https://gitlab.com/fireball3473/brain-tumor-detector.git
```

Install the dependencies:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn seaborn
```

### Classifying an image with the trained model

You don't need the dataset to use the trained model:

```python
import cv2
import numpy as np
import tensorflow as tf

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

model = tf.keras.models.load_model('brain_tumor_expert_model.keras')

img = cv2.imread('images/Hirnmetastase_MRT-T1_KM.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # the model was trained on RGB images
img = cv2.resize(img, (224, 224)) / 255.0

preds = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
print(CLASSES[np.argmax(preds)], f'{preds.max():.2%}')
```

### Retraining

Download the dataset, place it under `dataset/` as described below, and run `detector.ipynb` from top to bottom. On the original run, each of the 15 epochs took about 80 seconds.

## Dataset

<!-- TODO: add the dataset source link -->
Dataset source: _link to be added_

The notebook expects this layout, with one folder per class:

```
dataset/
├── Training/   # 8,582 images
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/    # 1,705 images
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## Model

The model is a small CNN trained from scratch with no pretrained weights:

```
Input 224×224×3
→ Conv2D(32, 3×3, ReLU)  → MaxPool(2×2)
→ Conv2D(64, 3×3, ReLU)  → MaxPool(2×2)
→ Conv2D(128, 3×3, ReLU) → MaxPool(2×2)
→ Flatten → Dense(128, ReLU) → Dropout(0.5)
→ Dense(4, softmax)
```

- **Optimizer:** Adam
- **Loss:** categorical cross-entropy
- **Training:** 15 epochs, batch size 32
- **Preprocessing:** pixel values scaled to [0, 1], no data augmentation

## Results

After 15 epochs, the training log reports:

| | Accuracy | Loss |
| --- | --- | --- |
| Training | 99.15% | 0.025 |
| Validation (`Testing/`) | **93.14%** | 1.305 |

Keep these limitations in mind when reading these numbers:

- **No separate test set.** The `Testing/` folder served as the validation set during training, so 93.14% is not a measurement on fully unseen data.
- **Overfitting.** Validation loss was lowest at epoch 3 (0.82) and kept rising afterward, while training loss went toward zero. Early stopping or data augmentation would probably help.

## Reference

This project draws on the following paper:

> Kör, H., & Mazman, R. (2025). Brain Tumor Detection and Classification with Deep Learning Based CNN Method. *Computational Systems and Artificial Intelligence*, 1(1), 15–19. https://doi.org/10.69882/adba.csai.2025073

The paper is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). It is cited here and not redistributed in this repository.
