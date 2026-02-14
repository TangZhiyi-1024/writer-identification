
# ✒️ Writer Identification with VLAD, GMP and Exemplar SVM

A complete image-retrieval pipeline for writer identification based on **local descriptors, residual aggregation and metric learning**.

This project investigates how different encoding and normalization strategies influence retrieval performance measured by **mean Average Precision (mAP)**.

---

---

## 🚀 Highlights

* Visual dictionary learning with MiniBatch k-means
* VLAD aggregation
* Power normalization (burstiness suppression)
* Generalized Max Pooling (GMP)
* Multi-VLAD + PCA whitening
* Query-adaptive embedding via Exemplar SVM
* Extensive empirical comparison

---

---

## 🧠 Method Overview

```
SIFT descriptors
      ↓
visual codebook (k-means)
      ↓
VLAD / GMP aggregation
      ↓
power normalization (optional)
      ↓
L2 normalization
      ↓
Exemplar SVM (optional)
      ↓
nearest neighbor search
      ↓
mAP
```

---

---

## ⚙️ Experimental Setup

| Item                    | Value                |
| ----------------------- | -------------------- |
| clusters (K)            | 100                  |
| descriptors for k-means | ~500k                |
| PCA dimension           | 1000                 |
| SVM                     | LinearSVC (balanced) |
| evaluation metric       | mAP                  |

---

---

## 📊 Results

### Baseline

| Method         | Top-1 | mAP   |
| -------------- | ----- | ----- |
| VLAD           | 0.809 | 0.615 |
| + Exemplar SVM | 0.875 | 0.731 |

---

### Power normalization

| Method           | Top-1 | mAP   |
| ---------------- | ----- | ----- |
| VLAD + powernorm | 0.823 | 0.631 |
| + Exemplar SVM   | 0.886 | 0.753 |

---

### Custom SIFT (angle=0) + Hellinger

| Method         | Top-1 | mAP   |
| -------------- | ----- | ----- |
| custom SIFT    | 0.823 | 0.631 |
| + Exemplar SVM | 0.888 | 0.753 |

---

### Generalized Max Pooling (GMP)

| Method         | Top-1 | mAP   |
| -------------- | ----- | ----- |
| GMP            | 0.842 | 0.657 |
| + Exemplar SVM | 0.902 | 0.780 |

---

### Multi-VLAD + PCA whitening

| Method         | Top-1 | mAP   |
| -------------- | ----- | ----- |
| multi-VLAD     | 0.888 | 0.729 |
| + Exemplar SVM | 0.891 | 0.742 |

---

---

## 📈 What matters most?

### 🔹 Exemplar SVM gives the strongest boost

Consistent improvement across almost all settings.
Acts as query-adaptive metric learning.

---

### 🔹 Better pooling > simple normalization

GMP and multi-VLAD provide larger gains than powernorm.

---

### 🔹 PCA reduces extra gain from E-SVM

After whitening, descriptors are already well conditioned.

---

## 🛠 Tech Stack

* Python
* NumPy
* OpenCV
* scikit-learn
* tqdm

---

---

## 📚 Concepts Behind the Work

* residual aggregation
* burstiness
* metric learning
* whitening
* high-dimensional similarity search

