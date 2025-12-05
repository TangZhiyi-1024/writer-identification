# Writer Identification & Retrieval using VLAD, GMP, and Exemplar-SVM  

*A Python implementation for historical writer identification (ICDAR17 dataset)*

## ⭐ Overview

This project implements a full writer identification and retrieval pipeline using:

- **SIFT local descriptors**

- **VLAD encoding** (Vectors of Locally Aggregated Descriptors)

- **Generalized Max Pooling (GMP)**

- **Exemplar SVM feature transformation (E-SVM)**

  

#### The goal is:

Given an image of a handwritten manuscript, find the **person who wrote this image** (Top-1 accuracy) or retrieve the **most similar document of the same author** (mAP).

Based on experimental results, this project achieved the following on the ICDAR17 test set:

| Method        | Top-1 Accuracy | mAP       |
| ------------- | -------------- | --------- |
| VLAD baseline | **0.81**       | **0.615** |
| VLAD + E-SVM  | **0.875**      | **0.731** |

------

## 📂 Dataset

This project uses the **ICDAR17 Historical Writer Identification** dataset:

- Train：394 writers × 3 images = **1182 images**
- Test：720 writers × 5 images = **3600 images**

We have pre-extracted **SIFT patch descriptors** and stored them in:

```
icdar17_local_features/
    train/
    test/
icdar17_labels_train.txt
icdar17_labels_test.txt
```

Each `.pkl.gz` file contains local features of a graph (such as an Nx128 SIFT).

------

## 🔧 Pipeline Summary

### 1️⃣ Sampling + Local Descriptors

- Using dense sampling SIFT
- Each image yields a large number of 128-D descriptors

### 2️⃣ Dictionary Learning

Construct a codebook using MiniBatch KMeans (clusters = K).

Corresponding VLAD embedding formula:
$$
ϕ_k(x) = (x - μ_k)
$$

### 3️⃣ VLAD Encoding

Aggregate local features for each image → a global 6400-D descriptor (K=50, D=128).



### 4️⃣ Power Normalization

Solving "visual burstiness"

### 5️⃣ (Optional) Generalized Max Pooling (GMP)

GMP is a ridge regression solution:
 
$$
\xi_{gmp} = \arg\min \Vert Φ^T ξ - 1\Vert^2 + λ \Vert ξ\Vert^2
 
$$
 Where λ is the `--gamma` parameter in the code.

### 6️⃣ Writer Retrieval via Cosine Distance

![image-20251205164352509](C:\Users\LENOVO\AppData\Roaming\Typora\typora-user-images\image-20251205164352509.png)

### 7️⃣ Exemplar SVM (E-SVM) Transformation

A personalized classifier is trained for each test image:


$$
\min_{w,b} \frac12 |w|^2

- c_p \max(0, 1 - w^T x_p - b)^2
- \sum_{x_n \in N} c_n \max(0, 1 + w^T x_n + b)^2
$$
 

Ultimately, **w / ||w||** was used as the new encoding.

------

## ▶️ Running the Code

### Train dictionary + VLAD + Evaluation

```bash
python skeleton.py \
  --in_train icdar17_local_features/train \
  --in_test icdar17_local_features/test \
  --labels_train icdar17_labels_train.txt \
  --labels_test icdar17_labels_test.txt
```

if use GMP：

```bash
python skeleton.py ... --gmp --gamma 1.0
```

------

## 📈 Experimental Results

### Baseline VLAD

```
Top-1 accuracy: 0.8089
mAP: 0.6152
```

### VLAD + Exemplar SVM

(It has replaced parmap in the Windows environment and is implemented using a single-threaded, safe method.)

```
Top-1 accuracy: 0.8753
mAP: 0.7312
```

------

## 🏗️ Code Structure

```
skeleton.py               # main pipeline
parmap.py                 # parallel map helper (unused on Windows)
icdar17_local_features/   # precomputed SIFT descriptors
    train/
    test/
icdar17_labels_train.txt
icdar17_labels_test.txt
README.md                 # this document
```

Core function:

| Function                | Purpose                                 |
| ----------------------- | --------------------------------------- |
| `loadRandomDescriptors` | Extracting features for KMeans training |
| `dictionary`            | Constructing cluster centers for VLAD   |
| `assignments`           | Nearest neighbor assignment αk(x)       |
| `vlad()`                | VLAD + normalization                    |
| `gmp_pooling()`         | Generalized Max Pooling                 |
| `distances()`           | cosine distance                         |
| `evaluate()`            | Top-1 & mAP                             |
| `esvm()`                | Exemplar-SVM Feature transformation     |

------

