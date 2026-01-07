# 📁 Data

The public dataset for the **GNN Mini Challenge** can be downloaded using the provided Python script.

## Download

Run the following script:

```bash
python data/download_data.py
```

This will download the dataset into the data/ folder with the filename **citeseer_challenge_public.pt**.

## 📦 Dataset Format

The downloaded file is a **list containing a PyTorch Geometric `Data` object**.

Each `Data` object includes the following fields:

- **`data.x`**  
  Node feature matrix

- **`data.edge_index`**  
  Graph connectivity in COO format (represents the adjacency matrix)

- **`data.y`**  
  Node labels  
  - A label value of `-1` indicates that the label is **hidden from participants**

---

## 🧪 Masks

The dataset provides two different task setups:
the **original CiteSeer task** and the **challenge task**.

### Original CiteSeer Masks
- **`data.train_mask`** – training mask for the original task  
- **`data.val_mask`** – validation mask for the original task  
- **`data.test_mask`** – test mask for the original task  

### Challenge Masks
- **`data.train_mask_challenge`** – training mask for the challenge task  
- **`data.val_mask_challenge`** – validation mask for the challenge task  
- **`data.test_mask_challenge`** – test mask for the challenge task  

---

## 🎯 Training & Evaluation Guidelines

- During training, **only use `data.train_mask_challenge`**
- Validation should be performed using **`data.val_mask_challenge`**
- The **main objective** is to maximize accuracy on **data.test_mask_challenge**

### Secondary Objective
Minimize the performance gap between:
- **`data.test_mask`** (original task)
- **`data.test_mask_challenge`** (challenge task)

This encourages models that:
- generalize well on the original problem
- adapt effectively to the more challenging evaluation setup

---

## ⚠️ Important Notes

- Hidden labels (`-1`) must **not** be used during training
- True labels for challenge nodes are stored securely and are **never exposed**
- Any attempt to bypass the intended setup may result in **disqualification**
