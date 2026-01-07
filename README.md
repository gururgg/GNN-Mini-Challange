# 🧠 GNN Mini Challenge: Node Classification on CiteSeer

Welcome to the **GNN Mini Challenge**!  
In this challenge, participants perform a **node classification task** on the well-known **CiteSeer citation network dataset** using **Graph Neural Networks (GNNs)** or any other graph-based approach.

---

## 📊 Dataset: CiteSeer

The **CiteSeer** dataset is a widely used benchmark in graph representation learning and node classification tasks.

It consists of:
- **3,327 nodes** (scientific publications)
- **4,732 edges** (citation links between publications)
- **3,703-dimensional binary node features**, representing word occurrences in document abstracts
- **6 node classes**, corresponding to different research topics

CiteSeer has been extensively used in **state-of-the-art GNN papers**, including GCN, GAT, GraphSAGE, and many others.

In the **original setup**, the dataset comes with predefined:
- training nodes
- validation nodes
- test nodes  

These splits are commonly used to benchmark node classification performance.

---

## 🎯 Challenge Setup

In this challenge, we define a **new and harder classification task**.

Instead of using the original dataset splits, we introduce:
- `train_mask_challenge`
- `val_mask_challenge`
- `test_mask_challenge`

These masks define **new train, validation, and test nodes**, making the task **more challenging than the original CiteSeer benchmark**.

### 🔒 Hidden Labels
- Node **features and graph structure are fully accessible**
- However, the **labels of test challenge nodes are hidden**
- Hidden labels are set to **`-1`**, meaning:
  > You do not have direct access to the true labels of challenge nodes

Participants must **infer labels purely from graph structure and node features**.

---

## 🧪 Objective

The **main goal** is to achieve the **highest classification accuracy on `test_mask_challenge`**.

In addition:
- You may use the **original CiteSeer masks** for analysis
- Comparing performance and differences between:
  - original val set
  - challenge test set  
  can provide useful insights and hints

🎯 **Bonus objective:**  
Achieve a **small performance gap** between the challenge task and the original task, indicating strong generalization.

---

## 📁 Submission Format

Participants must submit their predictions in a file located in **submissions/submission.csv**:

### File requirements:
- The file must contain **exactly one column**
- Column name must be **`preds`**
- Each row corresponds to **one node**
- Values must be **integer class labels** in `[0, 5]`

### 📌 Example

```csv
preds
2
4
1
0
3
```



## 🏆 Evaluation

- Submissions are evaluated automatically using **GitHub Actions**
- True labels are stored securely and are **never exposed**
- Only the **best challenge accuracy per participant** is kept
- Results are displayed on the **leaderboard.md**

---

## 🚀 Getting Started

1. Load the **CiteSeer graph** and node features by reading the **data/README.md**
2. Train your model using only the provided **train challenge masks**
3. Generate predictions for **all nodes**
4. Save your predictions in `submission.csv`
5. Submit your file via a **Pull Request** by addin **`submissions/submission.csv`** file

---

Good luck, and enjoy the challenge! 🧩  
