import os
import base64
import numpy as np
import torch
import pandas as pd
import json
from datetime import datetime
from sklearn.metrics import accuracy_score

# -----------------------------
# Helper: decode tensor from base64 secret
# -----------------------------
def decode_tensor(secret_name, dtype):
    b64 = os.environ[secret_name]
    bytes_data = base64.b64decode(b64)
    arr = np.frombuffer(bytes_data, dtype=dtype).copy()
    return torch.from_numpy(arr)

# -----------------------------
# Load secrets
# -----------------------------
y = decode_tensor("PRIVATE_Y", np.int64)
test_mask_challenge = decode_tensor("PRIVATE_TEST_MASK_CHALLENGE", np.bool_)
test_mask = decode_tensor("PRIVATE_TEST_MASK", np.bool_)

# -----------------------------
# Load participant submission
# -----------------------------
submission_files = [f for f in os.listdir("submissions") if f.endswith(".csv")]
assert len(submission_files) == 1, "Exactly one submission CSV required"

submission_path = os.path.join("submissions", submission_files[0])
user = os.path.splitext(submission_files[0])[0]

submission = pd.read_csv(submission_path)

if len(submission) != len(test_mask):
    raise ValueError(
        f"CSV length {len(submission)} does not match number of test nodes {len(test_mask)}"
    )

preds = torch.tensor(submission.values, dtype=torch.int64)

# -----------------------------
# Apply masks
# -----------------------------
y_challenge = y[test_mask_challenge]
y_original  = y[test_mask]

pred_challenge = preds[test_mask_challenge]
pred_original  = preds[test_mask]

# -----------------------------
# Metrics
# -----------------------------
challenge_acc = accuracy_score(y_challenge, pred_challenge.numpy())
original_acc  = accuracy_score(y_original, pred_original.numpy())
gap = challenge_acc - original_acc

print(f"Challenge Accuracy: {challenge_acc:.4f}")
print(f"Original Accuracy : {original_acc:.4f}")
print(f"Gap               : {gap:.4f}")

# -----------------------------
# Leaderboard update
# -----------------------------
entry = {
    "user": user,
    "challenge_acc": float(challenge_acc),
    "original_acc": float(original_acc),
    "gap": float(gap),
    "timestamp": datetime.utcnow().isoformat()
}

leaderboard_path = "leaderboard.json"


if os.path.exists(leaderboard_path):    
    with open(leaderboard_path) as f:
        board = json.load(f)

    if isinstance(board, str):
        board = json.loads(board)

    if not isinstance(board, list):
        board = []

# Keep best challenge score per user
board = [b for b in board if b["user"] != user]
board.append(entry)

board = sorted(board, key=lambda x: x["challenge_acc"], reverse=True)

with open(leaderboard_path, "w") as f:
    json.dump(board, f, indent=2)
    
    
md_path = "leaderboard.md"

lines = [
    "# 🏆 Leaderboard\n",
    "\n",
    "This leaderboard is automatically updated via GitHub Actions.\n",
    "Only the **best challenge accuracy per user** is kept.\n",
    "\n",
    "| Rank | User | Challenge Accuracy | Original Accuracy | Gap | Timestamp |\n",
    "|------|------|-------------------|-------------------|-----|-----------|\n",
]

for i, b in enumerate(board, start=1):
    lines.append(
        f"| {i} | {b['user']} | "
        f"{b['challenge_acc']:.4f} | "
        f"{b['original_acc']:.4f} | "
        f"{b['gap']:.4f} | "
        f"{b['timestamp']} |\n"
    )

with open(md_path, "w") as f:
    f.writelines(lines)

