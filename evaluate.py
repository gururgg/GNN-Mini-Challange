import os
import base64
import numpy as np
import torch
import pandas as pd
import json
from sklearn.metrics import accuracy_score
import argparse


# -----------------------------
# Helper: decode tensor from base64 secret
# --
def decode_tensor(secret_name, dtype):
    b64 = os.environ[secret_name]
    bytes_data = base64.b64decode(b64)
    arr = np.frombuffer(bytes_data, dtype=dtype).copy()
    return torch.from_numpy(arr)


def evaluate(submission_file):
    # -----------------------------
    # Load secrets
    # -----------------------------
    y = decode_tensor("PRIVATE_Y", np.int64)
    test_mask_challenge = decode_tensor("PRIVATE_TEST_MASK_CHALLENGE", np.bool_)
    test_mask = decode_tensor("PRIVATE_TEST_MASK", np.bool_)

    # -----------------------------
    # Load participant submission
    # -----------------------------
    submission = pd.read_csv(submission_file)

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
    
    scores = {
        'challange_accuracy': challenge_acc,
        'original_accuracy': original_acc,
        'accuracy_gap': gap
    }

    return scores


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description='Score a submission file')
    parser.add_argument('submission_file', help='Path to submission CSV file')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    args = parser.parse_args()
    
    scores = evaluate(args.submission_file)
    
    if args.json and scores:
        print("\n" + "="*60)
        print("JSON OUTPUT:")
        print("="*60)
        print(json.dumps(scores, indent=2))

