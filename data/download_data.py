import os
import requests

# -----------------------------
# Settings
# -----------------------------
def main():
    DATA_DIR = os.path.dirname(__file__)  # script folder = data/
    DATA_FILE = "citeseer_challenge_public.pt"

    # Zenodo direct download link
    ZENODO_URL = (
        "https://zenodo.org/record/18170986/files/citeseer_challenge_public.pt?download=1"
    )

    output_path = os.path.join(DATA_DIR, DATA_FILE)

    # -----------------------------
    # Create folder if it doesn't exist
    # -----------------------------
    os.makedirs(DATA_DIR, exist_ok=True)

    # -----------------------------
    # Download if not exists
    # -----------------------------
    if os.path.exists(output_path):
        print(f"Data already exists at {output_path}")
    else:
        print(f"Downloading {DATA_FILE} from Zenodo...")
        response = requests.get(ZENODO_URL, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download completed!")
        else:
            print(f"Failed to download, status code: {response.status_code}")

if __name__ == '__main__':
    main()
