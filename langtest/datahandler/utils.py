from datetime import datetime


def get_results(tokens, labels, text):
    current_entity = None
    current_span = []
    results = []
    char_pos = 0  # Tracks the character position in the text

    for i, (token, label) in enumerate(zip(tokens, labels)):
        token_start = char_pos
        token_end = token_start + len(token)
        if label.startswith("B-"):
            if current_entity:
                results.append(
                    {
                        "value": {
                            "start": current_span[0],
                            "end": current_span[-1],
                            "text": text[current_span[0] : current_span[-1]],
                            "labels": [current_entity],
                            "confidence": 1,
                        },
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                    }
                )
            current_entity = label[2:]
            current_span = [token_start, token_end]
        elif label.startswith("I-") and current_entity:
            current_span[-1] = token_end
        elif label == "O" and current_entity:
            results.append(
                {
                    "value": {
                        "start": current_span[0],
                        "end": current_span[-1],
                        "text": text[current_span[0] : current_span[-1]],
                        "labels": [current_entity],
                        "confidence": 1,
                    },
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                }
            )
            current_entity = None
            current_span = []

        # Move to the next character position (account for the space between tokens)
        char_pos = (
            token_end + 1
            if i + 1 < len(tokens) and tokens[i + 1] not in [".", ",", "!", "?"]
            else token_end
        )

    if current_entity:
        results.append(
            {
                "value": {
                    "start": current_span[0],
                    "end": current_span[-1],
                    "text": text[current_span[0] : current_span[-1]],
                    "labels": [current_entity],
                    "confidence": 1,
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
            }
        )
    return results


def process_document(doc):
    tokens = []
    labels = []

    # replace the -DOCSTART- tag with a newline
    doc = doc.replace("-DOCSTART-", "")

    for line in doc.strip().split("\n"):
        if line.strip():
            parts = line.strip().split()
            if len(parts) == 4:
                token, _, _, label = parts
                tokens.append(token)
                labels.append(label)

    text = ""
    for _, token in enumerate(tokens):
        if token in {".", ",", "!", "?"}:
            text = text.rstrip() + token + " "
        else:
            text += token + " "

    text = text.rstrip()

    results = get_results(tokens, labels, text)
    now = datetime.utcnow()
    current_date = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    json_output = {
        "created_ago": current_date,
        "result": results,
        "honeypot": True,
        "lead_time": 10,
        "confidence_range": [0, 1],
        "submitted_at": current_date,
        "updated_at": current_date,
        "predictions": [],
        "created_at": current_date,
        "data": {"text": text},
    }

    return json_output


def ensure_download_and_unzip(url: str, extract_to: str):
    """
    Ensures that a file is downloaded from the given URL
    and unzipped to the specified directory.

    Args:
        url (str): The URL of the file to download.
        extract_to (str): The directory where the file should be extracted.

    This function checks if the specified directory exists. If it does not exist,
    it creates the directory, downloads the file from the given URL, and extracts its contents into the directory.


    """
    import requests
    import zipfile
    import io
    import os

    try:
        # 1. Critical Check: Exit early if the path already exists
        if os.path.exists(extract_to):
            print(f"Skipping download. Path '{extract_to}' already exists.")

        else:
            # 2. Download the file (Removed stream=True since response.content reads all at once)
            response = requests.get(url)
            response.raise_for_status()

            # 3. Create the folder structure
            os.makedirs(extract_to, exist_ok=True)

            # 4. Unzip directly from memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(extract_to)

            print(f"Successfully downloaded and extracted to {extract_to}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid ZIP file.")
