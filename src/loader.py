from pathlib import Path

def load_text(file_path):
    """
    Reads a text file and returns its contents as a string.
    """

    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        text = file.read()

    return text