import re

def clean_text(text):
    """
    Cleans raw document text.
    """

    text = text.replace("\t", " ")

    text = re.sub(r" +", " ", text)

    text = re.sub(r"\n+", "\n", text)

    return text.strip()