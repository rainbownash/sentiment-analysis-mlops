import re
import string


def clean_base(text, lang):
    if not isinstance(text, str): # control de calidad, evita romper el pipeline si encuentra una entrada que no sea string
        return ""

    # espacios, saltos de línea y tabs
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\n\t\r]+', ' ', text)

    # URL e emails
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '<EMAIL>', text)

    # lowercase para idiomas europeos (incluso el alemán con ß, python 3 y unicode moderno lo procesan bien)
    if lang in ["en","es","fr", "de"]:
        text = text.lower()

    # reemplazar emojis por token <EMOJI>
    emoji_pattern = r'[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U0001F1E0-\U0001F1FF]' # se definen rangos unicode que cubren la mayoría de emojis
    text = re.sub(emoji_pattern, '<EMOJI>', text)

    # reemplazar emoticonos ASCII por <EMOJI>
    ascii_emoji_pattern = r'[:;=8][\-~]?[)\(D]'
    text = re.sub(ascii_emoji_pattern, '<EMOJI>', text)

    # eliminar puntuación
    punctuation = string.punctuation.replace("'", "").replace("-", "")
    text = text.translate(str.maketrans('', '', punctuation))

    # tokenización
    if lang in ["en","es","fr","de"]: # separar por palabra
        tokens = text.split()
    else:  # chino, japonés: separar por carácter
        tokens = list(text)

    return tokens