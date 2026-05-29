from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer


stopwords_es = set(["y","de","la","el","que","en","un","una","los","las","por","con"])
stopwords_fr = set(["et","le","la","les","de","des","un","une","en","que"])
stopwords_de = set(["und","der","die","das","ein","eine","in","zu","den"])

stopwords_all = {
    "en": ENGLISH_STOP_WORDS,
    "es": stopwords_es,
    "fr": stopwords_fr,
    "de": stopwords_de
}

def preprocess_bow(tokens, lang):
    """
    Elimina los stopwords en los idiomas europeos.
    """
    if lang in stopwords_all: # eliminar stopwords europeos
        tokens = [t for t in tokens if t not in stopwords_all[lang]]

    return tokens

def build_bow_matrix(df, text_column="tokens_bow", min_df=2, max_df=0.9):
    """
    Convierte tokens en matriz Bag-of-Words
    """

    # unir tokens en strings
    texts = df[text_column].apply(lambda x: ' '.join(x))

    # vectorizador
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)

    # matriz BoW
    X_bow = vectorizer.fit_transform(texts)

    return X_bow, vectorizer

def prepare_w2v_tokens(df):
    return df['text_clean'].copy()
