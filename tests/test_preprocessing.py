from src.features.preprocessing import clean_base
from src.features.text_features import preprocess_bow

def test_clean_base_returns_list():

    result = clean_base(
        "THIS PRODUCT IS AMAZING",
        "en"
    )

    assert isinstance(result, list)

def test_clean_base_non_string():

    result = clean_base(None, "en")

    assert result == ""

def test_clean_base_url_replacement():

    result = clean_base(
        "Visit https://google.com",
        "en"
    )

    assert "url" in result

def test_preprocess_bow_removes_stopwords():

    tokens = ["this", "is", "a", "good", "product"]

    result = preprocess_bow(tokens, "en")

    assert "this" not in result

def test_preprocess_bow_unknown_language():

    tokens = ["hola", "que", "tal"]

    result = preprocess_bow(tokens, "xx")

    assert result == tokens