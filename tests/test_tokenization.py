from my_microgpt.tokenization import Tokenizer


def test_from_docs_vocab_size():
    tok = Tokenizer.from_docs(["abc", "bcd"])
    assert tok.chars == ("a", "b", "c", "d")
    assert tok.bos == 4
    assert tok.vocab_size == 5


def test_encode_wraps_with_bos():
    tok = Tokenizer.from_docs(["ab"])
    encoded = tok.encode("ab")
    assert encoded[0] == tok.bos == encoded[-1]
    assert len(encoded) == 4  # BOS + a + b + BOS


def test_decode_strips_bos():
    tok = Tokenizer.from_docs(["ab"])
    encoded = tok.encode("ab")
    assert tok.decode(encoded) == "ab"


def test_roundtrip():
    tok = Tokenizer.from_docs(["hello", "world"])
    for word in ["hello", "world", "hold", "well"]:
        assert tok.decode(tok.encode(word)) == word
