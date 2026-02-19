"""Character-level tokenizer for microgpt: maps characters to integer token ids and back."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tokenizer:
    """Character-level tokenizer with a special BOS (Beginning of Sequence) token."""

    chars: tuple[str, ...]
    bos: int  # This won't have representation in the chars tuple, but we'll use it as an index in the encoding/decoding process.
    vocab_size: int
    char_to_id: dict[str, int]
    id_to_char: dict[int, str]

    @staticmethod
    def from_docs(docs: list[str]) -> "Tokenizer":
        chars = tuple(sorted(set("".join(docs))))
        bos = len(chars)
        vocab_size = len(chars) + 1
        char_to_id = {c: i for i, c in enumerate(chars)}
        id_to_char = {i: c for i, c in enumerate(chars)}
        return Tokenizer(chars=chars, bos=bos, vocab_size=vocab_size, char_to_id=char_to_id, id_to_char=id_to_char)

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token ids, wrapped with BOS on both sides."""
        return [self.bos] + [self.char_to_id[c] for c in text] + [self.bos]

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into a string, stripping BOS tokens."""
        return "".join(self.id_to_char[i] for i in ids if i != self.bos)


def main() -> None:
    from my_microgpt.dataset import load_docs

    docs = load_docs()
    tok = Tokenizer.from_docs(docs)
    print(f"vocab size: {tok.vocab_size}")
    print(f"chars: {''.join(tok.chars)}")
    print(f"BOS token id: {tok.bos}")
    example = docs[0]
    encoded = tok.encode(example)
    decoded = tok.decode(encoded)
    print(f"encode('{example}'): {encoded}")
    print(f"decode back: '{decoded}'")


if __name__ == "__main__":
    main()
