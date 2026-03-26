from pathlib import Path


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def main():
    text = Path("data.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)

    print("Loaded dataset successfully.\n")
    print("First 300 characters:\n")
    print(text[:300])

    print("\nVocabulary size:", tokenizer.vocab_size)
    print("Vocabulary characters:")
    print(sorted(tokenizer.stoi.keys()))

    encoded = tokenizer.encode(text[:80])
    decoded = tokenizer.decode(encoded)

    print("\nEncoded sample:")
    print(encoded)

    print("\nDecoded back:")
    print(decoded)


if __name__ == "__main__":
    main()
