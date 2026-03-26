from pathlib import Path
import torch


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


def get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def main():
    text = Path("data.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    print("Dataset length:", len(data))
    print("Vocabulary size:", tokenizer.vocab_size)

    block_size = 8
    batch_size = 4
    xb, yb = get_batch(data, block_size, batch_size)

    print("\nInput batch tensor:")
    print(xb)

    print("\nTarget batch tensor:")
    print(yb)

    print("\nDecoded first example:")
    print("x:", tokenizer.decode(xb[0].tolist()))
    print("y:", tokenizer.decode(yb[0].tolist()))


if __name__ == "__main__":
    main()
