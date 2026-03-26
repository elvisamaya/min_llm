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


class DataModule:
    def __init__(self, text: str):
        self.tokenizer = CharTokenizer(text)
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        split = int(0.9 * len(data))
        self.train_data = data[:split]
        self.val_data = data[split:]

    def get_batch(self, split: str, block_size: int, batch_size: int):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x, y


def main():
    text = Path("data.txt").read_text(encoding="utf-8")
    data_module = DataModule(text)

    print("Train set length:", len(data_module.train_data))
    print("Validation set length:", len(data_module.val_data))

    xb, yb = data_module.get_batch("train", block_size=8, batch_size=4)

    print("\nTrain batch example:")
    print(xb)

    print("\nDecoded example:")
    print(data_module.tokenizer.decode(xb[0].tolist()))
    print(data_module.tokenizer.decode(yb[0].tolist()))


if __name__ == "__main__":
    main()
