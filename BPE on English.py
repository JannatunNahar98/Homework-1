import re
from collections import Counter


paragraph = """
Natural language processing helps computers understand human language.
Subword tokenization breaks rare words into smaller reusable pieces.
Machine learning models can then represent unseen words efficiently.
Careful tokenization improves the quality of many NLP applications.
"""

words = re.findall(r"[A-Za-z]+", paragraph.lower())


def make_vocab(words):
    vocab = Counter()

    for word in words:
        vocab[tuple(list(word) + ["_"])] += 1

    return vocab


def get_pairs(vocab):
    pairs = Counter()

    for tokens, freq in vocab.items():
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += freq

    return pairs


def merge(vocab, pair):
    new_vocab = Counter()
    new_token = pair[0] + pair[1]

    for tokens, freq in vocab.items():
        result = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and (
                tokens[i], tokens[i + 1]
            ) == pair:
                result.append(new_token)
                i += 2
            else:
                result.append(tokens[i])
                i += 1

        new_vocab[tuple(result)] += freq

    return new_vocab, new_token


def train_bpe(words, number_of_merges=30):
    vocab = make_vocab(words)

    learned_tokens = set()

    for step in range(number_of_merges):
        pairs = get_pairs(vocab)

        if not pairs:
            break

        pair, count = pairs.most_common(1)[0]

        vocab, new_token = merge(vocab, pair)

        learned_tokens.add(new_token)

        print(
            f"Merge {step + 1}: "
            f"{pair} -> {new_token}, "
            f"frequency = {count}"
        )

    return learned_tokens


learned_tokens = train_bpe(words, 30)

print("\nFive longest final subword tokens:")

for token in sorted(
    learned_tokens,
    key=lambda x: (-len(x), x)
)[:5]:
    print(token)