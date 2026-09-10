from collections import Counter

corpus = """
low low low low low
lowest lowest
newer newer newer newer newer newer
wider wider wider
new new
"""

words = corpus.split()


def get_vocab(words):
    vocab = Counter()

    for word in words:
        vocab[tuple(list(word) + ["_"])] += 1

    return vocab


def get_pairs(vocab):
    pairs = Counter()

    for tokens, frequency in vocab.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += frequency

    return pairs


def merge_pair(vocab, pair):
    new_vocab = Counter()
    merged_token = pair[0] + pair[1]

    for tokens, frequency in vocab.items():
        new_tokens = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        new_vocab[tuple(new_tokens)] += frequency

    return new_vocab, merged_token


def train_bpe(words, num_merges=20):
    vocab = get_vocab(words)

    # Initial vocabulary
    vocabulary = set()

    for tokens in vocab:
        vocabulary.update(tokens)

    print("Initial vocabulary size:", len(vocabulary))

    merges = []

    for step in range(num_merges):
        pairs = get_pairs(vocab)

        if not pairs:
            break

        best_pair, count = pairs.most_common(1)[0]

        vocab, merged_token = merge_pair(vocab, best_pair)

        vocabulary.add(merged_token)
        merges.append(best_pair)

        print(
            f"Step {step + 1}: "
            f"{best_pair} -> {merged_token}, "
            f"count = {count}, "
            f"vocabulary size = {len(vocabulary)}"
        )

    return merges


def segment_word(word, merges):
    tokens = list(word) + ["_"]

    for pair in merges:
        new_tokens = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                    continue

            new_tokens.append(tokens[i])
            i += 1

        tokens = new_tokens

    return tokens


# Train BPE
merges = train_bpe(words, num_merges=20)

# Segment required words
test_words = [
    "new",
    "newer",
    "lowest",
    "widest",
    "newestest"
]

print("\nSegmentation:")

for word in test_words:
    print(word, "->", segment_word(word, merges))