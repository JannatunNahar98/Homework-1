import nltk
from nltk.tokenize import wordpunct_tokenize

paragraph = """
Students are learning NLP through practical exercises.
They don't always notice punctuation, contractions, and word forms.
A well-designed tokenizer separates symbols carefully while preserving useful expressions.
Natural-language tools can make this process easier.
"""

tokens = wordpunct_tokenize(paragraph)

print(tokens)