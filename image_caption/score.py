from nltk.translate import bleu
import nltk
from nltk.translate.bleu_score import sentence_bleu


def score(hypothesis, reference):
    return sentence_bleu(reference, hypothesis)


if __name__ == '__main__':
    print score(["this", "is", "a", "cat"], [["this", "is", "a", "cat"]])

