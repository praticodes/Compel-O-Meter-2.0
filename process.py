""" CSC111 Winter 2023 Course Project : Compel-O-Meter

Description
===========
This file contains the all the functions needed to process a written textual post into something
that we can create parse trees with and run sentiment analysis on :)

Copyright
==========
This file is Copyright (c) 2023 Akshaya Deepak Ramachandran, Kashish Mittal, Maryam Taj and Pratibha Thakur
"""

from __future__ import annotations
import re
from python_ta.contracts import check_contracts
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import spacy
import read_csv


def text_to_sentences(text: str) -> list[str]:
    """ Breaks a text up into a list of the sentences it's composed of.
    >>> text_to_sentences("We should not buy more? We have 13, 14, and 15 cars, trucks, and tractors, respectively.")
    ['We should not buy more', ' We have 13, 14, and 15 cars, trucks, and tractors, respectively']
    """
    separators = "[!?.]+"  # This regular expression matches one or more exclamation marks, question marks, or periods.

    sentences = re.split(separators, text)
    return [sentence for sentence in sentences if sentence != '']


def upper_to_lower(sentences: list[str]) -> list[str]:
    """Turns all upper case characters to lower case
    >>> upper_to_lower(['Hi','How are you'])
    ['hi', 'how are you']
    """
    return [str.lower(word) for word in sentences]


def lemmatize(word: str) -> str:
    """ Convert a word to its base or dictionary form, also known as a lemma. This makes it possible to compare the word
    with the words in the lexicon.
    >>> lemmatize('bats')
    'bat'
    >>> lemmatize('running')
    'run'
    >>> lemmatize('was')
    'be'
    """
    # download 'averaged_perceptron_tagger' if not already present in system.
    nltk.download('averaged_perceptron_tagger')
    # Initialize the WordNetLemmatizer function.
    lemmatizer = WordNetLemmatizer()

    # Find the parts of speech tag for the word.
    spacy.load('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

    doc = nlp(word)
    pos_tag = ''
    for token in doc:
        pos_tag = token.pos_

    # Since the lemmatizer takes in a word and the first letter of the part of speech (POS) tag as input, find the
    # first letter of the pos tag. Then, call the lemmatizer.
    if pos_tag.startswith('J'):
        pos_tag = wordnet.ADJ
        return lemmatizer.lemmatize(word, pos_tag)

    elif pos_tag.startswith('V') or pos_tag == 'AUX':
        pos_tag = wordnet.VERB
        return lemmatizer.lemmatize(word, pos_tag)

    elif pos_tag.startswith('N'):
        pos_tag = wordnet.NOUN
        return lemmatizer.lemmatize(word, pos_tag)

    elif pos_tag.startswith('R'):
        pos_tag = wordnet.ADV
        return lemmatizer.lemmatize(word, pos_tag)

    else:
        pos_tag = None
        return word


def is_intensifier(word: str) -> bool:
    """Check whether a word is an intensifier

    >>> is_intensifier('really')
    True
    >>> is_intensifier('very')
    True
    >>> is_intensifier('extremely')
    True
    >>> is_intensifier('quite')
    True
    >>> is_intensifier('nice')
    False
    """
    tokens = nltk.word_tokenize(word)
    tagged = nltk.pos_tag(tokens)
    intensifiers = [w for w, pos in tagged if pos == 'RB' and w in ['very', 'really', 'extremely', 'quite']]
    if not intensifiers:
        return False
    else:
        return True


def is_superlative(word: str) -> bool:
    """Check whether or not a word is a supperlative

    >>> is_superlative('best')
    True
    >>> is_superlative('happiest')
    True
    """

    nltk.download('averaged_perceptron_tagger')
    pos_tag = nltk.pos_tag([word])[0][1]
    exceptions = ['happiest', 'saddest']
    if ('JJ' in pos_tag and 'st' in word) or word in exceptions:
        return True
    else:
        return False


def is_numeral(word: str) -> bool:
    """ Return True if the given word is a numeral and False otherwise.

    >>> is_numeral('7')
    True
    >>> is_numeral('777')
    True
    >>> is_numeral('seventy one')
    True
    >>> is_numeral('hi')
    False
    """
    word_new = [string for string in word if string.isdigit()]
    if not word_new:
        singles = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        tens_first = ['ten', 'twent', 'thirt', 'forti', 'forty', 'fifti', 'fifty']
        powers = ['hundred', 'thousand', 'million', 'trillion', 'billion']
        teens_first = ['twel', 'thirt', 'fourt', 'fift']
        numbers = singles + tens_first + teens_first + powers
        word = word.lower()
        word = word.split(' ')
        for w in word:
            if any([num in w for num in numbers]):
                return True
        return False
    else:
        return True


def count_numerals(sentence: str) -> int:
    """Return the number of numerals in a sentence

    >>> count_numerals("I'm 19.")
    1
    >>> count_numerals("We have 13, 14, and 15 cars, trucks, and tractors, respectively.")
    3
    >>> count_numerals("79% of Vietnamese citizens and 21% of Indian citizens agreed with the bill.")
    2
    >>> count_numerals("'They tell you that you're lucky, but you're so confused', says Taylor Swift in a new song.")
    0
    """
    sentence = sentence.split(' ')
    return sum([is_numeral(num) for num in sentence])


def is_reasoning_text(text: str) -> bool:
    """Return True if the inputted text contains one or more words
    or phrases frequently used for reasoning purposes and False if not.

    >>> is_reasoning_text("I don't want to go because I'm scared")
    True
    >>> is_reasoning_text("Due to the failure of Congress, inflation is at 16%, marking an all-time high.")
    True
    >>> is_reasoning_text("Inflation is at 16%, marking an all-time high.")
    False
    """
    reasoning_words = read_csv.reasoning_words_list('data/reasoning_words.csv')
    text = text.lower()
    for word in reasoning_words:
        if word in text:
            return True
    return False


def count_logos_numerals(text: str) -> list[int]:
    """Return a list of counts of logos numerals in each sentece of a text.

    A logos numeral is one that is used for reasoning about something.

    IMPLEMENTATION NOTES:
    - Assume that a sentence's numerals are logos numerals if and only if the text it is a part of contains
    at least one word that is frequently used for reasoning.
    - use is_reasoning_text to complete this function

    >>> count_logos_numerals("Hi! We have 13, 14, and 15 cars, trucks, and tractors, respectively.")
    [0, 0]
    >>> count_logos_numerals("We should not buy more? We have 13, 14, and 15 cars, trucks, and tractors, respectively.")
    [0, 3]
    >>> count_logos_numerals("I went to the market today")
    [0]
    >>> count_logos_numerals("Covid cases are rising but the school doesn't care. This is unacceptable.")
    [0, 0]
    >>> count_logos_numerals("57 people died.")
    [0]
    >>> count_logos_numerals("57 people died because of you!")
    [1]
    """
    sentences = text_to_sentences(text)
    logos_num = []
    if is_reasoning_text(text):
        for sentence in sentences:
            num = count_numerals(sentence)
            logos_num.append(num)
    else:
        for _ in sentences:
            logos_num.append(0)
    return logos_num


def handle_multiline(text: str) -> str:
    """Return a new text where all lines and merged into one.
    """
    return text.replace('\n', ' ')


def process_text(text: str) -> list[str]:
    """Takes a given text and returns a list of independent clauses where all characters are in lower case
    >>> text = 'The castle crumbled overnight because I brought a knife to a gunfight. They took the crown but it is ok.'
    >>> process_text(text)
    ['the castle crumbled overnight because i brought a knife to a gunfight', ' they took the crown but it is ok']
    """
    text = handle_multiline(text)
    sentences = text_to_sentences(text)
    sentences = upper_to_lower(sentences)
    return sentences


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
