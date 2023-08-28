""" CSC111 Winter 2023 Course Project : Compel-O-Meter

Description
===========
Generating the lexicon used for data analysis

Copyright
==========
This file is Copyright (c) 2023 Akshaya Deepak Ramachandran, Kashish Mittal, Maryam Taj and Pratibha Thakur
"""
import csv
from python_ta.contracts import check_contracts


def read_csv_positive_file(csv_file1: str) -> dict[str, int]:
    """Takes in a csv file of positive words and returns a dictionary where each word in the file is assigned to 1."""

    with open(csv_file1) as file:
        reader = csv.reader(file)

        i = 0
        while i < 35:
            next(reader)
            i += 1

        words = {}
        for row in reader:
            words[row[0]] = 1
    return words


def read_csv_negative_file(csv_file1: str) -> dict[str, int]:
    """Takes in a csv file of negative words and returns a dictionary where each word in the file is assigned to -1."""

    with open(csv_file1) as file:
        reader = csv.reader(file)

        i = 0
        while i < 35:
            next(reader)
            i += 1

        words = {}
        for row in reader:
            words[row[0]] = -1
    return words


def return_dictionary(csv_file1: str, csv_file2: str) -> dict[str, int]:
    """Takes in 2 csv files,one containing postive words and the other containing negative words. It returns a
    dictionary containing words from both the files with appropriate setiment scores assigned."""
    positive = read_csv_positive_file(csv_file1)
    negative = read_csv_negative_file(csv_file2)

    words = positive
    for word in negative:
        words[word] = -1

    return words


def reasoning_words_list(csv_file: str) -> list:
    """ Takes in a csv file of reasoning words and returns a list containing these words."""
    with open(csv_file) as file:
        reader = csv.reader(file)
        reasoning_words = []
        for row in reader:
            row = row[0].lower()
            reasoning_words.append(row)
    return reasoning_words


def ai_lexicon_words_dict(csv_file: str) -> dict[str, float]:
    """
    Takes in a csv file (ai_lexicon in our case) and returns a dictionary 
    with each word in the lexicon along with its sentiment score.
    """
    ai_words = {}
    with open(csv_file) as file:
        reader = csv.reader(file)
        for row in reader:
            ai_words[row[0]] = float(row[1]) / float(row[2])
    return ai_words


if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=True)
