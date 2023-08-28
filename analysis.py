""" CSC111 Winter 2023 Course Project : Compel-O-Meter

Description
===========
This file contains the all the functions needed to process a written textual post into something
that we can create parse trees with and run sentiment analysis on :)

Copyright
==========
This file is Copyright (c) 2023 Akshaya Deepak Ramachandran, Kashish Mittal, Maryam Taj and Pratibha Thakur
"""
import csv
import os
import nltk
import parse_tree
import process
import read_csv
from typing import Union


def create_lexicon() -> dict:
    """Create a sentiment analysis dictionary. """
    lexicon = {}
    nltk.download('opinion_lexicon')
    positive_words = set(nltk.corpus.opinion_lexicon.positive())
    negative_words = set(nltk.corpus.opinion_lexicon.negative())
    for word in positive_words:
        lexicon[word] = 1
    for word in negative_words:
        lexicon[word] = -1
    dict_from_csv = read_csv.return_dictionary('data/positive_words.csv', 'data/negative_words.csv')
    lexicon.update(dict_from_csv)
    return lexicon


def relevant(tag: str) -> bool:
    """Returns whether the pos tag is relevant.

    A relevant tag for sentiment analysis is a verb, noun, or adjective as these
    parts of speech often carry sentiment.
    """
    return tag.startswith('JJ') or tag.startswith('NN') or (tag.startswith('VB') and tag != 'VBP')


def present_in_file(word: str, csv_file: str) -> bool:
    """Return whether a word is present in the lexicon or not

    NOTE: It is not possible to write doctests for this function because the lexicon keeps changing
    """
    with open(csv_file) as file:
        reader = csv.reader(file)
        for row in reader:
            if word in row:
                return True
    return False


def find_absents(text: str, old_lexicon: dict) -> list[str]:
    """Returns a set of all the words in a text that are not already there in the lexicon.
    """
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    absent = []

    for word_tag in tagged:
        if relevant(word_tag[1]) and word_tag[0] not in old_lexicon:
            absent.append(word_tag[0])

    return absent


def create_lexicon_ai(text: str, old_lexicon: dict) -> dict:
    """Return a sentiment analysis dictionary.

    If a word in the text is a noun or adjective or adverb and is not in the dictionary, add it
    to the ai_lexicon.csv file.
    """
    absent = find_absents(text, old_lexicon)

    if not absent:
        return old_lexicon
    else:
        lexicon = {}
        with open('data/ai_lexicon.csv') as file:
            reader = csv.reader(file)
            for row in reader:
                for word in absent:
                    if row[0] == word:
                        lexicon[row[0]] = float(row[1]) / float(row[2])

    lexicon.update(create_lexicon())
    return lexicon


def update_lexicon_data_ai(text: str, pathos: float, negative_sentiment: bool, old_lexicon: dict) -> None:
    """ This function will update the lexicon based on the missing words, and it's pathos score"""
    absent = find_absents(text, old_lexicon)

    for word in absent:
        if not present_in_file(word, 'data/ai_lexicon.csv'):
            with open('data/ai_lexicon.csv', 'a', newline='') as file:
                if negative_sentiment:
                    pathos = 0 - pathos
                writer = csv.writer(file)
                writer.writerow([word, pathos, 1])  # word, sentiment_count, word_count
        else:
            with open('data/ai_lexicon.csv', 'r', newline='') as file:
                reader = csv.reader(file)
                rows = []
                for row in reader:
                    if word in row:
                        if negative_sentiment:
                            pathos = 0 - pathos
                        rows.append([word, str((float(row[1]) + pathos)),
                                     str(float(row[2]) + 1)])
                    else:
                        rows.append(row)

            with open('data/ai_lexicon1.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows)

            os.remove('data/ai_lexicon.csv')
            os.rename('data/ai_lexicon1.csv', 'data/ai_lexicon.csv')


def initial_pathos_to_tuple(node: tuple) -> int:
    """Return the sentiment (pathos) scores of the given node

    >>> initial_pathos_to_tuple(('happy', 'aaa', 'bbb', 'ccc'))
    1
    """
    lexicon = create_lexicon()
    if node[0] in lexicon:
        return lexicon[node[0]]
    else:
        return 0


def initial_pathos_to_tuple_ai(node: tuple, text: str) -> Union[int, float]:
    """ Return the sentiment (pathos) scores of the given node

    Uses an AI lexicon.
    """
    lexicon = create_lexicon_ai(text, old_lexicon=create_lexicon())
    if node[0] in lexicon:
        return lexicon[node[0]]
    else:
        return 0


def count_logos(text: str) -> int:
    """Return the number of times a given text uses logos


    IMPLEMENTATION NOTES:
    - use count_logos_numerals function from process.py

    >>> count_logos("Hi! We have 13, 14, and 15 cars, trucks, and tractors, respectively.")
    0
    >>> count_logos("We should not buy more. We have 13, 14, and 15 cars, trucks, and tractors, respectively.")
    3
    >>> count_logos("I went to the market today")
    0
    >>> count_logos("Covid cases are rising but the school doesn't care. This is unacceptable.")
    0
    >>> count_logos("57 people died.")
    0
    >>> count_logos("57 people died because of you!")
    1
    """
    return sum(process.count_logos_numerals(text))


def get_logos(text: str) -> Union[float, int]:
    """Return a logos score for a given text.

    Logos scores are in the range 0 to 1. A score of 0 indicates the absence of logos. A score of 1 indicates a
    maximal use of logos.

    A logos score of 0 is achieved when the text contains no reasoning, as per process.is_reasoning().

    If a text contains reasoning, logos_score = 0.5 + average_count,
    where average count is the average number of counts per sentence in the text, scaled to a float between 0 and 0.5
    """
    if process.is_reasoning_text(text):
        count = count_logos(text) / len(process.text_to_sentences(text))
        return min(0.5 + count, 1.0)
    else:
        return 0.0


def get_pathos(text: str) -> tuple[Union[float, int], bool]:
    """Return the pathos score for the given text alongside its direction.

    The pathos score for a given text is the average of the pathos scores of all the roots of its
    constituent sentences.

    The pathos score should be rounded to the nearest 100th.
    """
    sentences = process.process_text(text)
    trees = parse_tree.trees_from_sentences(sentences)
    for tree in trees:
        tree.final_pathos_of_tree()
    pathos_score = sum([(parsetree.get_pathos()[0]) for parsetree in trees]) / max(len(trees), 1)
    negative_sentiment_present = any(parsetree.get_pathos()[1] for parsetree in trees)
    return pathos_score, negative_sentiment_present


def get_pathos_ai(text: str) -> (float, bool):
    """Returns the pathos score for the given text alongside its direction (a '+' or '-' or 'undetermined').

    The pathos score for a given text is the average of the pathos scores of all the roots of its
    constituent sentences.

    Uses AI.
    """
    sentences = process.process_text(text)
    trees = parse_tree.trees_from_sentences(sentences)
    for tree in trees:
        tree.final_pathos_of_tree_ai(text)
    pathos = [parsetree.get_pathos() for parsetree in trees]
    pathos_score = sum([result[0] for result in pathos]) / max(len(trees), 1)
    negative_sentiment_present = any(result[1] for result in pathos)
    return pathos_score, negative_sentiment_present


def find_problematic_buzzwords() -> list:
    """Returns a list of problematic buzzwords.

    Please note this function contains some disturbing language.
    """
    incel_buzzwords = ["chad", "normie", "femoid", "Stacy", "roastie", "blackpill", "beta", "cuck", "hypergamy",
                       "oneitis", "looksmaxing"]
    white_supremacist_buzzwords = ["race realism", "white genocide", "cultural Marxism", "Jewish Question",
                                   "replacement theory", "alt-right", "blood and soil", "14 words", "identitarianism",
                                   "racial purity", "ethnostate", "racial realism", "white power",
                                   "western civilization", "cuckservative", "triggered", "snowflake",
                                   "virtue signaling", "political correctness", "cultural appropriation",
                                   "anti-white", "anti-racist is a code word for anti-white"]
    neo_nazi_buzzwords = ["white power", "Sieg Heil", "14/88", "blood and soil", "Final Solution", "Zyklon B",
                          "Aryan race", "skinhead", "KKK", "racial purity", "white nationalism", "hate crime",
                          "racial supremacy", "Holocaust denial"]
    transphobic_buzzwords = ["biological sex", "transgender ideology", "gender dysphoria", "trans bathroom",
                             "sex change operation", "transgenderism", "mental illness", "trans agenda", "transphobia",
                             "gender ideology", "gender identity disorder", "trans regret"]
    misogynistic_buzzwords = ["men's rights", "alpha male", "friendzone", "female privilege", "hypergamy", "red pill",
                              "toxic femininity", "feminazi", "male oppression", "misandry", "incel", "femoid",
                              "pickup artist", "rape"]
    anti_semitic_buzzwords = ["globalist", "new world order", "Zionist Occupied Government (ZOG)", "Holohoax",
                              "Jewish conspiracy", "blood libel", "cultural Marxism", "white genocide",
                              "Judeo-Bolshevism", "Protocols of the Elders of Zion", "Israel firsters"]
    anti_muslim_buzzwords = ["radical Islamic terrorism", "jihadist", "sharia law", "Islamic invasion", "Islamization",
                             "Muslim ban", "clash of civilizations", "terrorist sympathizers", "Islamophobia",
                             "creeping Sharia", "jihadist cells", "Islamic extremism"]
    anti_hindu_buzzwords = ["Cow worshipper", "Kaffir", "Infidel", "Heathen", "Pagans", "Idol worshippers", "Bhakt",
                            "Saffron terror", "Sanghi", "Hindu Taliban", "Ghar Wapsi", "Love Jihad", "Anti-national",
                            "Fascist",
                            "Hindutva", "Intolerant", "Hindu Rashtra"]
    anti_asian_buzzwords = ["Kung Flu", "China virus", "Yellow Peril", "Chink", "Gook", "Jap", "Oriental",
                            "Asian Invasion",
                            "Model Minority Myth", "Foreigner", "Exotic", "Tiger Mom", "Crazy Rich Asians",
                            "Stereotype",
                            "Traditional", "Ninja", "Dragon Lady"]
    anti_latina_buzzwords = ["Spicy", "Exotic", "Mamacita", "Hot tamale", "Dirty", "Illegal", "Anchor baby",
                             "Welfare queen",
                             "Ghetto", "Gang member", "Drug dealer", "Frijolera", "Border bunny", "Maid", "Sex worker",
                             "Latina heat", "Sexy señorita"]

    anti_lgbtq_buzzwords = ["conversion therapy", "ex-gay", "traditional values", "family values", "homosexual agenda",
                            "gay lifestyle", "religious freedom", "God's plan", "unnatural", "abomination",
                            "sexual deviance", "queer agenda", "Adam and Eve, not Adam and Steve"]

    buzzwords = incel_buzzwords + white_supremacist_buzzwords + neo_nazi_buzzwords + transphobic_buzzwords
    buzzwords += misogynistic_buzzwords + anti_semitic_buzzwords + anti_hindu_buzzwords + anti_muslim_buzzwords
    buzzwords += anti_asian_buzzwords + anti_latina_buzzwords + anti_lgbtq_buzzwords

    return buzzwords


def count_problematic_buzzwords(text: str) -> int:
    """Returns a count of the number of problematic buzzwords in the given text
    """
    buzzwords = find_problematic_buzzwords()
    buzzword_count = {word: text.count(word) for word in buzzwords}
    return sum(buzzword_count[buzzword] for buzzword in buzzword_count)


def ethics_warning(text: str) -> str:
    """Return an ethics warning if and only if the text likely expresses views harmful to marginalized
    groups
    """
    sentences_count = len(process.text_to_sentences(text))
    if count_problematic_buzzwords(text) / sentences_count > 0.1:
        return "WARNING: This post may express dangerous sentiments towards marginalized groups. Think critically " \
               "about this post and remember to show respect to other people, regardless of your differences."
    else:
        return "No dangerous sentiments towards marginalized groups was detected in this text. Note that it is still " \
               "important to think critically about the sentiments expressed."


def get_logos_description(scores: tuple[Union[float, int], Union[float, int], Union[float, int], bool]) -> str:
    """Return a description of what the logos score means"""
    logos_score = scores[2]
    if logos_score <= 0.25:
        return "This text does not use logos as a significant tool for persuasion."
    elif logos_score <= 0.75:
        return "This text may use logos to convince the reader of its argument. Always cross-check" \
               "facts and figures found online with reputed and unbiased sources of information."
    elif logos_score <= 1.25:
        return "This text is rich in its use of logos. Always cross-check facts and figures online with reputed" \
               "and unbiased sources of information."
    else:
        return "This text exemplifies the use of logos. Always cross-check facts and figures online with reputed" \
               "and unbiased sources of information."


def get_pathos_description(scores: tuple[Union[float, int], Union[float, int], Union[float, int], bool]) -> str:
    """Return a description of what the pathos score means"""
    pathos_score = scores[1]
    if pathos_score <= 0.25:
        return "This text does not use pathos as a significant tool for persuasion."
    elif pathos_score <= 0.75:
        return "This text may use pathos to convince the reader of its argument."
    elif pathos_score <= 1.25:
        return "This text is rich in its use of pathos."
    else:
        return "This text exemplifies the use of pathos."


def get_negative_sentiment(scores: tuple[Union[float, int], Union[float, int], Union[float, int], bool]) -> str:
    """Return a description of what the pathos score means"""
    negative_sentiment_present = scores[3]
    if negative_sentiment_present:
        text1 = "This text has some negative sentiment present."
        text2 = "This indicates that the text may be attempting to convince you against something"
        return text1 + " " + text2
    else:
        text1 = "This text does not have negative sentiment present."
        text2 = "This indicates that the text may be attempting to convince you for something"
        return text1 + " " + text2


def get_compellingness(text: str) -> tuple[Union[float, int], Union[float, int], Union[float, int], bool]:
    """Return the compellingess score of the given text and its direction (a '+' or '-' or 'undetermined').

    This function uses the following piecewise formula:

        if initial_compellingness > 2.0:
            compellingness = 2.0
        else:
            compellingess = initial_compellingness

        where initial_compellingness = max(logos_score, pathos score) + 0.5 * min(logos_score, pathos score)

    >>> get_compellingness("I ate pizza")
    (0.0, 0.0, 0.0, False)
    >>> get_compellingness("I am happy")
    (1.0, 1.0, 0.0, False)
    >>> get_compellingness("I am happy but he is sad. Are you sad too? I went to the mall because I was sad.")
    (1.25, 1.0, 0.5, True)
    >>> get_compellingness("Even in my worst lies, you saw the truth in me.")
    (1.5, 1.5, 0.0, True)
    >>> string1 = "and the tennis court was covered up with some tent-like things and you asked me to dance"
    >>> string2 = "and i said dancing is a dangerous game"
    >>> get_compellingness(string1 + string2)
    (1.0, 1.0, 0.0, True)
    >>> get_compellingness("I did it because I had to.")
    (0.5, 0.0, 0.5, False)
    >>> get_compellingness("Because of the failure of Congress, 76 people lost their lives.")
    (2.0, 1.0, 1.5, True)
    """
    pathos = get_pathos(text)
    pathos_score = pathos[1]
    logos_score = get_logos(text)
    initial_compellingness = max(logos_score, pathos_score) + 0.5 * min(logos_score, pathos_score)
    if initial_compellingness > 2.0:
        compellingness = 2.0
    else:
        compellingness = initial_compellingness
    return compellingness, pathos_score, logos_score, pathos[1]


def get_compellingness_ai(text: str) -> tuple[Union[float, int], Union[float, int], Union[float, int], bool]:
    """Return the compellingess score of the given text and its direction (a '+' or '-' or 'undetermined').

    This function uses the following piecewise formula:

        if initial_compellingness > 2.0:
            compellingness = 2.0
        else:
            compellingess = initial_compellingness

        where initial_compellingness = max(logos_score, pathos score) + 0.5 * min(logos_score, pathos score)

    Uses AI
    """
    pathos = get_pathos_ai(text)
    pathos_score = pathos[0]
    negative_sentiment = pathos[1]
    logos_score = get_logos(text)
    initial_compellingness = max(logos_score, pathos_score) + 0.5 * min(logos_score, pathos_score)
    if initial_compellingness > 2.0:
        compellingness = 2.0
    else:
        compellingness = initial_compellingness

    update_lexicon_data_ai(text, pathos_score, negative_sentiment, create_lexicon())
    return compellingness, pathos_score, logos_score, negative_sentiment


def get_compellingness_description(scores: tuple[Union[float, int], Union[float, int], Union[float, int], bool]) -> str:
    """Return a description of what the pathos score means"""
    compellingess_score = scores[0]
    if compellingess_score <= 0.25:
        return "This text is not significantly compelling."
    elif compellingess_score <= 0.75:
        return "This text is somewhat compelling."
    elif compellingess_score <= 1.25:
        return "This text is very compelling."
    else:
        return "This text achieved the highest compellingness score category."


def compellingness_with_description(text: str) -> tuple[str, str, str, str, str, str]:
    """Returns descriptions of the scores given in get_compellingness as well as any ethics warnings.

    Preconditions:
    - scores[0] is the compellingness score
    - scores[1] is the pathos score
    - scores[2] is the logos score
    - [score <= 2 for score in scores]
    """
    scores = get_compellingness(text)
    compellingness_summary = "The compellingness score for the text was: " + str(round(scores[0], 2)) + '\n'
    pathos_summary = "The pathos score for the text was: " + str(round(scores[1], 2)) + '\n'
    logos_summary = "The logos score for the text was: " + str(round(scores[2], 2)) + '\n'
    results_summary = compellingness_summary + pathos_summary + logos_summary

    return results_summary, get_compellingness_description(scores), get_pathos_description(scores), \
        get_logos_description(scores), get_negative_sentiment(scores), ethics_warning(text)


def compellingness_description_ai(text: str) -> tuple[str, str, str, str, str, str]:
    """Returns descriptions of the scores given in get_compellingness as well as any ethics warnings,
    except if it encounters

    Preconditions:
    - scores[0] is the compellingness score
    - scores[1] is the pathos score
    - scores[2] is the logos score
    - [score <= 2 for score in scores]

    """
    scores = get_compellingness_ai(text)
    compellingness_summary = "The compellingness score for the text was: " + str(round(scores[0], 2)) + '\n'
    pathos_summary = "The pathos score for the text was: " + str(round(scores[1], 2)) + '\n'
    logos_summary = "The logos score for the text was: " + str(round(scores[2], 2)) + '\n'
    results_summary = compellingness_summary + pathos_summary + logos_summary

    return results_summary, get_compellingness_description(scores), get_pathos_description(scores), \
        get_logos_description(scores), get_negative_sentiment(scores), ethics_warning(text)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
