"""Contains the code that 'trains' the ai_lexicon by running the algorithm several times
on open source files, thus adding new words and updating each word's sentiment score.
"""
from analysis import get_compellingness_ai
from process import text_to_sentences


def run_on_sentences(text: str) -> None:
    """ Runs get_compellingess_ai on each sentence in a text file. """
    sentences = text_to_sentences(text)
    for sentence in sentences:
        get_compellingness_ai(sentence)


# Read the text file
file_path = "texts.txt"
with open(file_path, "r", encoding="utf-8") as file:
    file_content = file.read().lower()

# Process the text from the file
run_on_sentences(file_content)
