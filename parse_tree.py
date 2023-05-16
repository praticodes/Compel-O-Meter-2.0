""" CSC111 Winter 2023 Course Project : Compel-O-Meter

Description
===========
This file contains the ParseTree class, its methods, and all the functions needed to create parse trees
from a list of sentences :)

Copyright
==========
This file is Copyright (c) 2023 Akshaya Deepak Ramachandran, Kashish Mittal, Maryam Taj and Pratibha Thakur
"""

from __future__ import annotations
from typing import Any, Optional
import spacy

import analysis
import process


#######################################################################################
# The Parse Tree Class
#######################################################################################

class ParseTree:
    """A recursive tree data structure.

    Representation Invariants:
        - self._root is not None or self._subtrees == []
        - all(not subtree.is_empty() for subtree in self._subtrees)

    """
    # Private Instance Attributes:
    # - _root: The word and its corresponding descriptors stored at this tree's root, or None if the tree is empty.
    # - _subtrees: The list of subtrees of this tree, which is empty when self._root is None or has no subtrees.

    _root: tuple[str, str, str, str] | None
    _subtrees: list
    sentiment: int | float

    def __init__(self, root: Optional[Any], subtrees: list, sentiment: int = 0) -> None:  # list[Tree]
        """Initialize a new Tree with the given root tuple and subtree list.

        If root is None, the tree is empty.

        Preconditions:
            - root is not none or subtrees == []
        """
        self._root = root
        self._subtrees = subtrees
        self.sentiment = sentiment

    def is_empty(self) -> bool:
        """
        >>> tree = ParseTree(('is', 'ROOT', 'is', 'VERB'), [[('python', 'dep', 'is', 'NOUN'), []]])
        >>> not tree.is_empty()
        True
        >>> tree = ParseTree(None, [])
        >>> tree.is_empty()
        True
        """
        return self._root is None

    def contains(self, pos: str) -> bool:
        """For a given ParseTree, return whether a certain POS is present.

        Preconditions:
        - The parse tree tuples contain the POS of the word they represent in the 4th position, at index 3.

        >>> tree = trees_from_sentence('She drove the Greek piano')
        >>> tree[0].contains('NOUN')
        True
        >>> tree[0].contains('PRON')
        True
        >>> tree[0].contains('ADJ')
        True
        >>> not tree[0].contains('FAKE_POS')
        True
        """
        if self.is_empty():
            return False
        elif self._root[3] == pos:
            return True
        else:
            for subtree in self._subtrees:
                if subtree.contains(pos):
                    return True
            return False

    def pos_instances(self, pos: str) -> list:
        """ Returns the tuples of all nodes that contain the given POS.

        >>> tree = trees_from_sentence("Water isn't good for people who aren't human")
        >>> occurrences = tree[0].pos_instances('AUX')
        >>> occurrences
        [('is', 'ROOT', 'is', 'AUX'), ('are', 'relcl', 'people', 'AUX')]
        >>> occurrences = tree[0].pos_instances('NOUN')
        >>> occurrences
        [('Water', 'nsubj', 'is', 'NOUN'), ('people', 'pobj', 'for', 'NOUN')]
        """
        instances = []

        if self.is_empty():
            return instances
        elif self._root[3] == pos:
            instances.append(self._root)
        for subtree in self._subtrees:
            rec_value = subtree.pos_instances(pos)
            instances.extend(rec_value)

        return instances

    def dep_instances(self, tag: str) -> list[tuple]:
        """ Returns all the node tuples representing a word with a certain dependency

        >>> tree = trees_from_sentence("water isn't good for people who aren't good")
        >>> tree[0].dep_instances('neg')
        [("n't", 'neg', 'is', 'PART'), ("n't", 'neg', 'are', 'PART')]

        """
        dependencies = []
        if self._root[1] == tag:
            dependencies.append(self._root)
        for subtree in self._subtrees:
            rec = subtree.dep_instances(tag)
            dependencies.extend(rec)
        return dependencies

    def find_subtree_by_root(self, root: tuple) -> Any:
        """ Return the tree of a given node.

        Preconditions:
        - tree contains root
        """
        if self._root == root:
            return self
        else:
            for subtree in self._subtrees:
                value = subtree.find_subtree_by_root(root)
                if value is not None:
                    return value

    def upwards(self, tag: str, prev: Optional[Any] = None) -> Any:
        """ Takes a tag and returns the parent of the node that contains that tag.

        >>> tree = trees_from_sentence("water isn't good for people who aren't good")
        >>> tree[0].upwards('neg')
        [('is', 'ROOT', 'is', 'AUX'), ('are', 'relcl', 'people', 'AUX')]
        """
        # Base case:
        if self._root[1] == tag:
            return [prev]
        # Recursive case:
        else:
            parent_so_far = []
            prev = self._root
            for subtree in self._subtrees:
                curr = subtree
                parent_so_far.extend(curr.upwards(tag, prev))
            return [parent for parent in parent_so_far if parent is not None]

    def right(self, tag: str) -> Any:
        """ Takes a tag and returns the sibling to the right of the node that contains that tag

        >>> tree = trees_from_sentence("water isn't good for people who aren't good")
        >>> tree[0].right('neg')
        [('good', 'acomp', 'is', 'ADJ'), ('good', 'acomp', 'are', 'ADJ')]
        """
        parents = self.upwards(tag)
        if not parents:
            return None
        else:
            siblings_so_far = []
            for parent in parents:
                if self._root == parent:
                    sibling = [self._subtrees[i + 1]._root for i in range(0, len(self._subtrees) - 1)
                               if self._subtrees[i]._root[1] == 'neg']
                    siblings_so_far.extend(sibling)

                else:
                    for subtree in self._subtrees:
                        sibling = subtree.right(tag)
                        if sibling is not None:
                            siblings_so_far.extend(sibling)
            return siblings_so_far

    def initial_pathos_of_tree(self) -> None:
        """Assign sentiment (pathos) scores to each tree in the ParseTree

        IMPLEMENTATION NOTES:
        - This function should be implemented similarly to the function assigning GreedyGameTree's probabilities.
        - STEP 2: Combine with propagate_negations

        >>> tree2 = trees_from_sentence("the purse seam was shaped like a thread")
        >>> tree2[0].initial_pathos_of_tree()
        >>> tree2[0].sentiment
        0
        """
        if analysis.initial_pathos_to_tuple(self._root) != 0:
            self.sentiment = analysis.initial_pathos_to_tuple(self._root)
        for subtree in self._subtrees:
            subtree.initial_pathos_of_tree()

    def initial_pathos_of_tree_ai(self, text: str) -> None:
        """Assign sentiment (pathos) scores to each tree in the ParseTree

        Uses AI lexicon
        >>> tree = trees_from_sentence("I am happy")[0]
        >>> tree.initial_pathos_of_tree_ai("I am happy")
        """
        if self._root is not None:
            if analysis.initial_pathos_to_tuple_ai(self._root, text) != 0:
                self.sentiment = analysis.initial_pathos_to_tuple_ai(self._root, text)
            for subtree in self._subtrees:
                subtree.initial_pathos_of_tree_ai(text)

    def propagate_negations(self) -> None:
        """Propagate negations throughout the ParseTree, changing sentiment scores by 1 accordingly

        In some cases, a word may have a positive sentiment, but is associated with a negation. As a result,
        the overall sentiment would be negative. For example, in the sentence "It was not a good day." the node
        containing the word "good" would have a positive sentiment, 1, however because of the "not" in front of it,
        the sentiment should instead be -1. There is also a similar situation in which a negative word should have
        a positive sentiment due to a negation. Thus, this function updates the sentiments of certain nodes in the
        presence of a negation.

        Preconditions:
            - We assume that if the parent node is a sentiment bearing node, then the negation node is negating the
            parent node, otherwise it is negating the sibling nide that is immediately to the right of it.
        """
        if self.dep_instances('neg'):
            for parent in self.upwards('neg'):
                self.propogate_negation_helper(parent)

    def propogate_negation_helper(self, node: tuple) -> None:
        """Changes the sentiment of the appropriate sentiment bearing node for a negation"""
        if self.find_subtree_by_root(node).sentiment != 0:
            self.find_subtree_by_root(node).sentiment = - 1 * self.find_subtree_by_root(node).sentiment
        else:
            for sibling in self.right('neg'):
                if self.find_subtree_by_root(sibling).sentiment != 0:
                    self.find_subtree_by_root(sibling).sentiment = - 1 * self.find_subtree_by_root(
                        sibling).sentiment

    def has_intensifiers(self) -> bool:
        """Checks whether a sentence has an intensifier present"""
        if process.is_intensifier(self._root[0]):
            return True
        else:
            for subtree in self._subtrees:
                value = subtree.has_intensifiers()
                if value is True:
                    return value

            return False

    def handle_intensifiers(self) -> None:
        """ Improves the sentiment score of sentiment-bearing nodes that are affected by intensifiers

        In certain cases, there are intensifiers present which can change the sentiment of following words. If we
        compare the sentiments "That was a great show" and "That was a very great show", in the second sentence,
        the word "great" should have a greater sentiment compared to in the first sentence due to the presence of
        the word "very" before it. Thus, in the case of intensifiers, they increase the absolute value of an adjacent
        sentiment bearing node by one, which this function accounts for.

        Implementation Notes:
        - First implement the is_intensifier function in process.py
        """
        if self.has_intensifiers() is True and self.upwards('advmod') != []:
            for parent in self.upwards('advmod'):
                if self.find_subtree_by_root(parent).sentiment == 1:
                    self.find_subtree_by_root(parent).sentiment = 2
                elif self.find_subtree_by_root(parent).sentiment == -1:
                    self.find_subtree_by_root(parent).sentiment = -2

    def handle_superlatives(self) -> None:
        """ Improves the sentiment score of sentiment-bearing nodes that are superlatives

        Similar to intesifiers, superlatives can change the sentiment of words. When we look at the sentences
        "That was a weird thing I saw" and "That was the weirdest thing I've seen", the words "weird" and "weirdest",
        while orginating from the same root word, should have different sentiments, as "weirdest" conveys a greater
        sense of emotion. Thus, superlatives should increase the absolute value of nodes by one, which this function
        accounts for.

        Implementation Notes:
        - First implement the is_intensifier function in process.py
        """
        if process.is_superlative(self._root[0]):
            if self.sentiment == 1:
                self.sentiment += 1
            elif self.sentiment == -1:
                self.sentiment -= 1
        for subtree in self._subtrees:
            subtree.handle_superlatives()

    def final_pathos_of_tree(self) -> None:
        """Update all the sentiments of nodes based on negations, superlatives and intensifiers"""
        self.initial_pathos_of_tree()
        self.propagate_negations()
        self.handle_superlatives()
        self.handle_intensifiers()

    def final_pathos_of_tree_ai(self, text: str) -> None:
        """Update all the sentiments of nodes based on negations, superlatives and intensifiers using the ai version
        of the method initial_pathos_of_tree_ai
        """
        self.initial_pathos_of_tree_ai(text)
        self.propagate_negations()
        self.handle_superlatives()
        self.handle_intensifiers()

    def get_pathos_sum(self) -> tuple[int | float, bool]:
        """Returns the pathos (sentiment) score of the tree represented by the sentence as well as whether a negative
        sentiment bearing node is present.

        >>> tree = trees_from_sentence("I had a good dream")
        >>> tree[0].final_pathos_of_tree()
        >>> tree[0].get_pathos_sum()
        (1, False)
        >>> tree = ParseTree(('Hi', 'aaa', 'bbb', 'ccc'), [])
        >>> tree.final_pathos_of_tree()
        >>> tree.get_pathos_sum()
        (0, False)
        >>> tree = trees_from_sentence("I had a bad dream")
        >>> tree[0].final_pathos_of_tree()
        >>> tree[0].get_pathos_sum()
        (1, True)
        """
        negation_present = False
        sum_so_far = abs(self.sentiment)
        if self.sentiment < 0:
            negation_present = True
        for subtree in self._subtrees:
            sum_so_far += subtree.get_pathos_sum()[0]
            negation_present = negation_present or subtree.get_pathos_sum()[1]
        return (sum_so_far, negation_present)

    def count_sentiment_bearers(self) -> int:
        """ Count the number of nodes in this tree which has a sentiment that is not equal to 0.

        >>> tree = trees_from_sentence("she is happy")
        >>> tree[0].initial_pathos_of_tree()
        >>> tree[0].count_sentiment_bearers()
        1
        """
        count = 0
        if self.sentiment != 0:
            count += 1
        for subtree in self._subtrees:
            count += subtree.count_sentiment_bearers()
        return count

    def get_pathos(self) -> tuple[float, bool]:
        """ Get the overall pathos of the tree as well as whether a negative sentiment bearing node is present

        >>> tree = trees_from_sentence("Are you sad too")
        >>> tree[0].final_pathos_of_tree()
        >>> tree[0].get_pathos()
        (1.0, True)
        """
        return self.get_pathos_sum()[0] / max(self.count_sentiment_bearers(), 1), self.get_pathos_sum()[1]


#######################################################################################
# All the functions needed to create parse trees from lists of sentences!
#######################################################################################

def tree_list_from_sentence(sentence: str) -> list[list[tuple[str, Any, Any, Any] | list[str]]]:
    """Create a tree list for a given sentence.

    >>> my_tree_list = tree_list_from_sentence("She drove the Greek piano")
    >>> my_tree_list[0]
    [('She', 'nsubj', 'drove', 'PRON'), []]
    >>> my_tree_list[1]
    [('drove', 'ROOT', 'drove', 'VERB'), ['She', 'piano']]
    >>> my_tree_list[2]
    [('the', 'det', 'piano', 'DET'), []]
    >>> my_tree_list[3]
    [('Greek', 'amod', 'piano', 'ADJ'), []]
    >>> my_tree_list[4]
    [('piano', 'dobj', 'drove', 'NOUN'), ['the', 'Greek']]
    """
    nlp = spacy.load('en_core_web_sm')

    doc = nlp(sentence)

    tree_list = []

    for token in doc:
        tree_list.append([(token.text,
                           token.dep_,
                           token.head.text,
                           token.pos_),
                          [str(child) for child in token.children]])

    return tree_list


def find_root(tree_list: list[list[tuple[str, str, str, str] | list[str]]]) -> Optional[list[str]]:
    """Find the root value of the given tree list.

    Implementation notes:
    - if 'ROOT' not in [tree[0][1] for tree in tree_list], return None.

    >>> sentence = "And the tennis court was covered up with some tent-like things and you asked me to dance"
    >>> my_tree_list = tree_list_from_sentence(sentence)
    >>> my_tree = find_root(my_tree_list)
    >>> my_tree
    ['covered']
    """
    roots = [tree[0][0] for tree in tree_list if tree[0][1] == 'ROOT']
    return roots


def check_for_words(subtrees: list[Any]) -> bool:
    """ Return true if the subtrees list contains words. Else, return False.
    """
    return any([isinstance(entry, str) for entry in subtrees])


def tree_struct_from_word(word: str, tree_list: list, parent: str) -> list:
    """
    Takes a word and returns a nested list tree structure where the inputted word is the root node

    Implementation Notes:
    - Return an empty list if the inputted word is not in the tree list.

    """
    for tree in tree_list:
        subtrees = tree[1]
        if tree[0][0] == word and tree[0][2] == parent and not check_for_words(subtrees):
            return tree
        elif tree[0][0] == word and tree[0][2] == parent and check_for_words(subtrees):
            sub = []
            for subtree_word in subtrees:
                subtree = tree_struct_from_word(subtree_word, tree_list, tree[0][0])
                sub.append(subtree)
            return [tree[0], sub]

    return []


def impose_tree_struct_on_list(tree_list: list) -> list:
    """ Imposes the tree structure on a given nested list.

    Implementation notes:
        - This method is *not* mutating. It simply returns a new, structured tree list.

    """
    root_words = find_root(tree_list)
    return [tree_struct_from_word(root_word, tree_list, root_word) for root_word in root_words]


def leaves_to_subtrees(subtrees: list) -> list:
    """ Turns a subtrees list (which was a part of a nested list tree structure) into a list of parse trees.
    """
    new_subtrees = []
    for subtree in subtrees:
        # BASE CASE
        if not subtree[1]:
            # then it is a leaf
            subtree_root = subtree[0]
            subtree_subtrees = subtree[1]
            subtree = ParseTree(subtree_root, subtree_subtrees)
            new_subtrees.append(subtree)
        # RECURSIVE CASE
        else:
            parse_tree = ParseTree(subtree[0], leaves_to_subtrees(subtree[1]))
            new_subtrees.append(parse_tree)

    return new_subtrees


def tree_struct_to_tree(tree_struct: list) -> ParseTree:
    """ Convert the given tree structure to an actual tree.
    """
    # Find the root and subtrees
    root = tree_struct[0]
    subtrees_raw = tree_struct[1]
    # Convert all subtrees to trees
    subtrees_processed = leaves_to_subtrees(subtrees_raw)
    # Create the parse tree from the root and processed subtrees
    parse_tree = ParseTree(root, subtrees_processed)
    return parse_tree


def trees_from_sentence(sentence: str) -> Optional[list[ParseTree]]:
    """Return a list of parse tree for the given sentence.

    It returns a list because a sentence can contain multiple roots and in this case multiple trees would for rather
    than just one.
    """
    # Create a tree list from the sentence
    tree_list = tree_list_from_sentence(sentence)
    # Impose the tree structure onto the created tree list
    tree_structs = impose_tree_struct_on_list(tree_list)
    # Convert the tree list into a Parse Tree and return
    trees = [tree_struct_to_tree(tree_struct) for tree_struct in tree_structs]
    return trees


def trees_from_sentences(sentences: list[str]) -> list[ParseTree]:
    """Returns a list of parse trees from a list of sentences
    """
    trees = []
    for sentence in sentences:
        trees.extend(trees_from_sentence(sentence))
    return trees


if __name__ == '__main__':
    import doctest
    doctest.testmod()
