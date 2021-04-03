import pytest

from inverted_index import (
    InvertedIndex, build_inverted_index, load_documents, process_queries,
    DEFAULT_INVERTED_INDEX_STORE_PATH, callback_query
)

WIKIPEDIA_DATASET_FPATH = "wikipedia_sample.txt"
TINY_DATASET_FPATH = "tiny_wikipedia.txt"
QUERY_FILE_UTF8_FPATH = "queries.txt"
TINY_INVERTED_INDEX_STORE_PATH = "tiny_inverted.index"


def test_can_load_documents():
    documents = load_documents(TINY_DATASET_FPATH)
    etalon_documents = {
        "12": "another sentense four two one one three.\n",
        "25": "one two three four words.\n"
    }
    assert etalon_documents == documents, "load_documents incorrectly load dataset"


def test_query_inverted_index_with_query_file_utf_8():
    documents = load_documents(TINY_DATASET_FPATH)
    tiny_inverted_index = build_inverted_index(documents)
    tiny_inverted_index.dump_binary(TINY_INVERTED_INDEX_STORE_PATH)
    count = 1
    with open(QUERY_FILE_UTF8_FPATH) as q_file:
        for line in q_file:
            line = line.split()
            answer = tiny_inverted_index.query(line)
            if count == 1:
                etalon_answer = [12, 25]
            else:
                etalon_answer = [25]
            assert sorted(answer) == sorted(etalon_answer), (f"Expected answer is {etalon_answer},but you got {answer}")
            count += 1


def test_can_load_wikipedia_sample():
    documents = load_documents(WIKIPEDIA_DATASET_FPATH)
    assert len(documents) == 4100, "you incorrectly loaded Wikipedia sample"
