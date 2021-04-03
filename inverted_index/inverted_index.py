"""
inverted index library
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType, ArgumentTypeError
from collections import Counter
import sys
import struct
from io import TextIOWrapper
import re

DEFAULT_DATASET_PATH = "wikipedia_sample.txt"
DEFAULT_INVERTED_INDEX_STORE_PATH = "inverted.index"


class EncodedFileType(FileType):
    """Class for custom file encoding"""

    def __call__(self, string):
        if string == "-":
            if 'r' in self._mode:
                stdin = TextIOWrapper(sys.stdin.buffer, encoding=self._encoding)
                return stdin
            if 'w' in self._mode:
                stdout = TextIOWrapper(sys.stdout.buffer, encoding=self._encoding)
                return stdout
            msg = 'argument "-" with mode %r' % self._mode
            raise ValueError(msg)
        try:
            return open(string, self._mode, self._bufsize, self._encoding, self._errors)
        except OSError as err:
            message = "can't open '%s':%s"
            raise ArgumentTypeError(message % (string, err)) from err


class InvertedIndex:
    """Class for inverted index"""

    def __init__(self, data):
        self.data = data

    def query(self, words: list) -> list:
        """Return the list of relevant documents for the given query"""
        assert isinstance(words, list), (
            "query should be provided with a list of words,but user provided: "
            f"{repr(words)}"
        )
        print(f"query inverted index with request {repr(words)}", file=sys.stderr)
        if len(words) == 0:
            print("error,incorrect request,empty list of words", file=sys.stderr)
            return []
        if words[0] in self.data.keys():
            list_1 = self.data[words[0]]
        else:
            return []
        for i in range(1, len(words)):
            if words[i] in self.data.keys():
                list_2 = self.data[words[i]]
                list_1 = set(list_1) & set(list_2)
            else:
                return []
        return list(list_1)

    @classmethod
    def load_binary(cls, filepath: str):
        """Loads inverted index from file and decodes it from binary format"""
        read_file = open(filepath, 'rb')
        digit = read_file.read(4)
        count = struct.unpack('>i', digit)[0]
        data = {}
        ind = 0
        while ind < count:
            dlength = read_file.read(4)
            length = struct.unpack('>i', dlength)[0]
            str_format = '>' + str(length) + 's'
            d_str = read_file.read(length)
            str_b = struct.unpack(str_format, d_str)[0]
            key = str_b.decode()
            data[key] = []
            digit = read_file.read(4)
            num_documents_for_key = struct.unpack('>i', digit)[0]
            for _ in range(num_documents_for_key):
                d_num = read_file.read(4)
                num = struct.unpack('>i', d_num)[0]
                data[key].append(num)
            ind = ind + 1
        read_file.close()
        return cls(data)

    def dump_binary(self, filepath: str):
        """Encodes inverted index in binary format and saves to file"""
        write_file = open(filepath, 'wb')
        count = len(self.data)
        digit = struct.pack('>i', count)
        write_file.write(digit)
        for key in self.data:
            bt_key = key.encode()
            length = len(bt_key)
            digit = struct.pack('>i', length)
            write_file.write(digit)
            bin_str = struct.pack('>' + str(length) + 's', bt_key)
            write_file.write(bin_str)
            len_data_key = len(self.data[key])
            digit = struct.pack('>i', len_data_key)
            write_file.write(digit)
            for number in self.data[key]:
                digit = struct.pack('>i', number)
                write_file.write(digit)
        write_file.close()


def load_documents(filepath: str):
    """Loads document from file."""
    print(f"loading documents from path {filepath} to build inverted index..", file=sys.stderr)
    file = open(filepath, 'r', encoding="utf-8")
    documents = {}
    for line in file:
        number, document = line.split(maxsplit=1, sep='\t')
        if number.isnumeric():
            documents[number] = document
    return documents


def build_inverted_index(documents):
    """Builds inverted index from documents"""
    inv_index = InvertedIndex({})
    for key in documents:
        line = re.split(r'\s+', documents[key])
        # line = documents[key].split()
        count = Counter(line)
        for word in set(count.elements()):
            if word not in inv_index.data.keys():
                inv_index.data[word] = [int(key)]
            elif id not in inv_index.data[word]:
                inv_index.data[word].append(int(key))
    # print(inv_index.data)
    return inv_index


def callback_build(arguments):
    """Loads documents and builds inverted index from them."""
    print(f"call build subcommand with arguments: {arguments}", file=sys.stderr)
    documents = load_documents(arguments.dataset_path)
    inverted_index = build_inverted_index(documents)
    # inverted_index.dump(arguments.output)
    inverted_index.dump_binary(arguments.output)


def callback_query(arguments):
    """Processes query request"""
    print(f"call query subcommand with arguments: {arguments}", file=sys.stderr)
    file_flag = True
    if arguments.query:
        file_flag = False
        return process_queries(arguments.inverted_index_filepath, arguments.query, file_flag)
    return process_queries(arguments.inverted_index_filepath, arguments.query_file, file_flag)


def process_queries(inverted_index_filepath, query_file, file_flag):
    """Loads inverted index and precesses list of queries"""
    print(f"read queries from: {query_file}", file=sys.stderr)
    inv_index = InvertedIndex.load_binary(inverted_index_filepath)
    for line in query_file:
        if file_flag:
            query = re.split(r'\s+', line)
            # query = line.split()
        else:
            query = line
        print(f"use query: {query}", file=sys.stderr)
        documents_ids = inv_index.query(words=query)
        number_of_documents = len(documents_ids)
        if number_of_documents == 0:
            print()
        for i in range(number_of_documents):
            if i == number_of_documents - 1:
                print(documents_ids[i])
            else:
                print(documents_ids[i], ",", end="", sep="")


def setup_parser(parser):
    """Parser tuning"""
    subparsers = parser.add_subparsers(help="choose command")
    build_parser = subparsers.add_parser(
        "build", help="build inverted index and save in binary format into hard drive",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    build_parser.add_argument(
        "--dataset", dest="dataset_path",
        default=DEFAULT_DATASET_PATH,
        help="path to dataset to load, default path is %(default)s",
    )
    build_parser.add_argument(
        "--output", default=DEFAULT_INVERTED_INDEX_STORE_PATH,
        help="path to store inverted index",
    )
    build_parser.set_defaults(callback=callback_build)
    query_parser = subparsers.add_parser(
        "query", help="query inverted index",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    query_parser.add_argument(
        "--index", default=DEFAULT_INVERTED_INDEX_STORE_PATH,
        dest="inverted_index_filepath",
        help="path to read inverted index in a binary form",
    )
    query_file_group = query_parser.add_mutually_exclusive_group(required=True)
    query_file_group.add_argument(
        "--query", dest="query", action='append', nargs='+',
        help="query for inverted index",
    )
    query_file_group.add_argument(
        "--query-file-utf8", dest="query_file", type=EncodedFileType("r", encoding="utf-8"),
        default=TextIOWrapper(sys.stdin.buffer, encoding="utf-8"),
        help="query file to get queries for inverted index",
    )
    query_file_group.add_argument(
        "--query-file-cp1251", dest="query_file", type=EncodedFileType("r", encoding="cp1251"),
        default=TextIOWrapper(sys.stdin.buffer, encoding="cp1251"),
        help="query file to get queries for inverted index",
    )
    query_parser.set_defaults(callback=callback_query)


def process_arguments(dataset_path: str, query: list) -> list:
    """Loads documents,builds inverted index from them and processes queries"""
    documents = load_documents(dataset_path)
    inverted_index = build_inverted_index(documents)
    document_ids = inverted_index.query(query)
    return document_ids


def main():
    """Main function"""
    parser = ArgumentParser(
        prog="inverted-index",
        description="tool to build,dump,load and query inverted index",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    print(arguments, file=sys.stderr)
    arguments.callback(arguments)


if __name__ == "__main__":
    main()