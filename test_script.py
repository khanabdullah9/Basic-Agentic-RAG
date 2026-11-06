from vector import create_vector_store, collection_exists
import os
from utils import log_err, get_file_name

# def test_get_file_name():
#     assert isinstance(get_file_name(os.path.join("DOCS","HP1.pdf")), str)

# def test_create_vector_store():
#     assert create_vector_store(os.path.join("DOCS","HP1.pdf")) == True

def test_collection_exists():
    assert collection_exists(os.path.join("DOCS","HP1.pdf")) == True
