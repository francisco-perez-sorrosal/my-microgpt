from my_microgpt.dataset import load_docs


def test_load_docs(tmp_path):
    input_file = tmp_path / "names.txt"
    input_file.write_text("alice\nbob\ncharlie\n")

    docs = load_docs(path=str(input_file))

    assert len(docs) == 3
    assert set(docs) == {"alice", "bob", "charlie"}


def test_load_docs_skips_blank_lines(tmp_path):
    input_file = tmp_path / "names.txt"
    input_file.write_text("alice\n\n  \nbob\n")

    docs = load_docs(path=str(input_file))

    assert len(docs) == 2
    assert set(docs) == {"alice", "bob"}
