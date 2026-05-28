"""Tests for dataset and corpus builders."""

import pytest
from src.rdru import CharDataset


def test_chardataset_simple() -> None:
    text = "hello world"
    ds = CharDataset(text, seq_len=4)
    assert ds.vocab_size == 8  # h,e,l,o,' ',w,r,d
    assert len(ds) == 2
    x, y = ds[0]
    assert x.shape == (4,)
    assert y.shape == (4,)


def test_chardataset_no_overlap() -> None:
    text = "abcdefghij"
    ds = CharDataset(text, seq_len=5)
    assert len(ds) == 1
    x, y = ds[0]
    assert x.tolist() == [0, 1, 2, 3, 4]
    assert y.tolist() == [1, 2, 3, 4, 5]


def test_chardataset_vocab_consistency() -> None:
    text = "abracadabra"
    ds = CharDataset(text, seq_len=3)
    # stoi and itos should be inverses
    for c, i in ds.stoi.items():
        assert ds.itos[i] == c
    assert ds.vocab_size == len(set(text))
