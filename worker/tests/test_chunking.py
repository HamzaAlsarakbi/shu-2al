from shu2al_worker.chunking import batches_by_length, max_new_tokens, merge_regions


def test_merges_close_regions_up_to_max_length():
    regions = [(0.0, 2.0), (2.3, 4.0), (4.2, 9.0), (12.0, 13.0)]
    assert merge_regions(regions, max_chunk_s=10, max_gap_s=0.5) == [(0.0, 9.0), (12.0, 13.0)]


def test_respects_max_length_and_gap():
    regions = [(0.0, 6.0), (6.2, 12.0), (12.1, 13.0)]
    assert merge_regions(regions, max_chunk_s=10, max_gap_s=0.5) == [(0.0, 6.0), (6.2, 13.0)]
    assert merge_regions([(0.0, 1.0), (1.6, 2.0)], max_chunk_s=10, max_gap_s=0.5) == [
        (0.0, 1.0),
        (1.6, 2.0),
    ]


def test_splits_overlong_region_evenly():
    chunks = merge_regions([(0.0, 45.0)], max_chunk_s=20, max_gap_s=0.5)
    assert chunks == [(0.0, 15.0), (15.0, 30.0), (30.0, 45.0)]
    assert all(e - s <= 20 for s, e in chunks)


def test_ignores_empty_and_unsorted():
    assert merge_regions([(5.0, 6.0), (3.0, 3.0), (1.0, 2.0)], 20, 0.5) == [(1.0, 2.0), (5.0, 6.0)]


def test_batches_longest_first():
    chunks = [(0, 1), (0, 5), (0, 3), (0, 4)]
    assert batches_by_length(chunks, 2) == [[1, 3], [2, 0]]


def test_token_budget_is_capped():
    assert max_new_tokens(1) == 44
    assert max_new_tokens(1000) == 448
