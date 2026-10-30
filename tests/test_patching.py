import numpy as np
import torch

from noema import ca2d, circuit, data2d, gpt, patching, tokens

WORLD = {"side": 5, "frames": 8}
C = 25
STRIDE = C + 1

def tiny():
    torch.manual_seed(0)
    return gpt.GPT(gpt.Config(vocab=tokens.VOCAB, block=8 * STRIDE, emb=32, heads=2, layers=2)).eval()

def cases(direction, n=24, seed=0):
    return patching.make_cases(WORLD, np.random.default_rng(seed), n, direction, pool=2000)

def test_a_case_flips_exactly_the_neighbour_in_that_direction():
    """One token differs. Read as a grid it is the cell next to the predicted one, in the frame before,
    and the neighbourhood number of the predicted cell changes in exactly that neighbour's bit."""
    for direction, weight in patching.WEIGHT.items():
        cs = cases(direction)
        assert len(cs["q"]) == 24
        clean, corrupt = data2d.to_frames(cs["clean"].numpy()), data2d.to_frames(cs["corrupt"].numpy())
        for k in range(len(cs["q"])):
            assert (cs["clean"][k] != cs["corrupt"][k]).sum() == 1
            f, i = divmod(int(cs["q"][k]) + 1, STRIDE)
            r, c = divmod(i, 5)
            n_clean = ca2d.neighbourhoods(clean[k, f - 1])[r, c]
            n_corrupt = ca2d.neighbourhoods(corrupt[k, f - 1])[r, c]
            assert n_clean ^ n_corrupt == weight, direction

def test_a_cases_answer_is_the_clean_token_at_the_predicted_cell():
    for direction in patching.WEIGHT:
        cs = cases(direction)
        assert set(cs["answer"].tolist()) <= {0, 1}
        assert (cs["clean"][torch.arange(24), cs["q"] + 1] == cs["answer"]).all()      # the answer is the clean token

def test_the_distance_a_head_would_look_back_is_the_one_the_circuit_names():
    names = circuit.named_offsets_2d(5, 5)
    for direction in ("up", "left", "self", "right", "down"):
        assert names[patching.expected_distance(direction, 5)] == direction

def test_score_is_positive_for_the_clean_answer_whichever_it_is():
    logits = torch.zeros(2, 3, 3)
    logits[0, 1, 1] = 5.0                                      # favours 1
    logits[1, 2, 0] = 5.0                                      # favours 0
    q, answer = torch.tensor([1, 2]), torch.tensor([1, 0])
    assert (patching.score(logits, q, answer) > 0).all()
    assert (patching.score(logits, q, 1 - answer) < 0).all()

def test_record_keeps_what_every_head_writes_at_the_query():
    model, cs = tiny(), cases("up", 8)
    rec, logits = patching.record(model, cs["clean"], cs["q"])
    assert len(rec) == 2 and rec[0].shape == (8, 2, 16)
    assert logits.shape == (8, 8 * STRIDE, 3)

def test_writing_a_head_its_own_output_changes_nothing():
    model, cs = tiny(), cases("up", 8)
    rec, logits = patching.record(model, cs["clean"], cs["q"])
    again = patching.with_head_from(model, cs["clean"], cs["q"], 1, 0, rec[1][:, 0])
    assert torch.allclose(again, logits, atol=1e-5)

def test_writing_a_head_another_output_changes_the_logits_at_the_query_only_from_there_on():
    model, cs = tiny(), cases("up", 8)
    rec, logits = patching.record(model, cs["clean"], cs["q"])
    other = patching.with_head_from(model, cs["clean"], cs["q"], 0, 1, rec[0][:, 1] * 5 + 1)
    idx = torch.arange(8)
    assert not torch.allclose(other[idx, cs["q"]], logits[idx, cs["q"]])
    before = torch.arange(logits.shape[1])[None, :] < cs["q"][:, None]
    assert torch.allclose(other[before], logits[before], atol=1e-5)             # causal: earlier positions do not move

def test_patching_every_head_of_a_layer_pair_gives_shares_that_are_finite():
    model, cs = tiny(), cases("down", 12)
    r = patching.patch_all(model, cs)
    assert r["denoise"].shape == (2, 2) and np.isfinite(r["denoise"]).all() and np.isfinite(r["noise"]).all()

def test_a_wrapped_flip_changes_the_neighbourhood_of_the_edge_cell():
    for direction in ("up", "left", "right", "down"):
        cases = patching.make_cases(WORLD, np.random.default_rng(3), 8, direction, pool=2000, wrap=True)
        side, stride = WORLD["side"], WORLD["side"] ** 2 + 1
        for clean, corrupt, q in zip(cases["clean"].numpy(), cases["corrupt"].numpy(), cases["q"].tolist()):
            flips = np.flatnonzero(clean != corrupt)
            assert len(flips) == 1
            assert q - flips[0] == patching.wrapped_distance(direction, side)
            frame, cell = (q + 1) // stride, (q + 1) % stride
            r, c = cell // side, cell % side
            assert {"up": r == 0, "down": r == side - 1, "left": c == 0, "right": c == side - 1}[direction]

def test_a_flip_in_1d_is_one_row_back_at_its_offset():
    for offset in (-2, 0, 2):
        cases = patching.make_cases_1d(np.random.default_rng(4), 8, offset, pool=2000)
        for clean, corrupt, q in zip(cases["clean"].numpy(), cases["corrupt"].numpy(), cases["q"].tolist()):
            flips = np.flatnonzero(clean != corrupt)
            assert len(flips) == 1 and q - flips[0] == patching.WIDTH_1D - offset
