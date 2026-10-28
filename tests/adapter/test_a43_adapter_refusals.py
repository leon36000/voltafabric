"""A43 refusal pins for adapters/skinny_moe.py; CPU fake providers only.

Every case makes exactly one operand of one ``if ...: raise`` guard true and
asserts the exception type and the guard's full message (anchored), so a later
check refusing with a different message cannot mask a removed operand. The
"meta" device stands in for a foreign device without any GPU.
"""

import re
from types import SimpleNamespace
import unittest
import torch
from adapters import skinny_moe as adapter
from tests.adapter.test_skinny_adapter import align_reference, layer

META = torch.device("meta")


def exact(message):
    return "^" + re.escape(message) + "$"


# ---------------------------------------------------------------- fixtures


def checkpoint(e=1, n13=128, k13=64, n2=64, k2=64):
    """Valid checkpoint staging planes for prepack_experts (shape is overridable)."""
    return SimpleNamespace(
        w13_weight=torch.zeros((e, n13, k13 // 2), dtype=torch.uint8),
        w13_weight_scale=torch.full((e, n13, k13 // 16), 0x38, dtype=torch.uint8).view(
            torch.float8_e4m3fn
        ),
        w13_weight_scale_2=torch.ones((e, 2)),
        w2_weight=torch.zeros((e, n2, k2 // 2), dtype=torch.uint8),
        w2_weight_scale=torch.full((e, n2, k2 // 16), 0x38, dtype=torch.uint8).view(
            torch.float8_e4m3fn
        ),
        w2_weight_scale_2=torch.ones((e,)),
    )


class RecordingPrepack:
    def __init__(self):
        self.calls = 0

    def __call__(self, codes, scales):
        self.calls += 1
        return codes.reshape(-1).clone(), scales.reshape(-1).clone()


class Provider:
    """Explicit CPU contract provider: zero FP16 rows, optional per-call corruption."""

    def __init__(self, hooks=None):
        self.hooks = hooks or {}
        self.calls = 0

    def gemm_qpn2_moe(self, x, *args):
        s, n = args[6], args[7]
        hook = self.hooks.get(self.calls)
        self.calls += 1
        if hook is not None:
            return hook(s, n)
        return torch.zeros((s, n), dtype=torch.float16)


def local(l, x=None, weights=None, ids=None, provider=None):
    ids = torch.tensor([[0]]) if ids is None else ids
    return adapter.moe_apply_local(
        l,
        torch.full((1, 64), 0.01) if x is None else x,
        torch.ones(ids.shape) if weights is None else weights,
        ids,
        None,
        None,
        kernel=Provider() if provider is None else provider,
        align=align_reference,
    )


def stamp(**override):
    s = dict(run_id="a43", model_revision="m", plan_id="p", layer=0, batch=1, epoch=7)
    s.update(override)
    return s


def bound_layer():
    l = layer()
    l.expert_map = torch.full((100,), -1, dtype=torch.int64)
    l.expert_map[99] = 0
    l.expert_map[10] = 1
    return l


def rank(st):
    return dict(
        stamp=dict(st),
        rank=2,
        hidden=64,
        tokens=1,
        resident_order=[900, 3],
        slots=[
            dict(
                physical_expert=1,
                slot=dict(token=0, slot=0, expert=10, rank=2, local=3, weight=0.2),
            ),
            dict(
                physical_expert=0,
                slot=dict(token=0, slot=1, expert=99, rank=2, local=900, weight=0.8),
            ),
        ],
    )


IDS = torch.tensor([[10, 99]])
WEIGHTS = torch.tensor([[0.2, 0.8]])


# ------------------------------------------------- _validate_sm70_scale_domain


class Sm70ScaleDomainRefusals(unittest.TestCase):
    DIMS = exact("SM70 scale domain dimensions or dtypes differ")

    def domain(self, k=64, tiles=1, scales=None, gscales=None):
        tile_bytes = max(k // 16 * 32, 0)
        if scales is None:
            scales = torch.full((tiles * tile_bytes,), 0x38, dtype=torch.uint8)
        if gscales is None:
            gscales = torch.ones((tiles,))
        return adapter._validate_sm70_scale_domain(scales, gscales, k)

    def test_controls_are_admitted(self):
        self.assertIsNone(self.domain(k=64))
        self.assertIsNone(self.domain(k=65536, gscales=torch.full((1,), 2**-10)))

    def test_k_outside_range_but_tile_aligned(self):
        # k is a multiple of 64, so only the range operand is true.
        for k, tiles in ((0, 0), (-64, 0), (65600, 1)):
            with self.subTest(k=k):
                with self.assertRaisesRegex(ValueError, self.DIMS):
                    self.domain(k=k, tiles=tiles, gscales=torch.full((tiles,), 2**-10))

    def test_k_in_range_but_not_tile_aligned(self):
        with self.assertRaisesRegex(ValueError, self.DIMS):
            self.domain(k=96)

    def test_scale_and_factor_dtypes(self):
        cases = {
            "int16 scales": dict(scales=torch.full((128,), 0x38, dtype=torch.int16)),
            "float64 factors": dict(gscales=torch.ones((1,), dtype=torch.float64)),
        }
        for label, kwargs in cases.items():
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, self.DIMS):
                    self.domain(**kwargs)

    def test_plane_lengths(self):
        with self.assertRaisesRegex(
            ValueError, exact("SM70 scale domain plane lengths differ")
        ):
            self.domain(scales=torch.full((256,), 0x38, dtype=torch.uint8))

    def test_nonfinite_global_factor_without_nonpositive(self):
        # nan/inf are not <= 0; without the finiteness operand the half-scale
        # overflow check would refuse instead, with a different message.
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, exact("SM70 global factors must be positive finite")
                ):
                    self.domain(gscales=torch.tensor([value]))

    def test_infinite_half_factor_with_zero_group_scale(self):
        # 4.0 * 16384 rounds to FP16 inf; a zero group scale makes scale*gm NaN,
        # so only the gm finiteness operand refuses.
        with self.assertRaisesRegex(
            ValueError, exact("SM70 half-scale intermediate overflow")
        ):
            self.domain(
                scales=torch.zeros((128,), dtype=torch.uint8),
                gscales=torch.tensor([4.0]),
            )


# ----------------------------------------------------------- _prepack_stacked


class PrepackStackedRefusals(unittest.TestCase):
    def test_exporter_returning_none_plane(self):
        codes = torch.zeros((1, 32, 32), dtype=torch.uint8)
        scales = torch.full((1, 32, 4), 0x38, dtype=torch.uint8)
        for label, prepack in (
            ("codes", lambda c, s: (None, s.clone())),
            ("scales", lambda c, s: (c.clone(), None)),
        ):
            with self.subTest(missing=label):
                with self.assertRaisesRegex(
                    ValueError, exact("expert shape not tile-aligned")
                ):
                    adapter._prepack_stacked(codes, scales, prepack)


# ------------------------------------------------------------ prepack_experts


class PrepackExpertsRefusals(unittest.TestCase):
    SHAPES = exact(
        "checkpoint gate/up/down resident shapes differ or are not tile aligned"
    )

    def refuse(self, l, message):
        prepack = RecordingPrepack()
        with self.assertRaisesRegex(ValueError, message):
            adapter.prepack_experts(l, prepack=prepack)
        self.assertEqual(prepack.calls, 0)
        self.assertFalse(hasattr(l, "skinny_moe_qc13"))

    def test_control_checkpoint_is_admitted(self):
        l = checkpoint()
        adapter.prepack_experts(l, prepack=RecordingPrepack())
        self.assertEqual(l.skinny_moe_shape, (1, 128, 64, 64, 64))

    def test_code_planes_must_be_three_dimensional(self):
        for name in ("w13_weight", "w2_weight"):
            with self.subTest(plane=name):
                l = checkpoint()
                setattr(l, name, getattr(l, name)[0])
                self.refuse(
                    l, exact("checkpoint expert codes must have three dimensions")
                )

    def test_single_shape_operand(self):
        # Each tuple is (e, n13, k13, n2, k2) satisfying every other operand:
        # n13 // 2 == k2, n2 == k13, and the remaining alignments.
        # `not n13`/`not k2`, `not n2`/`not k13` and `n2 % 32` cannot be made
        # the sole reason (see report); they are not forced here.
        cases = {
            "more than 1024 experts": (1025, 128, 64, 64, 64),
            "n13 not a multiple of 64": (1, 129, 64, 64, 64),
            "k13 not a multiple of 64": (1, 128, 96, 96, 64),
            "k2 not a multiple of 64": (1, 64, 64, 64, 32),
            "gate/up width differs from down K": (1, 128, 64, 64, 128),
            "down width differs from gate/up K": (1, 128, 64, 128, 64),
        }
        for label, (e, n13, k13, n2, k2) in cases.items():
            with self.subTest(case=label):
                self.refuse(checkpoint(e, n13, k13, n2, k2), self.SHAPES)

    def test_expert_counts_differ(self):
        l = checkpoint()
        l.w2_weight = torch.zeros((2, 64, 32), dtype=torch.uint8)
        self.refuse(l, self.SHAPES)

    def test_code_dtypes(self):
        for name in ("w13_weight", "w2_weight"):
            with self.subTest(plane=name):
                l = checkpoint()
                setattr(l, name, getattr(l, name).to(torch.int8))
                self.refuse(l, exact("checkpoint codes must be uint8"))

    def test_group_scale_dtypes(self):
        for name in ("w13_weight_scale", "w2_weight_scale"):
            with self.subTest(plane=name):
                l = checkpoint()
                setattr(l, name, getattr(l, name).view(torch.int8))
                self.refuse(l, exact("checkpoint group scales must encode F8_E4M3"))

    def test_group_scale_shapes(self):
        for name in ("w13_weight_scale", "w2_weight_scale"):
            with self.subTest(plane=name):
                l = checkpoint()
                setattr(l, name, getattr(l, name)[:, :, :3])
                self.refuse(l, exact("checkpoint group scale shape differs from codes"))

    def test_global_factor_shapes_and_dtypes(self):
        message = exact(
            "checkpoint global factors must be distinct F32 gate/up and down"
        )
        cases = {
            "gate/up factors not (e, 2)": ("w13_weight_scale_2", torch.ones((1, 3))),
            "down factors scalar": ("w2_weight_scale_2", torch.tensor(1.0)),
            "gate/up factors float64": (
                "w13_weight_scale_2",
                torch.ones((1, 2), dtype=torch.float64),
            ),
            "down factors float64": (
                "w2_weight_scale_2",
                torch.ones((1,), dtype=torch.float64),
            ),
        }
        for label, (name, value) in cases.items():
            with self.subTest(case=label):
                l = checkpoint()
                setattr(l, name, value)
                self.refuse(l, message)

    def test_planes_on_a_foreign_device(self):
        for name in (
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_scale_2",
            "w2_weight_scale_2",
        ):
            with self.subTest(plane=name):
                l = checkpoint()
                setattr(l, name, getattr(l, name).to(META))
                self.refuse(l, exact("checkpoint planes must share a device"))

    def test_nonfinite_or_nonpositive_global_factor(self):
        # Without this check the SM70 domain admission would refuse with its own
        # message, so the anchored message is what pins it.
        cases = [
            ("w13_weight_scale_2", torch.tensor([[1.0, float("nan")]])),
            ("w13_weight_scale_2", torch.tensor([[float("inf"), 1.0]])),
            ("w2_weight_scale_2", torch.tensor([0.0])),
            ("w2_weight_scale_2", torch.tensor([-1.0])),
        ]
        for name, value in cases:
            with self.subTest(plane=name, value=value.tolist()):
                l = checkpoint()
                setattr(l, name, value)
                self.refuse(
                    l, exact("checkpoint global factor is nonfinite or nonpositive")
                )

    def test_signed_or_nonfinite_scale_bytes(self):
        for name in ("w13_weight_scale", "w2_weight_scale"):
            for byte in (0x7F, 0x80):
                with self.subTest(plane=name, byte=byte):
                    l = checkpoint()
                    raw = getattr(l, name).view(torch.uint8).clone()
                    raw.view(-1)[-1] = byte
                    setattr(l, name, raw.view(torch.float8_e4m3fn))
                    self.refuse(
                        l,
                        exact("checkpoint group scales must be unsigned finite E4M3"),
                    )


# --------------------------------------------------------- _select_local_slots


class SelectLocalSlotsRefusals(unittest.TestCase):
    def test_control(self):
        positions, _, locals_ = adapter._select_local_slots(
            torch.tensor([[0, 1]]), None, 2, torch.tensor([[True, True]])
        )
        self.assertEqual(positions.tolist(), [0, 1])
        self.assertEqual(locals_.tolist(), [0, 1])

    def test_topk_ids_rank_and_dtype(self):
        for label, ids in (
            ("one-dimensional", torch.tensor([0])),
            ("floating", torch.tensor([[0.0, 1.0]])),
        ):
            with self.subTest(case=label):
                with self.assertRaisesRegex(
                    ValueError,
                    exact("topk_ids must be an integer [tokens,topk] tensor"),
                ):
                    adapter._select_local_slots(ids, None, 2, None)

    def test_negative_local_count_or_empty_topk(self):
        for label, ids, count in (
            ("negative local experts", torch.tensor([[0]]), -1),
            ("empty top-k", torch.empty((1, 0), dtype=torch.int64), 2),
        ):
            with self.subTest(case=label):
                with self.assertRaisesRegex(
                    ValueError, exact("invalid local expert count or empty top-k")
                ):
                    adapter._select_local_slots(ids, None, count, None)

    def test_expert_map_rank_dtype_and_device(self):
        message = exact(
            "expert_map must be an integer global-to-physical vector on the same device"
        )
        for label, mapping in (
            ("two-dimensional", torch.tensor([[0, 1]])),
            ("floating", torch.tensor([0.0, 1.0])),
            ("foreign device", torch.tensor([0, 1], device=META)),
        ):
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, message):
                    adapter._select_local_slots(torch.tensor([[0]]), mapping, 2, None)

    def test_selected_mask_shape_and_device(self):
        message = exact("selected_mask must identify exact routed copies")
        ids = torch.tensor([[0, 1]])
        for label, mask in (
            # Same element count, so a flattened use would silently accept it.
            ("flat shape", torch.tensor([True, True])),
            ("foreign device", torch.ones((1, 2), dtype=torch.bool, device=META)),
        ):
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, message):
                    adapter._select_local_slots(ids, None, 2, mask)


# ---------------------------------------------------------- bind_rank_dispatch


class BindRankDispatchRefusals(unittest.TestCase):
    def bind(self, r, st, ids=IDS, weights=WEIGHTS):
        return adapter.bind_rank_dispatch(bound_layer(), r, st, ids, weights)

    def test_control(self):
        st = stamp()
        self.assertEqual(self.bind(rank(st), st).tolist(), [[True, True]])

    def test_route_tensor_rank_and_weight_shape(self):
        st = stamp()
        for label, ids, weights in (
            (
                "one-dimensional routes",
                torch.tensor([10, 99]),
                torch.tensor([0.2, 0.8]),
            ),
            ("weight shape differs", IDS, torch.tensor([[0.2]])),
        ):
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, exact("route tensors differ")):
                    self.bind(rank(st), st, ids, weights)

    def test_incomplete_stamp_even_when_it_matches(self):
        message = exact("rank dispatch has stale or incomplete Stamp")
        missing = stamp()
        del missing["epoch"]
        extra = stamp(shard=0)
        for label, st in (("missing epoch", missing), ("extra key", extra)):
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, message):
                    self.bind(rank(st), st)

    def test_dispatch_dimensions(self):
        st = stamp()
        for key, value in (("hidden", 128), ("tokens", 2)):
            with self.subTest(key=key):
                r = rank(st)
                r[key] = value
                with self.assertRaisesRegex(
                    ValueError,
                    exact("rank dispatch dimensions differ from tensor frontier"),
                ):
                    self.bind(r, st)

    def test_resident_order(self):
        st = stamp()
        message = exact("physical resident order differs from stack")
        tuple_order = rank(st)
        tuple_order["resident_order"] = (900, 3)
        # Length differs while the distinct count still equals the stack size.
        repeated = rank(st)
        repeated["resident_order"] = [900, 3, 3]
        # Right length, one distinct local index; slot locals agree with it.
        aliased = rank(st)
        aliased["resident_order"] = [900, 900]
        aliased["slots"][0]["slot"]["local"] = 900
        for label, r in (
            ("not a list", tuple_order),
            ("longer with repeat", repeated),
            ("repeated local index", aliased),
        ):
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, message):
                    self.bind(r, st)

    def test_slot_index_outside_frontier(self):
        st = stamp()
        message = exact("rank slot index outside tensor frontier")
        cases = {
            "token past end": ("slot", "token", 1),
            "negative token": ("slot", "token", -1),
            "slot past end": ("slot", "slot", 2),
            "negative slot": ("slot", "slot", -1),
            "physical past stack": (None, "physical_expert", 2),
            "negative physical": (None, "physical_expert", -1),
        }
        for label, (inner, key, value) in cases.items():
            with self.subTest(case=label):
                r = rank(st)
                target = r["slots"][0] if inner is None else r["slots"][0][inner]
                target[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    self.bind(r, st)

    def test_foreign_rank_slot(self):
        st = stamp()
        r = rank(st)
        r["slots"][0]["slot"]["rank"] = 3
        with self.assertRaisesRegex(
            ValueError, exact("duplicate slot or foreign physical resident")
        ):
            self.bind(r, st)

    def test_physical_expert_differs_from_mapping(self):
        # The global expert matches the tensor and the local index matches the
        # resident order at the claimed physical slot; only the mapping differs.
        st = stamp()
        r = rank(st)
        r["slots"][0]["physical_expert"] = 0
        r["slots"][0]["slot"]["local"] = 900
        with self.assertRaisesRegex(
            ValueError, exact("dispatch global/local expert differs from tensor route")
        ):
            self.bind(r, st)

    def test_coefficient_nonfinite_or_negative_even_when_equal(self):
        st = stamp()
        message = exact("dispatch coefficient differs from tensor route")
        for value in (float("inf"), -0.2):
            with self.subTest(weight=value):
                r = rank(st)
                r["slots"][0]["slot"]["weight"] = value
                weights = WEIGHTS.clone()
                weights[0, 0] = value
                with self.assertRaisesRegex(ValueError, message):
                    self.bind(r, st, weights=weights)


# ------------------------------------------------------------- moe_apply_local


class MoeApplyLocalRefusals(unittest.TestCase):
    def refuse(self, message, l=None, provider=None, **kwargs):
        provider = Provider() if provider is None else provider
        with self.assertRaisesRegex(ValueError, exact(message)):
            local(layer() if l is None else l, provider=provider, **kwargs)
        return provider

    def test_control(self):
        provider = Provider()
        positions, h2 = local(layer(), provider=provider)
        self.assertEqual(positions.tolist(), [0])
        self.assertEqual(tuple(h2.shape), (1, 64))
        self.assertEqual(provider.calls, 2)

    def test_token_route_weight_shapes(self):
        message = "token/route/weight shape mismatch"
        cases = {
            "three-dimensional x": dict(x=torch.full((1, 64, 1), 0.01)),
            "one-dimensional routes": dict(
                ids=torch.tensor([0]), weights=torch.ones((1,))
            ),
            "weight shape differs": dict(weights=torch.ones((1, 2))),
            "token count differs": dict(x=torch.full((2, 64), 0.01)),
        }
        for label, kwargs in cases.items():
            with self.subTest(case=label):
                provider = self.refuse(message, **kwargs)
                self.assertEqual(provider.calls, 0)

    def test_hidden_and_resident_shape_alignment(self):
        message = "QPN2 MoE gate/up/down shapes are not tile aligned"
        with self.subTest(case="hidden differs"):
            self.refuse(message, x=torch.full((1, 128), 0.01))
        # (e, n13, k13, n2, k2); every other operand of the guard holds.
        # `n2 % 32` is implied by n2 == k13 and k13 % 64 and is not forced.
        cases = {
            "n13 not a multiple of 64": ((2, 129, 64, 64, 64), 64),
            "k13 not a multiple of 64": ((2, 128, 96, 96, 64), 96),
            "k2 not a multiple of 64": ((2, 64, 64, 64, 32), 64),
            "gate/up width differs from down K": ((2, 128, 64, 64, 128), 64),
            "down width differs from gate/up K": ((2, 128, 64, 128, 64), 64),
        }
        for label, (shape, hidden) in cases.items():
            with self.subTest(case=label):
                l = layer()
                l.skinny_moe_shape = shape
                self.refuse(message, l=l, x=torch.full((1, hidden), 0.01))

    def test_activation_and_coefficient_dtypes(self):
        message = "floating activations and coefficients required"
        for label, kwargs in (
            ("integer activations", dict(x=torch.ones((1, 64), dtype=torch.int64))),
            (
                "integer coefficients",
                dict(weights=torch.ones((1, 1), dtype=torch.int64)),
            ),
        ):
            with self.subTest(case=label):
                self.refuse(message, **kwargs)

    def test_route_devices(self):
        message = "routes and activations must share the device"
        for label, kwargs in (
            ("routes", dict(ids=torch.tensor([[0]], device=META))),
            ("coefficients", dict(weights=torch.ones((1, 1), device=META))),
        ):
            with self.subTest(case=label):
                self.refuse(message, **kwargs)

    def test_nonfinite_input_or_invalid_coefficient(self):
        message = "nonfinite input or invalid routing coefficient"
        cases = {
            "nan activation": dict(x=torch.full((1, 64), float("nan"))),
            "inf coefficient": dict(weights=torch.tensor([[float("inf")]])),
            "nan coefficient": dict(weights=torch.tensor([[float("nan")]])),
            "negative coefficient": dict(weights=torch.tensor([[-1.0]])),
        }
        for label, kwargs in cases.items():
            with self.subTest(case=label):
                provider = self.refuse(message, **kwargs)
                self.assertEqual(provider.calls, 0)

    def test_packed_plane_dtypes(self):
        message = "packed QPN2 plane dtypes differ from ABI"
        for name, dtype in (
            ("skinny_moe_qc13", torch.int8),
            ("skinny_moe_qc2", torch.int8),
            ("skinny_moe_qs13", torch.int16),
            ("skinny_moe_gs2", torch.float64),
        ):
            with self.subTest(plane=name):
                l = layer()
                setattr(l, name, getattr(l, name).to(dtype))
                self.assertEqual(self.refuse(message, l=l).calls, 0)

    def test_packed_plane_lengths(self):
        message = "packed QPN2 plane lengths differ from resident shape"
        cases = {
            "codes short": ("skinny_moe_qc13", lambda t: t[:-1].clone()),
            "scales short": ("skinny_moe_qs2", lambda t: t[:-1].clone()),
            # Same element count, wrong shape.
            "factors flattened": ("skinny_moe_gs13", lambda t: t.reshape(-1).clone()),
        }
        for label, (name, change) in cases.items():
            with self.subTest(case=label):
                l = layer()
                setattr(l, name, change(getattr(l, name)))
                self.assertEqual(self.refuse(message, l=l).calls, 0)

    def test_nonfinite_or_nonpositive_packed_factor(self):
        message = (
            "packed QPN2 scale must be unsigned finite E4M3 and global factor positive"
        )
        for name in ("skinny_moe_gs13", "skinny_moe_gs2"):
            for value in (float("inf"), 0.0, -1.0):
                with self.subTest(plane=name, value=value):
                    l = layer()
                    getattr(l, name).view(-1)[0] = value
                    self.assertEqual(self.refuse(message, l=l).calls, 0)

    def test_packed_planes_contiguous_on_activation_device(self):
        message = "packed QPN2 planes must be contiguous on the activation device"

        def strided(t):
            return torch.stack([t, t], dim=-1)[..., 0]

        cases = {
            "strided codes": ("skinny_moe_qc13", strided),
            "strided scales": ("skinny_moe_qs2", strided),
            "transposed factors": (
                "skinny_moe_gs13",
                lambda t: t.t().contiguous().t(),
            ),
            "codes on foreign device": ("skinny_moe_qc2", lambda t: t.to(META)),
        }
        for label, (name, change) in cases.items():
            with self.subTest(case=label):
                l = layer()
                setattr(l, name, change(getattr(l, name)))
                self.assertFalse(
                    getattr(l, name).is_contiguous()
                    and getattr(l, name).device == torch.device("cpu")
                )
                self.assertEqual(self.refuse(message, l=l).calls, 0)

    def test_malformed_gate_up_provider_buffer(self):
        message = "gate/up provider returned a malformed compact buffer"
        hooks = {
            "extra rows": lambda s, n: torch.zeros((s + 1, n), dtype=torch.float16),
            "float32": lambda s, n: torch.zeros((s, n), dtype=torch.float32),
            "foreign device": lambda s, n: torch.zeros(
                (s, n), dtype=torch.float16, device=META
            ),
        }
        for label, hook in hooks.items():
            with self.subTest(case=label):
                provider = self.refuse(message, provider=Provider({0: hook}))
                self.assertEqual(provider.calls, 1)

    def test_nonfinite_gate_up_output(self):
        provider = self.refuse(
            "nonfinite gate/up provider output",
            provider=Provider(
                {0: lambda s, n: torch.full((s, n), float("nan"), dtype=torch.float16)}
            ),
        )
        self.assertEqual(provider.calls, 1)

    def test_malformed_down_provider_buffer(self):
        message = "down provider returned a malformed compact buffer"
        hooks = {
            "extra column": lambda s, n: torch.zeros((s, n + 1), dtype=torch.float16),
            "float32": lambda s, n: torch.zeros((s, n), dtype=torch.float32),
            "foreign device": lambda s, n: torch.zeros(
                (s, n), dtype=torch.float16, device=META
            ),
        }
        for label, hook in hooks.items():
            with self.subTest(case=label):
                self.refuse(message, provider=Provider({1: hook}))

    def test_nonfinite_down_output(self):
        self.refuse(
            "nonfinite down provider output",
            provider=Provider(
                {1: lambda s, n: torch.full((s, n), float("inf"), dtype=torch.float16)}
            ),
        )


if __name__ == "__main__":
    unittest.main()
