"""Regression tests for the per-event cuts shared by the reconstruction stages."""

from math import radians
import unittest
from unittest.mock import patch

import numpy as np

from wcte.counting import (
    DEFAULT_TRACK_COUNTING,
    TrackCounting,
    optical_sources_resolved,
    track_counting,
    use_track_counting,
)
from wcte.engine import initialize


class RingDefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize()

    def test_cut_is_event_local(self):
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            with use_track_counting(TrackCounting(3, 3)):
                self.assertEqual(track_counting.get().min_unique_pmts, 3)
                raise RuntimeError("cancelled")
        self.assertEqual(track_counting.get(), DEFAULT_TRACK_COUNTING)

    def test_two_cone_seed_floor_stays_fixed(self):
        from wctering.seeding.search import seeded_event

        raw = dict(
            raw_pmt_ids=np.empty(0, dtype=np.int64),
            charges=np.empty(0),
            times=np.empty(0),
            pmt_id_space="wcsim",
        )
        with use_track_counting(TrackCounting(3.1, 3)):
            _, _, _, config = seeded_event(raw, 0)
        self.assertEqual(config.minimum_partition_pmts, 10)
        self.assertEqual(config.minimum_observable_ring_pmts, 4)
        self.assertEqual(config.minimum_three_ring_pmts, 4)

    def test_timed_daughter_obeys_both_settings(self):
        from pulse_residual import qualify

        fractions = np.eye(3)[[1, 2]]
        measured = [{"lower_statistical": 60.0}, {"lower_statistical": 22.0}]
        with (
            patch("joined_segments.separated_directions", return_value=(radians(34), True)),
            patch("timed_two_track.grouped_pulse", return_value=({}, measured)) as timing,
            patch("chain_comparison.counted", return_value=np.empty((2, 2))),
            patch("ring_supported.preserves_prompt", return_value=(False, [1])),
        ):
            with use_track_counting(TrackCounting(22.0, 15.0)):
                _, lower = qualify(None, (None, None), None, {}, fractions, {}, (8, 10))
            with use_track_counting(TrackCounting(22.1, 15.0)):
                _, higher = qualify(None, (None, None), None, {}, fractions, {}, (8, 10))
            with use_track_counting(TrackCounting(22.0, 40.0)):
                _, wide_turn = qualify(None, (None, None), None, {}, fractions, {}, (8, 10))
        self.assertIn("lost_prompt_hits", lower)
        self.assertNotIn("lost_prompt_hits", higher)
        self.assertNotIn("conditional_support", wide_turn)
        self.assertEqual(timing.call_count, 2)

    def test_angular_cut_distinguishes_nearby_and_detached_optical_sources(self):
        axes = [[0, 0, 1], [np.sin(radians(10)), 0, np.cos(radians(10))]]
        same_origin = [[0, 0, 0], [0, 0, 0]]
        separate_origins = [[0, 0, 0], [800, 0, 0]]
        self.assertFalse(optical_sources_resolved(axes, same_origin))
        self.assertTrue(optical_sources_resolved(axes, separate_origins))
        with use_track_counting(TrackCounting(30, 3)):
            self.assertTrue(optical_sources_resolved(axes, same_origin))
