import unittest
from unittest.mock import patch


class TestGeoForgeDatasetGetSamplesBetween(unittest.TestCase):
    def test_returns_samples_in_inclusive_range(self) -> None:
        from geo_forge.dataset import GeoForgeDataset

        dataset = GeoForgeDataset.__new__(GeoForgeDataset)
        dataset.samples = [
            {"timestamp": 10},
            {"timestamp": 20},
            {"timestamp": 30},
            {"timestamp": 40},
        ]

        def _fake_getitem(self: GeoForgeDataset, idx: int) -> dict[str, object]:
            return {"idx": idx, "timestamp": self.samples[idx]["timestamp"]}

        with patch.object(GeoForgeDataset, "__getitem__", _fake_getitem):
            out = dataset.get_samples_between(20, 30)

        self.assertEqual([sample["idx"] for sample in out], [1, 2])

    def test_returns_samples_in_exclusive_range(self) -> None:
        from geo_forge.dataset import GeoForgeDataset

        dataset = GeoForgeDataset.__new__(GeoForgeDataset)
        dataset.samples = [
            {"timestamp": 10},
            {"timestamp": 20},
            {"timestamp": 30},
            {"timestamp": 40},
        ]

        def _fake_getitem(self: GeoForgeDataset, idx: int) -> dict[str, object]:
            return {"idx": idx, "timestamp": self.samples[idx]["timestamp"]}

        with patch.object(GeoForgeDataset, "__getitem__", _fake_getitem):
            out = dataset.get_samples_between(20, 40, inclusive=False)

        self.assertEqual([sample["idx"] for sample in out], [2])

    def test_raises_when_start_after_end(self) -> None:
        from geo_forge.dataset import GeoForgeDataset

        dataset = GeoForgeDataset.__new__(GeoForgeDataset)
        dataset.samples = [{"timestamp": 10}]

        with self.assertRaises(ValueError):
            dataset.get_samples_between(20, 10)


if __name__ == "__main__":
    unittest.main()
