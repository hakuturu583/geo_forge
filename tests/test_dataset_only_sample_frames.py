import unittest
from unittest.mock import Mock, patch


class _DummyNuScenes:
    def __init__(self, sample_data_by_token: dict[str, dict[str, object]]) -> None:
        self._sample_data_by_token = sample_data_by_token

    def get(self, table_name: str, token: str) -> dict[str, object]:
        if table_name != "sample_data":
            raise KeyError(f"Unsupported table_name {table_name}")
        return self._sample_data_by_token[token]


class TestGeoForgeDatasetOnlySampleFrames(unittest.TestCase):
    def test_pose_index_uses_synchronized_samples_when_enabled(self) -> None:
        from geo_forge.dataset import GeoForgeDataset

        dataset = GeoForgeDataset.__new__(GeoForgeDataset)
        dataset.nusc = _DummyNuScenes(
            {
                "cam_token": {
                    "ego_pose_token": "ego_pose",
                    "calibrated_sensor_token": "calib",
                }
            }
        )
        dataset.scene_filter = {"scene-1"}
        dataset.camera_filter = {"cam_front"}
        dataset.only_sample_frames = True

        sample_info = {
            "scene_name": "scene-1",
            "timestamp": 222,
            "cameras": {"CAM_FRONT": {"token": "cam_token", "timestamp": 111}},
            "is_key_frame": True,
        }

        sync_iter = Mock(return_value=iter([sample_info]))
        sweep_iter = Mock(return_value=iter([]))
        with patch("geo_forge.dataset.iterate_synchronized_samples", sync_iter), patch(
            "geo_forge.dataset.iterate_all_sweep_camera_frames", sweep_iter
        ):
            pose_index = dataset._build_pose_index()

        sync_iter.assert_called_once()
        sweep_iter.assert_not_called()

        self.assertIn(("scene-1", "cam_front", 111), pose_index)
        self.assertIn(("scene-1", "cam_front", 222), pose_index)

    def test_pose_index_uses_sweeps_when_disabled(self) -> None:
        from geo_forge.dataset import GeoForgeDataset

        dataset = GeoForgeDataset.__new__(GeoForgeDataset)
        dataset.nusc = _DummyNuScenes(
            {
                "cam_token": {
                    "ego_pose_token": "ego_pose",
                    "calibrated_sensor_token": "calib",
                }
            }
        )
        dataset.scene_filter = None
        dataset.camera_filter = None
        dataset.only_sample_frames = False

        sweep_info = {
            "scene_name": "scene-1",
            "timestamp": 333,
            "cameras": {"CAM_FRONT": {"token": "cam_token", "timestamp": 333}},
            "is_key_frame": False,
        }

        sync_iter = Mock(return_value=iter([]))
        sweep_iter = Mock(return_value=iter([sweep_info]))
        with patch("geo_forge.dataset.iterate_synchronized_samples", sync_iter), patch(
            "geo_forge.dataset.iterate_all_sweep_camera_frames", sweep_iter
        ):
            pose_index = dataset._build_pose_index()

        sweep_iter.assert_called_once()
        sync_iter.assert_not_called()

        self.assertIn(("scene-1", "cam_front", 333), pose_index)
        # No extra lidar-timestamp key for sweeps.
        self.assertEqual(len(pose_index), 1)


if __name__ == "__main__":
    unittest.main()
