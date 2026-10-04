from pathlib import Path

from src.config import read_config


def test_relative_paths_are_resolved_from_project_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    config_dir = project / "config"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "feature_config.yaml"
    config_path.write_text(
        """
project:
  max_len: 50
paths:
  dataset_root: ../data/videos
  label_file: ../data/videos/labels.xlsx
  work_dir: work
  output_dir: results
  ffmpeg: ffmpeg
  openface_feature_extraction: FeatureExtraction
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = read_config(config_path)

    assert config["paths"]["dataset_root"] == str((project / "../data/videos").resolve())
    assert config["paths"]["label_file"] == str(
        (project / "../data/videos/labels.xlsx").resolve()
    )
    assert config["paths"]["work_dir"] == str((project / "work").resolve())
    assert config["paths"]["output_dir"] == str((project / "results").resolve())
    assert config["paths"]["ffmpeg"] == "ffmpeg"
    assert config["paths"]["openface_feature_extraction"] == "FeatureExtraction"
