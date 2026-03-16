from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from traffic_sign_attacks.pipeline import run_pipeline


def _make_synthetic_sign(path: Path) -> None:
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, 56, 56), fill=(240, 240, 240), outline=(200, 0, 0), width=6)
    draw.text((24, 24), "30", fill="black")
    image.save(path)


class PipelineTest(unittest.TestCase):
    def test_pipeline_generates_shadow_and_occlusion_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            output_root = root / "generated"
            image_dir = dataset_root / "Test"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "00000.png"
            _make_synthetic_sign(image_path)

            with (dataset_root / "Test.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "Path",
                        "Width",
                        "Height",
                        "Roi.X1",
                        "Roi.Y1",
                        "Roi.X2",
                        "Roi.Y2",
                        "ClassId",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Path": "Test/00000.png",
                        "Width": 64,
                        "Height": 64,
                        "Roi.X1": 8,
                        "Roi.Y1": 8,
                        "Roi.X2": 56,
                        "Roi.Y2": 56,
                        "ClassId": 1,
                    }
                )

            config = {
                "dataset_root": str(dataset_root),
                "output_root": str(output_root),
                "split": "test",
                "seed": 3,
                "output_extension": ".png",
                "physical_transform": {"enabled": False},
                "attacks": {
                    "shadow": {
                        "enabled": True,
                        "variants_per_image": 1,
                        "coefficient_range": [0.40, 0.45],
                        "coverage_ratio": [0.10, 0.70],
                        "edge_blur_radius": [0.0, 0.0],
                    },
                    "occlusion": {
                        "enabled": True,
                        "variants_per_image": 1,
                        "styles": ["edge_patch"],
                        "opacity_range": [1.0, 1.0],
                        "edge_band_width_ratio": [0.05, 0.05],
                        "palette": ["#1A1A1A"],
                    },
                },
            }

            summary = run_pipeline(config)
            self.assertEqual(summary["num_input_samples"], 1)
            self.assertEqual(summary["num_generated_samples"], 2)

            manifest_path = output_root / "manifest.csv"
            self.assertTrue(manifest_path.exists())

            shadow_output = output_root / "test" / "shadow" / "01" / "00000_shadow_00.png"
            occlusion_output = output_root / "test" / "occlusion" / "01" / "00000_occlusion_00.png"
            self.assertTrue(shadow_output.exists())
            self.assertTrue(occlusion_output.exists())

            original = Image.open(image_path).convert("RGB")
            shadow = Image.open(shadow_output).convert("RGB")
            occlusion = Image.open(occlusion_output).convert("RGB")

            self.assertNotEqual(list(original.getdata()), list(shadow.getdata()))
            self.assertNotEqual(list(original.getdata()), list(occlusion.getdata()))


if __name__ == "__main__":
    unittest.main()
