import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pipeline


class PipelineProvenanceTests(unittest.TestCase):
    def _legacy_config(self):
        return SimpleNamespace(
            walking_speed_m_s=1.4,
            accessibility_thresholds_min=(10, 15, 20),
        )

    def _ctx(self):
        return SimpleNamespace(
            name="Parma",
            code="034027",
        )

    def test_build_legacy_spec_does_not_change_legacy_parameters(self):
        spec = pipeline.build_legacy_analysis_spec(
            ctx=self._ctx(),
            config=self._legacy_config(),
            analysis_date="2026-10-08",
        )
        self.assertEqual(spec.walking_speed_m_s, 1.4)
        self.assertEqual(spec.execution_profile, "legacy_v1_regression")
        self.assertEqual(spec.analysis_date, "2026-10-08")

    def test_initialize_run_metadata_writes_spec_and_manifest(self):
        spec = pipeline.build_legacy_analysis_spec(
            ctx=self._ctx(),
            config=self._legacy_config(),
            analysis_date="2026-10-08",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "requirements.txt").write_text("networkx\n", encoding="utf-8")
            with patch.object(pipeline, "ROOT", root):
                manifest, run_dir, spec_path, manifest_path = (
                    pipeline.initialize_run_metadata(spec)
                )

            self.assertTrue(run_dir.exists())
            self.assertTrue(spec_path.exists())
            self.assertTrue(manifest_path.exists())

            spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))
            manifest_payload = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest_payload["analysis_spec_hash"], manifest.analysis_spec_hash
            )
            self.assertEqual(
                manifest_payload["analysis_spec_hash"], spec.spec_hash
            )
            self.assertEqual(
                spec_payload["execution_profile"], "legacy_v1_regression"
            )

    def test_run_stage_records_completed_stage(self):
        spec = pipeline.build_legacy_analysis_spec(
            ctx=self._ctx(),
            config=self._legacy_config(),
            analysis_date="2026-10-08",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(pipeline, "ROOT", root):
                manifest, _, _, manifest_path = pipeline.initialize_run_metadata(spec)
                with patch.object(pipeline, "run_command") as mocked:
                    pipeline.run_stage(
                        stage="istat",
                        stage_commands=[["python", "fake.py"]],
                        manifest=manifest,
                        manifest_path=manifest_path,
                    )

            mocked.assert_called_once()
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["stages"]["istat"]["status"], "completed")
            self.assertEqual(
                payload["stages"]["istat"]["metrics"]["command_count"], 1
            )


if __name__ == "__main__":
    unittest.main()
