import json
import tempfile
import unittest
from pathlib import Path

from core.analysis_spec import AnalysisSpec
from core.run_manifest import (
    FreshnessStatus,
    RunManifest,
    SourceRecord,
    StageStatus,
)


class RunManifestTests(unittest.TestCase):
    def test_manifest_tracks_sources_and_stages(self):
        spec = AnalysisSpec.default_for_city("Matera", analysis_date="2026-10-06")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "requirements.txt").write_text("networkx\n", encoding="utf-8")
            manifest = RunManifest.create(spec, repo_root=root)
            manifest.register_source(
                "istat_census",
                SourceRecord(
                    source_name="ISTAT",
                    reference_date_or_period="2023-12-31",
                    freshness_status=FreshnessStatus.FRESH,
                    checksum_sha256="abc123",
                ),
            )
            manifest.start_stage("demand")
            manifest.complete_stage("demand", outputs=["origins.parquet"])
            self.assertEqual(
                manifest.stages["demand"].status, StageStatus.COMPLETED
            )
            self.assertEqual(manifest.source_checksums["istat_census"], "abc123")

            path = root / "manifest.json"
            manifest.write_json(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["analysis_spec_hash"], spec.spec_hash)
            self.assertEqual(payload["stages"]["demand"]["status"], "completed")

    def test_degraded_stage_is_not_failure(self):
        spec = AnalysisSpec.default_for_city("Parma", analysis_date="2026-10-06")
        with tempfile.TemporaryDirectory() as tmp:
            manifest = RunManifest.create(spec, repo_root=Path(tmp))
            manifest.complete_stage(
                "community_houses",
                degraded=True,
                warnings=["Operational registry is stale."],
            )
            self.assertEqual(
                manifest.stages["community_houses"].status, StageStatus.DEGRADED
            )


if __name__ == "__main__":
    unittest.main()
