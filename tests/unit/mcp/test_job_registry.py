import json
import tempfile
import unittest
from pathlib import Path

from oumi.mcp.job_service import JobRecord, JobRegistry, JobRuntime


class TestJobRecord(unittest.TestCase):
    def test_all_fields_are_strings(self):
        r = JobRecord(
            job_id="train_001",
            command="train",
            config_path="/tmp/train.yaml",
            cloud="gcp",
            cluster_name="cluster-a",
            model_name="meta-llama/Llama-3.1-8B",
            submit_time="2099-01-01T12:00:00Z",
        )
        for field_name in [
            "job_id",
            "command",
            "config_path",
            "cloud",
            "cluster_name",
            "model_name",
            "submit_time",
        ]:
            self.assertIsInstance(getattr(r, field_name), str)


class TestJobRuntime(unittest.TestCase):
    def test_defaults_are_none(self):
        rt = JobRuntime()
        self.assertIsNone(rt.process)
        self.assertIsNone(rt.cluster_obj)
        self.assertIsNone(rt.runner_task)
        self.assertIsNone(rt.oumi_status)


class TestJobRegistry(unittest.TestCase):
    def test_add_and_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            reg = JobRegistry(path)
            r = JobRecord(
                job_id="j1",
                command="train",
                config_path="/tmp/t.yaml",
                cloud="local",
                cluster_name="",
                model_name="test",
                submit_time="2099-01-01T00:00:00Z",
            )
            reg.add(r)
            self.assertEqual(reg.get("j1").job_id, "j1")

    def test_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            reg = JobRegistry(path)
            r = JobRecord(
                job_id="j1",
                command="train",
                config_path="/tmp/t.yaml",
                cloud="gcp",
                cluster_name="c1",
                model_name="test",
                submit_time="2099-01-01T00:00:00Z",
            )
            reg.add(r)
            # Load a new registry from the same file
            reg2 = JobRegistry(path)
            loaded = reg2.get("j1")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.cloud, "gcp")
            self.assertEqual(loaded.job_id, "j1")

    def test_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            reg = JobRegistry(path)
            r = JobRecord(
                job_id="j1",
                command="train",
                config_path="/tmp/t.yaml",
                cloud="local",
                cluster_name="",
                model_name="test",
                submit_time="2099-01-01T00:00:00Z",
            )
            reg.add(r)
            reg.update("j1", cluster_name="updated-cluster")
            self.assertEqual(reg.get("j1").cluster_name, "updated-cluster")

    def test_remove(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            reg = JobRegistry(path)
            r = JobRecord(
                job_id="j1",
                command="train",
                config_path="/tmp/t.yaml",
                cloud="local",
                cluster_name="",
                model_name="test",
                submit_time="2099-01-01T00:00:00Z",
            )
            reg.add(r)
            reg.remove("j1")
            self.assertIsNone(reg.get("j1"))
            # Verify persisted
            reg2 = JobRegistry(path)
            self.assertIsNone(reg2.get("j1"))

    def test_find_by_cloud(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            reg = JobRegistry(path)
            r = JobRecord(
                job_id="sky-99",
                command="train",
                config_path="/tmp/t.yaml",
                cloud="gcp",
                cluster_name="c1",
                model_name="test",
                submit_time="2099-01-01T00:00:00Z",
            )
            reg.add(r)
            found = reg.find_by_cloud("gcp", "sky-99")
            self.assertEqual(found.job_id, "sky-99")
            self.assertIsNone(reg.find_by_cloud("aws", "sky-99"))

    def test_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            reg = JobRegistry(path)
            for i in range(3):
                reg.add(
                    JobRecord(
                        job_id=f"j{i}",
                        command="train",
                        config_path="/tmp/t.yaml",
                        cloud="local",
                        cluster_name="",
                        model_name="test",
                        submit_time="2099-01-01T00:00:00Z",
                    )
                )
            self.assertEqual(len(reg.all()), 3)

    def test_load_corrupt_file_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            path.write_text("not valid json{{{", encoding="utf-8")
            reg = JobRegistry(path)
            self.assertEqual(len(reg.all()), 0)

    def test_load_missing_file_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            reg = JobRegistry(path)
            self.assertEqual(len(reg.all()), 0)

    def test_prune_old(self):
        """Records older than 7 days are pruned on load."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            reg = JobRegistry(path)
            reg.add(
                JobRecord(
                    job_id="old",
                    command="train",
                    config_path="/tmp/t.yaml",
                    cloud="gcp",
                    cluster_name="c",
                    model_name="m",
                    submit_time="2020-01-01T00:00:00+00:00",
                )
            )
            reg.add(
                JobRecord(
                    job_id="new",
                    command="train",
                    config_path="/tmp/t.yaml",
                    cloud="gcp",
                    cluster_name="c",
                    model_name="m",
                    submit_time="2099-01-01T00:00:00+00:00",
                )
            )
            reg2 = JobRegistry(path)
            self.assertIsNone(reg2.get("old"))
            self.assertIsNotNone(reg2.get("new"))

    def test_update_persists(self):
        """update() writes the change through to disk."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            reg = JobRegistry(path)
            reg.add(
                JobRecord(
                    job_id="j1",
                    command="train",
                    config_path="/tmp/t.yaml",
                    cloud="gcp",
                    cluster_name="",
                    model_name="m",
                    submit_time="2099-01-01T00:00:00Z",
                )
            )
            reg.update("j1", cluster_name="cl-1")
            reg2 = JobRegistry(path)
            loaded = reg2.get("j1")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.cluster_name, "cl-1")

    def test_update_missing_noop(self):
        """update() on an unknown job_id is a no-op (no crash, no record created)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            reg = JobRegistry(path)
            reg.update("nonexistent", cluster_name="cl-99")  # should not raise
            self.assertIsNone(reg.get("nonexistent"))

    def test_legacy_records_with_status(self):
        """Legacy JSON records containing a 'status' field load without error."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            legacy = [
                {
                    "job_id": "legacy-1",
                    "command": "train",
                    "config_path": "/tmp/t.yaml",
                    "cloud": "gcp",
                    "cluster_name": "cl",
                    "model_name": "m",
                    "submit_time": "2099-01-01T00:00:00+00:00",
                    "status": "RUNNING",  # legacy field — should be silently dropped
                }
            ]
            path.write_text(json.dumps(legacy), encoding="utf-8")
            reg = JobRegistry(path)
            loaded = reg.get("legacy-1")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.job_id, "legacy-1")
            self.assertFalse(hasattr(loaded, "status"))


if __name__ == "__main__":
    unittest.main()
