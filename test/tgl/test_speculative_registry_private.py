import inspect
import unittest
from unittest import mock

from sglang.private.speculative.phoenix_worker import PhoenixWorker
from sglang.private.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_info import list_registered_workers


class PrivateSpeculativeRegistryTests(unittest.TestCase):
    def test_phoenix_registration(self):
        phoenix = SpeculativeAlgorithm.from_string("PHOENIX")

        self.assertTrue(phoenix.is_phoenix())
        self.assertTrue(phoenix.is_eagle())

        registered_workers = list_registered_workers()
        self.assertIn("PHOENIX", registered_workers)
        worker_entry = registered_workers["PHOENIX"]
        self.assertTrue(callable(worker_entry))

        if inspect.isclass(worker_entry):
            self.assertIs(worker_entry, PhoenixWorker)
            print("worker_entry is PhoenixWorker class")
        else:
            with mock.patch(
                "sglang.private.speculative.phoenix_worker.PhoenixWorker"
            ) as mock_worker:
                print("worker_entry is a factory function")
                sentinel_instance = object()
                mock_worker.return_value = sentinel_instance

                instance = worker_entry()

                self.assertIs(instance, sentinel_instance)
                mock_worker.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
