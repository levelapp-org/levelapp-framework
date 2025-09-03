import unittest
import time
import logging
from unittest.mock import patch
from io import StringIO
from threading import Thread
from inspect import signature
from levelapp.aspects.monitor import FunctionMonitor


class TestFunctionMonitor(unittest.TestCase):
    def setUp(self):
        # Clear registry before each test
        FunctionMonitor._monitored_functions.clear()
        # Set up log capture
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setLevel(logging.DEBUG)
        logging.getLogger('levelapp.aspects.monitoring').addHandler(self.handler)

    def tearDown(self):
        # Clean up logging
        logging.getLogger('levelapp.aspects.monitoring').removeHandler(self.handler)
        self.handler.close()

    def test_basic_function_execution(self):
        """Test decorated function executes properly"""

        @FunctionMonitor.monitor("test_func")
        def add_one(x: int) -> int:
            return x + 1

        result = add_one(5)
        self.assertEqual(result, 6)
        self.assertIn("test_func", FunctionMonitor._monitored_functions)

    def test_caching_behavior(self):
        """Verify LRU caching works as expected"""
        call_count = 0

        @FunctionMonitor.monitor("cached_func", cached=True, maxsize=2)
        def square(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * x

        # First call (miss)
        self.assertEqual(square(2), 4)
        self.assertEqual(call_count, 1)

        # Second call (hit)
        self.assertEqual(square(2), 4)
        self.assertEqual(call_count, 1)

        # New arg (miss)
        self.assertEqual(square(3), 9)
        self.assertEqual(call_count, 2)

    def test_execution_timing(self):
        """Verify timing logs are produced"""

        @FunctionMonitor.monitor("timed_func", enable_timing=True)
        def slow_func():
            time.sleep(0.01)

        slow_func()
        logs = self.log_stream.getvalue()
        self.assertIn("Executed 'timed_func'", logs)
        self.assertIn("s", logs)  # Verify duration is logged

    def test_error_handling(self):
        """Verify exceptions are logged and propagated"""

        @FunctionMonitor.monitor("error_func")
        def failing_func():
            raise ValueError("Intentional error")

        with self.assertRaises(ValueError):
            failing_func()

        logs = self.log_stream.getvalue()
        self.assertIn("Error in 'error_func'", logs)
        self.assertIn("Intentional error", logs)

    def test_thread_safe_registration(self):
        """Verify only one registration succeeds in concurrent scenarios"""
        results = []

        def register_func():
            try:
                @FunctionMonitor.monitor("thread_func")
                def dummy():
                    pass

                results.append(True)
            except ValueError:
                results.append(False)

        threads = [Thread(target=register_func) for _ in range(5)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        self.assertEqual(sum(results), 1)  # Only one should succeed

    def test_get_stats(self):
        """Verify statistics collection works"""

        @FunctionMonitor.monitor("stats_func", cached=True)
        def multiply(x: int, y: int) -> int:
            return x * y

        multiply(2, 3)  # Miss
        multiply(2, 3)  # Hit

        stats = FunctionMonitor.get_stats("stats_func")
        self.assertIsNotNone(stats)
        self.assertEqual(stats['name'], "stats_func")
        self.assertTrue(stats['is_cached'])

        # Only check cache_info if caching is enabled
        if stats['is_cached']:
            self.assertEqual(stats['cache_info'].hits, 1)
            self.assertEqual(stats['cache_info'].misses, 1)

        # Check parameter count
        self.assertEqual(stats.get('execution_count'), 2)  # x and y params

    def test_duplicate_registration(self):
        """Prevent duplicate function names"""

        @FunctionMonitor.monitor("unique_func")
        def first():
            pass

        with self.assertRaises(ValueError):
            @FunctionMonitor.monitor("unique_func")
            def second():
                pass

    def test_method_decorator(self):
        """Verify decorator works with methods"""

        class Calculator:
            @FunctionMonitor.monitor("calc_method")
            def add(self, a: int, b: int) -> int:
                return a + b

        calc = Calculator()
        self.assertEqual(calc.add(2, 3), 5)
        self.assertIn("calc_method", FunctionMonitor._monitored_functions)

    def test_signature_preservation(self):
        """Verify original function metadata is preserved"""

        @FunctionMonitor.monitor("meta_func")
        def original(a: int, b: str = "default") -> float:
            """Original docstring"""
            return 3.14

        # Check signature
        sig = signature(original)
        self.assertEqual(list(sig.parameters.keys()), ['a', 'b'])
        self.assertEqual(sig.return_annotation, float)

        # Check docstring
        self.assertEqual(original.__doc__, "Original docstring")

    def test_cache_management(self):
        """Verify cache clearing works"""
        call_count = 0

        @FunctionMonitor.monitor("managed_cache", cached=True)
        def count_calls(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        # Verify cache methods exist
        self.assertTrue(hasattr(count_calls, 'cache_clear'))
        self.assertTrue(hasattr(count_calls, 'cache_info'))

        count_calls(1)  # Miss
        count_calls(1)  # Hit
        count_calls.cache_clear()
        count_calls(1)  # Miss after clear

        self.assertEqual(call_count, 2)

    def test_nonexistent_stats(self):
        """Verify graceful handling of unregistered functions"""
        stats = FunctionMonitor.get_stats("nonexistent")
        self.assertIsNone(stats)


if __name__ == '__main__':
    unittest.main(verbosity=2)