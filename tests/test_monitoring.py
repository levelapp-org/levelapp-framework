import unittest
import time
import logging
from unittest.mock import patch
from levelapp.utils.monitoring import FunctionMonitor


class TestFunctionMonitor(unittest.TestCase):
    def setUp(self):
        # Clear registry and enable logging capture
        FunctionMonitor._monitored_functions.clear()
        self.logger = logging.getLogger('levelapp.utils.monitoring')
        self.log_capture = self._setup_log_capture()

    def _setup_log_capture(self):
        """Helper to capture log output"""
        from io import StringIO
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)
        return stream

    def test_timing_logging(self):
        """Test execution time logging"""

        @FunctionMonitor.register("timed_func", enable_timing=True)
        def func():
            time.sleep(0.01)
            return "done"

        func()
        logs = self.log_capture.getvalue()
        self.assertIn("Executed 'timed_func'", logs)

    def test_error_handling(self):
        """Test error logging"""

        @FunctionMonitor.register("error_func")
        def func():
            raise ValueError("Test error")

        with self.assertRaises(ValueError):
            func()

        logs = self.log_capture.getvalue()
        self.assertIn("Error in 'error_func'", logs)

    def tearDown(self):
        logging.disable(logging.NOTSET)  # Re-enable logging

    def test_basic_functionality(self):
        """Test basic decorated function execution"""
        @FunctionMonitor.register("test_func")
        def func(x):
            return x * 2

        result = func(5)
        self.assertEqual(result, 10)

    def test_caching_behavior(self):
        """Test LRU caching functionality"""
        call_count = 0

        @FunctionMonitor.register("cached_func", cached=True, maxsize=2)
        def func(x):
            nonlocal call_count
            call_count += 1
            return x * 3

        # First call (cache miss)
        self.assertEqual(func(2), 6)
        self.assertEqual(call_count, 1)

        # Second call with same args (cache hit)
        self.assertEqual(func(2), 6)
        self.assertEqual(call_count, 1)

        # Different args (cache miss)
        self.assertEqual(func(3), 9)
        self.assertEqual(call_count, 2)

    def test_timing_logging(self):
        """Test execution time logging"""
        with patch.object(FunctionMonitor, '_monitored_functions', {}):
            @FunctionMonitor.register("timed_func", enable_timing=True)
            def func():
                time.sleep(0.1)
                return "done"

            with self.assertLogs(logger='levelapp.utils.monitoring', level='INFO') as cm:
                func()
                self.assertTrue(any("Executed 'timed_func'" in log for log in cm.output))

    def test_error_handling(self):
        """Test error logging"""
        @FunctionMonitor.register("error_func")
        def func():
            raise ValueError("Test error")

        with self.assertLogs(logger='levelapp.utils.monitoring', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                func()
            self.assertTrue(any("Error in 'error_func'" in log for log in cm.output))

    def test_thread_safety(self):
        """Test concurrent registration safety"""
        from threading import Thread

        results = []

        def register_func():
            try:
                @FunctionMonitor.register("thread_func")
                def func():
                    pass
                results.append(True)
            except Exception:
                results.append(False)

        threads = [Thread(target=register_func) for _ in range(5)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Only one registration should succeed
        self.assertEqual(sum(results), 1)

    def test_get_stats(self):
        """Test statistics retrieval"""
        @FunctionMonitor.register("stats_func", cached=True)
        def func(x):
            return x + 1

        func(1)  # Cache miss
        func(1)  # Cache hit

        stats = FunctionMonitor.get_stats("stats_func")
        self.assertIsNotNone(stats)
        self.assertEqual(stats['cache_info'].hits, 1)
        self.assertEqual(stats['cache_info'].misses, 1)

    def test_duplicate_registration(self):
        """Test duplicate function name prevention"""
        @FunctionMonitor.register("dupe_func")
        def func1():
            pass

        with self.assertRaises(ValueError):
            @FunctionMonitor.register("dupe_func")
            def func2():
                pass

    def test_method_decorator(self):
        """Test decorator works with methods"""
        class TestClass:
            @FunctionMonitor.register("test_method")
            def method(self, x):
                return x * 2

        obj = TestClass()
        self.assertEqual(obj.method(3), 6)

    def test_signature_preservation(self):
        """Test original function signature is preserved"""
        from inspect import signature

        @FunctionMonitor.register("sig_func")
        def func(a: int, b: str = "test") -> float:
            """Test function"""
            return 3.14

        sig = signature(func)
        self.assertEqual(list(sig.parameters.keys()), ['a', 'b'])
        self.assertEqual(sig.return_annotation, float)
        self.assertEqual(func.__doc__, "Test function")

    def test_cache_clear(self):
        """Test cache clearing functionality"""
        call_count = 0

        @FunctionMonitor.register("clear_cache_func", cached=True)
        def func(x):
            nonlocal call_count
            call_count += 1
            return x

        func(1)  # Miss
        func(1)  # Hit
        func.cache_clear()  # Clear cache
        func(1)  # Miss again after clear

        self.assertEqual(call_count, 2)

if __name__ == '__main__':
    unittest.main(verbosity=2)