import unittest
import requests

from unittest.mock import patch, MagicMock
from levelapp.clients.anthropic import AnthropicClient


class TestClaudeClient(unittest.TestCase):

    @patch('requests.post')
    def test_call_success(self, mock_post):
        # Setup
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "success"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = AnthropicClient(api_key="fake_api_key")

        # Exercise
        response = client.call("Hello, world!")

        # Verify
        self.assertEqual(response, {"response": "success"})
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_call_http_error(self, mock_post):
        # Setup
        mock_post.side_effect = requests.exceptions.HTTPError("HTTP Error")

        client = AnthropicClient(api_key="fake_api_key")

        # Exercise and Verify
        with self.assertRaises(requests.exceptions.HTTPError):
            client.call("Hello, world!")

    @patch('requests.post')
    def test_call_connection_error(self, mock_post):
        # Setup
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection Error")

        client = AnthropicClient(api_key="fake_api_key")

        # Exercise and Verify
        with self.assertRaises(requests.exceptions.ConnectionError):
            client.call("Hello, world!")

    def test_init_missing_api_key(self):
        # Setup and Exercise
        with self.assertRaises(ValueError):
            AnthropicClient()


if __name__ == '__main__':
    unittest.main()
