# LevelApp: AI/LLM Evaluation Framework for Regression Testing

[![PyPI version](https://badge.fury.io/py/levelapp.svg)](https://badge.fury.io/py/levelapp)  
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)  
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

## Overview

LevelApp is an evaluation framework designed for regression testing (black-box) of already built LLM-based systems in production or testing phases. It focuses on assessing the performance and reliability of AI/LLM applications through simulation and comparison modules. Powered by Norma.

Key benefits:
- Configuration-driven: Minimal coding required; define evaluations via YAML files.
- Supports LLM-as-a-judge for qualitative assessments and quantitative metrics for metadata evaluation.
- Modular architecture for easy extension to new workflows, evaluators, and repositories.

## Features

- **Simulator Module**: Evaluates dialogue systems by simulating conversations using predefined scripts. It uses an LLM as a judge to score replies against references and supports metrics (e.g., Exact, Embedded, Token-based, Fuzzy) for comparing extracted metadata to ground truth.
- **Comparator Module**: Evaluates metadata extraction from JSON outputs (e.g., from legal/financial document processing with LLMs) by comparing against reference/ground-truth data.
- **Configuration-Based Workflow**: Users provide YAML configs for endpoints, parameters, data sources, and metrics, reducing the need for custom code.
- **Supported Workflows**: SIMULATOR, COMPARATOR, ASSESSOR (coming soon!).
- **Repositories**: FIRESTORE, FILESYSTEM, MONGODB.
- **Evaluators**: JUDGE, REFERENCE, RAG.
- **Metrics**: Exact, Levenshtein, and more (see docs for full list).
- **Data Sources**: Local or remote JSON for conversation scripts.

## Installation

Install LevelApp via pip:

```bash
pip install levelapp
```

### Prerequisites
- Python 3.12 or higher.
- API keys for LLM providers (e.g., OpenAI, Anthropic) if using external clients—store in a `.env` file.
- Optional: Google Cloud credentials for Firestore repository.
- Dependencies are automatically installed, including `openai`, `pydantic`, `numpy`, etc. (see `pyproject.toml` for full list).

## Configuration

LevelApp uses a YAML configuration file to define the evaluation setup. Create a `workflow_config.yaml` with the following structure:

```yaml
process:
  project_name: "test-project"
  workflow_type: SIMULATOR # Pick one of the following workflows: SIMULATOR, COMPARATOR, ASSESSOR.
  evaluation_params:
    attempts: 1  # Add the number of simulation attempts.
    batch_size: 5

evaluation:
  evaluators: # Select from the following: JUDGE, REFERENCE, RAG.
    - JUDGE
    - REFERENCE
  providers:
    - openai
    - ionos
  metrics_map:
    field_1: EXACT
    field_2 : LEVENSHTEIN

reference_data:
  path: "../data/conversation_example_1.json"
  data:

endpoint:
  name: conversational-agent
  base_url: http://127.0.0.1:8000
  path: /v1/chat
  method: POST
  timeout: 60
  retry_count: 3
  retry_backoff: 0.5
  headers:
    - name: model_id
      value: meta-llama/Meta-Llama-3-8B-Instruct
      secure: false
    - name: x-api-key
      value: API_KEY  # Load from .env file using python-dotenv.
      secure: true
    - name: Content-Type
      value: application/json
      secure: false
  request_schema:
    # Static field to be included in every request.
    - field_path: message.source
      value: system
      value_type: static
      required: true
      
    # Dynamic field to be populated from runtime context.
    - field_path: message.text
      value: message_text  # the key from the runtime context.
      value_type: dynamic
      required: true
      
    # Env-based field (from OS environment variables).
    - field_path: metadata.env
      value: ENV_VAR_NAME
      value_type: env
      required: false
      
  response_mapping:
    # Map the response fields that will be extracted.
    - field_path: reply.text
      extract_as: agent_reply  # The simulator requires this key: 'agent_reply'.
    - field_path: reply.metadata
      extract_as: generated_metadata  # The simulator requires this key: 'generated_metadata'.
    - field_path: reply.guardrail_flag
      extract_as: guardrail_flag  # The simulator requires this key: 'guardrail_flag'.

repository:
  type: FIRESTORE # Pick one of the following: FIRESTORE, FILESYSTEM
  project_id: "(default)"
  database_name: ""
```

- **Endpoint Configuration**: Define how to interact with your LLM-based system (base URL, headers, request/response payload schema).
- **Placeholders**: For dynamic request schema fields, use the values ('value') to dynamically populate these fields during runtime (e.g., `context = {'message_text': "Hello, world!"}`).
- **Secrets**: Store API keys in `.env` and load via `python-dotenv` (e.g., `API_KEY=your_key_here`).

For conversation scripts (used in Simulator), provide a JSON file with this schema:

```json
{
  "scripts": [
    {
      "interactions": [
        {
          "user_message": "Hello, I would like to book an appointment with a doctor.",
          "reference_reply": "Sure, I can help with that. Could you please specify the type of doctor you need to see?",
          "interaction_type": "initial",
          "reference_metadata": {},
          "guardrail_flag": false
        },
        {
          "user_message": "I need to see a cardiologist.",
          "reference_reply": "When would you like to schedule your appointment?",
          "interaction_type": "intermediate",
          "reference_metadata": {},
          "guardrail_flag": false
        },
        {
          "user_message": "I would like to book it for next Monday morning.",
          "reference_reply": "We have an available slot at 10 AM next Monday. Does that work for you?",
          "interaction_type": "intermediate",
          "reference_metadata": {
            "appointment_type": "Cardiology",
            "date": "next Monday",
            "time": "10 AM"
          },
          "guardrail_flag": false
        },
        {
          "id": "f4f2dd35-71d7-4b75-ba2b-93a4f546004a",
          "user_message": "Yes, please book it for 10 AM then.",
          "reference_reply": "Your appointment with the cardiologist is booked for 10 AM next Monday. Is there anything else I can help you with?",
          "interaction_type": "final",
          "reference_metadata": {},
          "guardrail_flag": false
        }
      ],
      "description": "A conversation about booking a doctor appointment.",
      "details": {
        "context": "Booking a doctor appointment"
      }
    }
  ]
}
```
- **Fields**: Include user messages, reference/references replies, metadata for comparison, guardrail flags, and request payloads.

In the `.env` you need to add the LLM providers credentials that will be used for the evaluation process. 
```
OPENAI_API_KEY=
IONOS_API_KEY=
ANTHROPIC_API_KEY=
MISTRAL_API_KEY=

# For IONOS, you must include the base URL and the model ID.
IONOS_BASE_URL="https://inference.de-txl.ionos.com"
IONOS_MODEL_ID="0b6c4a15-bb8d-4092-82b0-f357b77c59fd"

WORKFLOW_CONFIG_PATH="../../src/data/workflow_config_1.yaml"
```

## Usage Example

To run an evaluation:

1. Prepare your YAML config and JSON data files.
2. Use the following Python script:

```python
if __name__ == "__main__":
    from levelapp.workflow import WorkflowConfig
    from levelapp.core.session import EvaluationSession

    # Load configuration from YAML
    config = WorkflowConfig.load(path="../data/workflow_config.yaml")

    # Run evaluation session (You can enable/disable the monitoring aspect)
    with EvaluationSession(session_name="test-session-1", workflow_config=config, enable_monitoring=False) as session:
        session.run()
        results = session.workflow.collect_results()
        print("Results:", results)

    stats = session.get_stats()
    print(f"session stats:\n{stats}")
```

Alternatively, if you want to pass the configuration and reference data from in-memory variables, 
you can manually load the data like the following:
```python
if __name__ == "__main__":
    from levelapp.workflow import WorkflowConfig
    from levelapp.core.session import EvaluationSession

    
    config_dict = {
        "process": {
            "project_name": "test-project",
            "workflow_type": "SIMULATOR",  # Pick one of the following workflows: SIMULATOR, COMPARATOR, ASSESSOR.
            "evaluation_params": {
                "attempts": 1,  # Add the number of simulation attempts.
            }
        },
        "evaluation": {
            "evaluators": ["JUDGE", "REFERENCE"],  # Select from the following: JUDGE, REFERENCE, RAG.
            "providers": ["openai", "ionos"],
            "metrics_map": {
                "field_1": "EXACT",
                "field_2": "LEVENSHTEIN"
            }
        },
        "reference_data": {
            "path": "../data/conversation_example_1.json",
            "data": None
        },
        "endpoint": {
            "name": "conversational-agent",
            "base_url": "http://127.0.0.1:8000",
            "path": "/v1/chat",
            "method": "POST",
            "timeout": 60,
            "retry_count": 3,
            "retry_backoff": 0.5,
            "headers": [
                {
                    "name": "model_id",
                    "value": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "secure": False
                },
                {
                    "name": "x-api-key",
                    "value": "API_KEY",  # Load from .env file using python-dotenv.
                    "secure": True 
                },
                {
                    "name": "Content-Type",
                    "value": "application/json",
                    "secure": False
                }
            ],
            "request_schema": [
                {
                    "field_path": "message.source",
                    "value": "system",
                    "value_type": "static",
                    "required": True
                },
                {
                    "field_path": "message.text",
                    "value": "message_text",  # the key from the runtime context.
                    "value_type": "dynamic",
                    "required": True
                },
                {
                    "field_path": "metadata.env",
                    "value": "ENV_VAR_NAME",
                    "value_type": "env",
                    "required": False
                }
            ],
            "response_mapping": [
                {
                    "field_path": "reply.text",
                    "extract_as": "agent_reply"  # Remember that the simulator requires this key: 'agent_reply'.
                },
                {
                    "field_path": "reply.metadata",
                    "extract_as": "agent_reply"  # Remember that the simulator requires this key: 'agent_reply'.
                },
                {
                    "field_path": "reply.guardrail_flag",
                    "extract_as": "metadata"  # Remember that the simulator requires this key: 'agent_reply'.
                }
            ]
        },
        "repository": {
            "type": "FIRESTORE",  # Pick one of the following: FIRESTORE, FILESYSTEM
            "project_id": "(default)",
            "database_name": ""
        }
    }

    content = {
        "scripts": [
            {
                "interactions": [
                    {
                        "user_message": "Hello!",
                        "reference_reply": "Hello, how can I help you!"
                    },
                    {
                        "user_message": "I need an apartment",
                        "reference_reply": "sorry, but I can only assist you with booking medical appointments."
                    },
                ]
            },
        ]
    }

    # Load configuration from a dict variable
    config = WorkflowConfig.from_dict(content=config_dict)

    # Load reference data from dict variable
    config.set_reference_data(content=content)

    evaluation_session = EvaluationSession(
        session_name="test-session", 
        workflow_config=config, 
        enable_monitoring=True  # To disable the monitoring aspect, set this to False.
    )

    with evaluation_session as session:
        # Optional: Run connectivity test before the full evaluation
        test_results = session.run_connectivity_test(
            context={"user_message": "I want to book an appointment with a dentist."}
        )
        print(f"Connectivity Test Results:\n{test_results}\n---")
        session.run()
        results = session.workflow.collect_results()
        print("Results:", results)

    stats = session.get_stats()
    print(f"session stats:\n{stats}")

```


- This loads the config, runs the specified workflow (e.g., Simulator), collects results, and prints stats.

For more examples, see the `examples/` directory.

## Documentation

Detailed docs are in the `docs/` directory, including API references and advanced configuration.

## Contributing

Contributions are welcome! Please follow these steps:
- Fork the repository on GitHub.
- Create a feature branch (`git checkout -b feature/new-feature`).
- Commit changes (`git commit -am 'Add new feature'`).
- Push to the branch (`git push origin feature/new-feature`).
- Open a pull request.

Report issues via GitHub Issues. Follow the code of conduct (if applicable).

## Acknowledgments

- Powered by Norma.
- Thanks to contributors and open-source libraries like Pydantic, NumPy, and OpenAI SDK.

## License

This project is licensed under the MIT License - see the [LICENCE](LICENCE) file for details.

---