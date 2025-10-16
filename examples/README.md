# Quickstart Guide: Using LevelApp's Conversation Simulator for Developers

---
#### Welcome to LevelApp Quickstart Guide!
This guide provides a step-by-step walkthrough for developers to set up and use the Simulator Module in LevelApp.
<br>
<br>
The **Simulator** focuses on black-box testing by simulating dialogues using predefined scripts, evaluating responses against references, and computing metrics on extracted metadata. 
It leverages LLM-as-a-judge for qualitative scoring and supports quantitative metrics like exact matches or fuzzy comparisons.
<br>
<figure>
    <img 
    src="../docs/media/simulator-module-diagram.PNG"
    alt="Sequence Diagram">
    <figcaption>Fig.1 - Simulator Module Diagram</figcaption>
</figure>
<br>
We'll emphasize technical details, including configuration schemas, placeholders, evaluators, metrics, and code execution flow. This assumes you're familiar with Python, YAML/JSON, and REST APIs for LLM endpoints. By the end, you'll have a runnable example for evaluating a chatbot's conversation flow.

---
## Introduction
First, let's have a quick introduction on what LevelApp is and what it provides as a framework.

The idea behind LevelApp is to build a framework that assists developers to perform regression tests on their LLM-powered systems ensuring that recent changes to code have not negatively impacted existing functionality or introduced new defects. <br>
The evaluation of dialogue systems is very cost/time intensive and problematic since assessing the quality of a dialogue requires multiple iteration where a human conducts a message/reply evaluation for each interaction (quite tedious and boring task, if you ask me!).

Automating the evaluation and introducing an LLM-as-a-judge as an approach to evaluate the correctness of responses can
ease the process and render it more efficient.
---
## Walkthrough
### Step1: Installation and Prerequisites
Install LevelApp using pip. This pulls in dependencies like `pydantic`, `numpy`, `python-dotenv`, 
and others for handling LLM clients, data validation, and metrics computation.

```bash
  pip install levelapp
```

#### Technical Prerequisites:

* **Python Version**: 3.12+. LevelApp uses modern features like type hints and async support (via `asyncio` for potential batch processing).
* **LLM Provider Credentials**: You'll need API keys for at least one supported provider (e.g., OpenAI, Anthropic, IONOS, Mistral). These are loaded via `python-dotenv` from a `.env` file. Without them, evaluators like JUDGE won't function.
* **No Internet for Dependencies**: All core deps are installed automatically; no manual `pip install` needed beyond the initial command.
* **Environment Setup**: Create a `.env` file in your project root. Example structure (replace with your actual keys):

```
IONOS_API_KEY=your-ionos-key
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
MISTRAL_API_KEY=your-mistral-key

IONOS_BASE_URL=https://inference.de-txl.ionos.com
IONOS_MODEL_ID=0b6c4a15-bb8d-4092-82b0-f357b77c59fd

# Optional: Path to workflow config if not loading programmatically
WORKFLOW_CONFIG_PATH=../data/workflow_config.yaml
```
**Note**: For IONOS, the base_url and model_id are mandatory in .env as they aren't always configurable via YAML alone.
LevelApp uses these to construct API requests.

### Step2: Understanding the Simulator Workflow
The Simulator Module simulates conversations by:
1. Sending user messages (from a JSON script) to your LLM-based system's endpoint.
2. Capturing generated responses and metadata.
3. Evaluating them using selected evaluators (e.g., JUDGE for LLM-scored quality, REFERENCE for direct comparison).
4. Computing metrics on metadata (e.g., EXACT for string matching, LEVENSHTEIN for edit distance).

<figure>
    <img 
    src="../docs/media/simulator-sequence-diagram.png"
    alt="Sequence Diagram">
    <figcaption>Fig.2 - Conversation Simulator Sequence Diagram</figcaption>
</figure>

#### Key Technical Concepts:

Workflow Type: Set to `SIMULATOR` in YAML. This triggers dialogue simulation logic in `levelapp.workflow`.

* **Evaluators**:
    * `JUDGE`: Uses an LLM (from providers like OpenAI) to score generated replies against references (e.g., on relevance, fluency). Configurable via providers list.
    * `REFERENCE`: Direct comparison without LLM, using metrics for metadata (Used for comparing extracted metadata).
    * `RAG`: Retrieval-Augmented Generation evaluator (for knowledge-grounded responses; requires additional setup).
<br>
<br>
* **Metrics Map**: A dict mapping metadata fields to comparison methods (e.g., `EXACT` for exact string match, `LEVENSHTEIN` for fuzzy matching with distance thresholds). 
Full list in docs: includes Token-based, Embedded (vector similarity), Fuzzy.
<br>
<br>
* **Attempts and Batching**: `evaluation_params` attempts runs simulations multiple times for averaging scores (useful for non-deterministic LLMs). batch_size controls concurrent requests to avoid rate limits.
<br>
<br>
* **Placeholders in Payloads**: 
    * `default_request_payload_template`: For this section, you need to change **field** (e.g,. change the field name `prompt` to `message`) names and not the **placeholder** values. The placeholders are used by the simulator to populate the request body.
    * `default_response_payload_template`: For this section, you need to change the placeholders values and not the fields, contrary to the request section. The simulator will use the provided placeholder values to extract and map the reply and metadata from the response body.

### Step 3: Creating the YAML Configuration File
Create `workflow_config.yaml` to define the workflow. This is parsed into a `WorkflowConfig` Pydantic model for validation.

Example `workflow_config.yaml` for Simulator:
```YAML
# PROCESS SECTION:
process:
  project_name: "chatbot-evaluation"
  workflow_type: SIMULATOR  # Must be SIMULATOR for conversation testing
  evaluation_params:
    attempts: 3  # Run each interaction 3 times, average results
    batch_size: 10  # Process 10 interactions concurrently

# EVALUATION SECTION:
evaluation:
  evaluators:  # Array of evaluators to apply
    - JUDGE
    - REFERENCE  # REFERENCE evaluator can be used if your dialogue system returns additional metadata.
  providers:  # LLM providers for JUDGE (At least one must be provided for the JUDGE evaluator)
    - openai
    - ionos
  metrics_map:  # Map metadata fields to metrics
    appointment_type: EXACT  # Exact match for strings
    date: LEVENSHTEIN  # Fuzzy match for dates (e.g., tolerates formatting differences)
    time: TOKEN_BASED  # Token-level overlap

# REFERENCE DATA SECTION:
reference_data:
  path: "conversation_script.json"  # Path to JSON script
  data: {}  # Inline data if not using path (dict of scripts)

# ENDPOINT CONFIGURATION SECTION:
endpoint:
  base_url: "http://127.0.0.1:8000"  # Your chatbot's API base URL
  url_path: "chat"  # Endpoint path (full URL = base_url + url_path)
  api_key: ""  # Optional; overrides .env if set
  bearer_token: ""  # For auth
  model_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Model for your endpoint (if applicable).
  default_request_payload_template:  # Template for POST body
    message: "${user_message}"  # Adapt to your API (e.g., 'prompt' for some)
    payload: "${request_payload}"  # Additional data from JSON script
  default_response_payload_template:  # Extract from API response
    agent_reply: "${generated_reply}"  # Map to your response field
    generated_metadata: "${metadata}"  # e.g., extracted entities

# REPOSITORY SECTION (Optional):
repository:
  type: FILESYSTEM  # Or FIRESTORE/MONGODB for persistence
  project_id: ""  # For FIRESTORE
  database_name: ""  # For FIRESTORE/MONGODB
  source: "LOCAL"  # Or IN_MEMORY for non-persistent
```

For the endpoint configuration section (`endpoint`), essentially, you need to provide:
* base_url
* url_path
* headers data: API Key, Bearer Token, or any additional header data.

As for the request payload, for example, if you have the following request payload schema:
```JSON
{
  "prompt": "Hello, world!",
  "user_id": "0001",
  "user_role": "ADMIN",
}
```
You need to configure the `default_request_payload_template` like the following:
```YAML
default_request_payload_template:
  prompt: "${user_message}"  # As you can notice, we only changed the field name and not the placeholder value.
  payload: "${request_payload}" # The rest of the data will be fetched from the "request_payload" field in the reference data JSON file.
```
while providing the rest of the payload request inside the reference data JSON file content:
```JSON
{
  "scripts": [
    {
      "interactions": [
        {
          "user_message": "Hello, I would like to book an appointment with a doctor.",
          "reference_reply": "Sure, I can help with that. Could you please specify the type of doctor you need to see?",
          "interaction_type": "initial",
          "reference_metadata": {},
          "guardrail_flag": false,
          "request_payload": {"user_id":  "0001", "user_role": "ADMIN"}  // Here we add the rest of the request payload data.
        }
...
```
And for the response payload, if you have the following response payload schema:
```JSON
{
  "response": "Hello, world!",
  "metadata": {"k1": "v1", "k2": "v2"},
  "timestamp": "2025-10-14T14:49:00.123Z",
  "status": "COMPLETE"
}
```
You need to configure the `default_response_payload_template` like the following:
```YAML
  default_response_payload_template:
    agent_reply: "${response}"  # we changed the placeholder value here by adding "response" field where the reply value is held.
    generated_metadata: "${metadata}"
```

### Step 4: Creating the JSON Conversation Script
The script defines simulation flows. It's a dict with a `scripts` array, each containing `interactions` (sequential turns).
<br>
Example `conversation_script.json`:
```JSON
{
  "scripts": [
    {
      "interactions": [
        {
          "user_message": "Hello, book a doctor appointment.",
          "reference_reply": "What type of doctor?",
          "interaction_type": "initial",
          "reference_metadata": {},
          "guardrail_flag": false,
          "request_payload": {"user_id": "123", "role": "user"}
        },
        {
          "user_message": "Cardiologist.",
          "reference_reply": "When?",
          "interaction_type": "intermediate",
          "reference_metadata": {"type": "Cardiology"},
          "guardrail_flag": false,
          "request_payload": {"user_id": "123", "role": "user"}
        },
        {
          "user_message": "Next Monday at 10 AM.",
          "reference_reply": "Booked for 10 AM next Monday.",
          "interaction_type": "final",
          "reference_metadata": {
            "appointment_type": "Cardiology",
            "date": "2025-10-20",
            "time": "10:00"
          },
          "guardrail_flag": false,
          "request_payload": {"user_id": "123", "role": "user"}
        }
      ],
      "description": "Doctor booking flow",
      "details": {"context": "Medical chatbot"}
    }
  ]
}
```
#### Technical Notes:

* **Schema Validation**: Interactions are validated against a schema (e.g., user_message: str, reference_metadata: dict).
* **Metadata Comparison**: generated_metadata from your endpoint is compared to reference_metadata using metrics_map.
* **Interaction Types**: initial/intermediate/final for flow control; can influence evaluator behavior (e.g., stricter on final turns).
* **Request Payload**: Merged into the endpoint request template for context (e.g., user auth).

### Step 5: Writing and Running the Python Script
Use this to load configs, run the simulation, and collect results. LevelApp handles session management via context managers.
<br>
Example run_simulation.py:
```Python
from dotenv import load_dotenv
from levelapp.workflow import WorkflowConfig
from levelapp.core.session import EvaluationSession


# Load .env (automatically done in LevelApp, but explicit for clarity)
load_dotenv()

if __name__ == "__main__":
    # Load YAML config (validates via Pydantic)
    config = WorkflowConfig.load(path="workflow_config.yaml")
    
    # Alternative: Load from dict for in-memory config (e.g., from DB)
    # config_dict = {...}  # As in README
    # config = WorkflowConfig.from_dict(content=config_dict)
    # config.set_reference_data(content={"scripts": [...]})  # Inline script

    # Create session (handles logging, repository init)
    with EvaluationSession(session_name="chatbot-sim-1", workflow_config=config) as session:
        # Run simulation: Sends requests, evaluates, stores in repo
        session.run()
        
        # Collect the evaluation results
        results = session.workflow.collect_results()
        print("Evaluation Results:", results) 

    stats = session.get_stats()
    print("Session Stats:\n", stats)
```

Technical Execution Flow:

1. `WorkflowConfig.load()`: Parses YAML, loads .env secrets, validates.
2. `EvaluationSession`: Initializes the evaluation session.
3. `session.run()`: Loops over scripts/interactions:
   * Substitutes placeholders, sends POST to endpoint.
   * Extracts chatbot reply and generated metadata.
   * Applies evaluators (e.g., JUDGE prompts LLM with "Score reply on scale 0-3: generated vs reference").
   * Computes metrics (e.g., Levenshtein distance via numpy).
4. `collect_results()`: Returns the evaluation results.
5. `get_stats()`: Retrieves monitoring stats (API calls details, caching details, processing time, etc.).

---
### Let's Test It:
First, install the packages required to run the examples test:
<br>
(it is always recommended to set up a virtual environment for testing)
```Bash
  pip install fastapi uvicorn levelapp
```
Second, run the chatbot (`example_chatbot.py`) using `uvicorn`:
<br>
(don't forget to add your `OPENAI_API_KEY`!)
```Bash
  uvicorn example_chatbot:app --reload --port 8000
```
Next, optionally, run a health test to see if the chatbot is alive:
```Bash
  curl http://localhost:8000/healthz
```
Finally, run the evaluation:
```Bash
  python example_evaluation.py
```

That's it! All you need now is to verify and interpret the evaluation results.
**Good luck!**
