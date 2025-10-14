from dotenv import load_dotenv
from levelapp.workflow.schemas import WorkflowConfig
from levelapp.core.session import EvaluationSession

# Load .env (automatically done in LevelApp, but explicit for clarity)
load_dotenv()

if __name__ == "__main__":
    # 1. Load YAML config
    config = WorkflowConfig.load(path="workflow_config.yaml")

    # Alternatively: Load from dict for in-memory config (e.g., from DB)
    # config_dict = {...}  # As in README
    # config = WorkflowConfig.from_dict(content=config_dict)
    # config.set_reference_data(content={"scripts": [...]})  # Inline script

    # 2. Create an evaluation session
    with EvaluationSession(session_name="chatbot-sim-1", workflow_config=config) as session:
        # 2.1. Run session (simulation session)
        session.run()

        # 2.2. Collect evaluation results
        results = session.workflow.collect_results()
        print("Evaluation Results:", results)

    # 3. Get aggregated stats (monitoring stats)
    stats = session.get_stats()
    print("Session Stats:\n", stats)
