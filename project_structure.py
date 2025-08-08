import os
from pathlib import Path

# Define the complete directory structure for the project
directories = [
    "data/market_data",
    "data/news",
    "data/processed",
    "data/raw",
    "data/vector_store",
    "src/ai_core",
    "src/alerting",
    "src/data_collection",
    "src/dashboard/components",
    "src/document_processing",
    "src/models",
    "src/rag",
    "src/scheduler",
    "src/utils",
    "models/saved",
]

# Define all the Python files to be created (including __init__.py for packages)
files = [
    "main.py",
    "requirements.txt",
    "README.md",
    ".env.example",
    "src/ai_core/__init__.py",
    "src/ai_core/feature_engineering.py",
    "src/ai_core/qualitative_brain.py",
    "src/ai_core/quantitative_brain.py",
    "src/alerting/__init__.py",
    "src/alerting/alert_system.py",
    "src/data_collection/__init__.py",
    "src/data_collection/market_data.py",
    "src/data_collection/news.py",
    "src/data_collection/sec_filings.py",
    "src/dashboard/__init__.py",
    "src/dashboard/streamlit_app.py",
    "src/dashboard/components/__init__.py",
    "src/dashboard/components/style.css",
    "src/document_processing/__init__.py",
    "src/document_processing/parser.py",
    "src/models/__init__.py",
    "src/models/pca_health_scorer.py",
    "src/models/xgboost_predictor.py",
    "src/rag/__init__.py",
    "src/rag/query_processor.py",
    "src/rag/retrieval_engine.py",
    "src/scheduler/__init__.py",
    "src/scheduler/task_scheduler.py",
    "src/utils/__init__.py",
    "src/utils/config.py",
    "src/utils/financial_ratios.py",
]

def create_project_structure():
    """
    Creates the defined directory and file structure for the project.
    """
    print("Creating project structure...")

    # Create all directories
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"  Created directory: {directory}")
        except Exception as e:
            print(f"  Error creating directory {directory}: {e}")

    # Create all empty files
    for file_path in files:
        try:
            Path(file_path).touch(exist_ok=True)
            print(f"  Created file:      {file_path}")
        except Exception as e:
            print(f"  Error creating file {file_path}: {e}")
    
    # Add a placeholder to the .gitignore file
    try:
        with open(".gitignore", "w") as f:
            f.write("# Python\n")
            f.write("__pycache__/\n")
            f.write("*.pyc\n")
            f.write("\n# Environment\n")
            f.write(".env\n")
            f.write("venv/\n")
            f.write("\n# Data & Models\n")
            f.write("data/\n")
            f.write("models/saved/*.pkl\n")
            f.write("models/saved/*.json\n")
            f.write("\n# Logs\n")
            f.write("*.log\n")
        print("  Created file:      .gitignore")
    except Exception as e:
        print(f"  Error creating .gitignore: {e}")


    print("\nProject structure created successfully!")
    print("You can now start populating the files with the provided code.")

if __name__ == "__main__":
    create_project_structure()