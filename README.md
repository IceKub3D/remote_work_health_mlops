# Remote Work Health MLOps Project

This project deploys a machine learning model from the Kaggle notebook "Health Impacts of Remote 
Work"[](https://www.kaggle.com/code/calebboen/health-impacts-of-remote-work) to predict health outcomes of 
remote work using a FastAPI endpoint.

## Project Structure
remote_work_health_mlops/
├── .github/workflows/      # GitHub Actions for CI/CD
├── data/                   # Raw and processed data (versioned with DVC)
├── models/                 # Serialized models (versioned with DVC)
├── src/                    # Python scripts for preprocessing, training, serving
├── tests/                  # Unit tests
├── k8s/                    # Kubernetes configurations
├── notebooks/              # Kaggle notebook
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── .dvcignore              # DVC ignore file
├── .gitignore              # Git ignore file
├── dvc.yaml                # DVC pipeline configuration
├── README.md               # Project documentation
├── setup.sh                # Setup script

## Setup
1. Clone the repository:

   git clone github.com/IceKub3D/remote_work_health_mlops
   cd remote_work_health_mlops

2. Activate virtual environment (assumes already created):

source .venv/bin/activate

3. Install dependencies:

./setup.sh

4. Configure DVC remote storage (e.g., AWS S3):

dvc remote add -d myremote s3://mybucket

**Requirements**
Python 3.9+
Dependencies listed in requirements.txt

**License**
MIT License


