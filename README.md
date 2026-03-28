# 1. Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# 2. Create the Conda environment
# This will install the specific Python version and packages
conda create --name cad_env python=3.9

# 3. Activate the environment
conda activate cad_env

# 4. Install dependencies
pip install -r requirements.txt

# OR if you have an environment.yml file (recommended for Conda):
# conda env update --file environment.yml --prune

# 5. Run the project
python autocheck360.py
