### Get Started
```bash
# Create virtual environment
python -m venv venv

# Activate python virtual environment (windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Starting app in developer mode
uvicorn app.main:app --reload
```