## Getting Started

### Development
Create virtual environment
```bash
python -m venv venv
```

Activate python virtual environment
```bash
.\venv\Scripts\activate # windows
source venv/bin/activate  # unix
```

Install dependencies
```bash
pip install -r requirements.txt
```

Edit environment variebels
```bash
mv .env.example .env
```

Starting app in developer mode
```bash
uvicorn app.main:app --reload
```

### Production
Create user with for this app
```bash
sudo useradd -M -s /bin/false appuser 
```

Clone this repo
```bash
git clone https://github.com/CEPP-Project/MAPP-Strabismus-api.git
```

Change permission of this repo to appuser
```bash
sudo chown -R appuser:appuser ./MAPP-Strabismus-api
```

Edit environment variebles (docker and app)
```bash
mv .env.example .env.prod
mv .dockerenv.example .dockerenv
```

Starting app in docker
```bash
docker compose --env-file .dockerenv up -d --build
```

