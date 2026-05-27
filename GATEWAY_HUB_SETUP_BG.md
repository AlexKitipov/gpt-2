# Gateway-Hub: бърз setup на среда (Windows / macOS / Linux)

Тъй като достъпът до GitHub може да е блокиран (напр. `CONNECT tunnel failed, response 403`), ето сигурен начин да си създадеш средата локално.

## 1) Клониране на репото

```bash
git clone https://github.com/AlexKitipov/Gateway-Hub.git
cd Gateway-Hub
```

Ако клонирането даде 403:
- провери дали имаш корпоративен proxy/VPN;
- пробвай от домашна мрежа;
- или използвай GitHub Desktop/ZIP Download като временен workaround.

## 2) Създай виртуална среда

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3) Инсталиране на зависимости

Използвай това, което съществува в репото:

```bash
# ако има requirements.txt
pip install -r requirements.txt

# ако е Poetry проект
poetry install

# ако е Pipenv проект
pipenv install
```

## 4) Стартиране

Провери README.md в репото и изпълни описаната команда (например `python main.py`, `uvicorn ...`, `docker compose up` и т.н.).

## 5) Бърза диагностика

```bash
python --version
pip --version
which python   # Windows: where python
```

Ако искаш, мога да ти дам и **готови команди 1:1** според твоята ОС (Windows/macOS/Linux) и грешката, която ти излиза.
