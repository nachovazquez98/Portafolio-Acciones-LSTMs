# Atajos locales. Requiere uv (https://docs.astral.sh/uv/) o un venv con pip.
PY := .venv/bin/python

install:            ## crea .venv e instala dependencias
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev]"

install-deep:       ## añade PyTorch para el motor LSTM
	uv pip install --python $(PY) -e ".[deep]"

run:                ## pipeline completo (ambos mercados)
	.venv/bin/stockcast run --market all

app:                ## dashboard en http://localhost:8501
	.venv/bin/stockcast app

test:
	$(PY) -m pytest -q

schedule:           ## instala el timer diario de systemd (usuario)
	mkdir -p ~/.config/systemd/user
	cp scripts/stockcast.timer ~/.config/systemd/user/
	sed "s|__REPO__|$(CURDIR)|" scripts/stockcast.service > ~/.config/systemd/user/stockcast.service
	systemctl --user daemon-reload && systemctl --user enable --now stockcast.timer
	systemctl --user list-timers stockcast.timer

.PHONY: install install-deep run app test schedule
