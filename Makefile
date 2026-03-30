.PHONY: install train test app

install:
	pip install -r requirements.txt

train:
	python src/models/train.py

test:
	pytest tests/

app:
	python app/gradio_app.py
