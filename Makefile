repro:
	dvc repro eval.json.dvc
	dvc repro random_baseline.json.dvc
	cat resources/eval.json
	cat resources/random_baseline.json

init:
	python3 -m venv venv
	source venv/bin/activate
	pip3 install -r requirements.txt
