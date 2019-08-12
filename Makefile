repro:
	dvc repro eval.json.dvc
	dvc repro random_baseline.json.dvc
	cat resources/eval.json
	cat resources/random_baseline.json
