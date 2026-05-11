install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	echo "" >> report.md
	echo "## Confusion Matrix" >> report.md
	echo "![Confusion Matrix](./Results/model_results.png)" >> report.md
	cml comment create report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git add Results/ farmer_optimized_xgb.json feature_names.pkl
	git commit -m "Update model and results [skip ci]"
	git push --force origin HEAD:update

hf-login:
	pip install -U "huggingface_hub[cli]"
	hf auth login --token $(HF)

fetch-model:
	git fetch origin update
	git checkout origin/update -- Model/ Results/ feature_names.pkl

push-hub:
	hf upload Uthman89/farmers_credit ./Apps/requirements.txt /requirements.txt --repo-type=space --commit-message="Sync requirements"
	hf upload Uthman89/farmers_credit ./Apps/apps.py /apps.py --repo-type=space --commit-message="Sync App file"
	hf upload Uthman89/farmers_credit ./feature_names.pkl /feature_names.pkl --repo-type=space --commit-message="Sync feature names"
	hf upload Uthman89/farmers_credit ./Model /Model --repo-type=space --commit-message="Sync Model"
	hf upload Uthman89/farmers_credit ./Results /Results --repo-type=space --commit-message="Sync Results"

deploy: hf-login fetch-model push-hub
