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
	git pull origin update
	git switch update
	pip install -U "huggingface_hub[cli]"
    huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload Uthman89/farmers_credit ./apps.py /apps.py --repo-type=space --commit-message="Sync App file"
	huggingface-cli upload Uthman89/farmers_credit ./farmer.png /farmer.png --repo-type=space --commit-message="Sync App assets"
	huggingface-cli upload Uthman89/farmers_credit ./Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload Uthman89/farmers_credit ./Results /Results --repo-type=space --commit-message="Sync Results"
	huggingface-cli upload Uthman89/farmers_credit ./feature_names.pkl /feature_names.pkl --repo-type=space --commit-message="Sync feature names"

deploy: hf-login push-hub
all: install format train eval update-branch deploy
