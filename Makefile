ENV_FOLDER=./environment
VENV_NAME=edl_venv
VENV_PATH=$(ENV_FOLDER)/$(VENV_NAME)
VENV_ACTIVATE_PATH=$(VENV_PATH)/bin/activate
REQUIREMENTS_PATH=$(ENV_FOLDER)/requirements.txt


help:
	@echo "======================== Project makefile  ========================"
	@echo "Use make {command} with any of the below options:"
	@echo "    * create_env - creating the project virtual environment"
	@echo "    * activate-env-command - printing the command to activate environment in the console"
	@echo "    * run-tensorboard - view training results via tensorboard" 

create-env:
	@echo "======================== Creating the project virtual environment ========================" 
	python3 -m virtualenv --system-site-packages -p python3.6 $(VENV_PATH)
	. $(VENV_ACTIVATE_PATH) && \
	python3 -m pip install pip --upgrade && \
	python3 -m pip install -r $(REQUIREMENTS_PATH)

activate-env-command:
	@echo "======================== Execute the below command in terminal ========================" 
	@echo source $(VENV_ACTIVATE_PATH)

run-tensorboard:
	@echo "======================== Run the displayed link in your browser to view training results via tensorboard ========================" 
	tensorboard --logdir ./output/

purge-output:
	rm -r output/*