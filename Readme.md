# Reservoire learning using the Izhikevich neuron model

## To develop it
Create a virtual env for Python:
```sh
python -m venv .env
```

Activate the virtual env:
```sh
source .env/bin/activate
```

Install the requirements:
```sh
pip install -r requirements.txt
```

Compile the rust packages:
```sh
maturin develop
```

Now you should be able to run the test file:
```sh
cd python
python3 framework_test.py
```


## Running sensitivity analysis
Install docker compose and run:
```sh
docker-compose up -d
```

You can check the connection by connnecting to the local db:
```
psql postgres://user:password@localhost:5432/db
```

Then you can run the sensitivity analysis
```
cd python
python3 sensitivity_plots.py
```
Note that this will take some time to finish.

