import json
from core.runner import ODVM

with open("config.json", "r") as f:
    config = json.load(f)

odvm = ODVM(data="data1.csv", target="Profit", config=config)
odvm.run(eda=True, preprocess=True, model=True, report=True, deploy=False)