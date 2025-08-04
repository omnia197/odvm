import json
from core.runner import ODVM

with open("config.json", "r") as f:
    config = json.load(f)

odvm = ODVM(data="data3.csv", target="Weekly_Sales", config=config)
odvm.run(eda=True, preprocess=True, model=True)
