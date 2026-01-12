from pathlib import Path
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="Name of the model to aggregate results for")
args = parser.parse_args()

base_dir = Path(f"evaluation_results/{args.model_name}")
model_name = base_dir.name
adict = {}

for subdir in base_dir.iterdir():
    if subdir.is_dir():
        adict[subdir.name] = {}
        for f in subdir.iterdir():        
            if f.suffix == ".json":
                domain_name = f.name.replace(".json", "")
                print(f)
                with open(f) as fp:
                    data = json.load(fp)
                adict[subdir.name][domain_name] = {
                    "overall_scores": data["overall_scores"],
                    "evaluation_results": data["evaluation_results"]
                }
        print(len(adict[subdir.name]))
        assert len(adict[subdir.name]) == 31

with open(base_dir / f"{model_name}.json", "w") as f:
    f.write(json.dumps(adict, ensure_ascii=False, indent=4))