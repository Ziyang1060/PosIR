from pathlib import Path
import json

base_dir = Path("evaluation_results/Qwen3-Embedding-0.6B")
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