"""Model provenance: source repositories, revisions and weight-file hashes of the two frozen models.

    python3 -I checks/03_model_provenance.py

Reads only metadata (git revisions, Hugging Face download metadata, file hashes); no external code is run.
Writes checks/03_model_provenance.json, used by Supplementary Table S3.
"""
import hashlib
import json
import subprocess
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
EXT = PROJ / "external"
INTRO = EXT / "IntroECG"
ECHONEXT_W = INTRO / "7-EchoNext Minimodel/models/echonext_multilabel_minimodel/weights.pt"
DOCKER = EXT / "deepecg/DeepECG_Docker"
HF = EXT / "deepecg/weights/deepecg-sl_finetuned_LVEF_MHI"
DEEPECG_W = HF / "LVEF_MSE_SL.pt"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo):
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "date": run("log", "-1", "--format=%cs"), "remote": run("remote", "get-url", "origin")}


def hf_revision(path):
    # huggingface_hub download metadata: line 1 = repo revision, line 2 = etag
    return (HF / ".cache/huggingface/download" / f"{path.name}.metadata").read_text().splitlines()[0]


def license_of(repo):
    files = [p.name for p in repo.iterdir() if p.name.upper().startswith(("LICENSE", "LICENCE", "COPYING"))]
    return files or None


def main():
    card = (HF / "README.md").read_text()
    out = {"echonext_mini": {"repository": git_head(INTRO), "weights": str(ECHONEXT_W.relative_to(INTRO)),
                             "weights_sha256": sha256(ECHONEXT_W), "license_files": license_of(INTRO)},
           "deepecg_sl": {"hf_repo": "heartwise/deepecg-sl_finetuned_LVEF_MHI", "hf_revision": hf_revision(DEEPECG_W),
                          "weights": DEEPECG_W.name, "weights_sha256": sha256(DEEPECG_W),
                          "license": next((l.split(":", 1)[1].strip() for l in card.splitlines() if l.startswith("license:")), None),
                          "preprocessing_repository": git_head(DOCKER)}}
    (PROJ / "checks/03_model_provenance.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
