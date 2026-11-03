"""Build a standalone notebook embedding the versioned project sources."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip().splitlines(True)})
def code(text):
    text = text.strip() + '\n'
    if not text.startswith('%pip'):
        ast.parse(text)
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.splitlines(True)})

md('''# Domain-Specific Text Generation for Machine Translation
## A resource-constrained reproduction and filtering ablation (Arabic to English)
**Student:** Nguyễn Hoàng Anh Kiệt | **ID:** DE200278 | **Class:** AI20B

Based on Moslem et al. (AMTA 2022). Repository: https://github.com/Anhkietttt/TMG301

This notebook is self-contained: its source modules are embedded below, so no GitHub login, repository clone, manually uploaded data, or API key is needed. Enable **GPU and Internet**, start a fresh Kaggle session, and **Run All**. It downloads public models and datasets and performs real generation, back-translation, training and evaluation. There are no prefilled experimental scores. Keep `/kaggle/working/domain_mt_runs` when resuming within the same environment. Across sessions, copy a saved run folder from an attached Kaggle output into that directory before running; do not overwrite it with a different configuration.

**Scope:** only Setup 1, Arabic-to-English. Pythia-410M replaces GPT-J-6B; pretrained Marian checkpoints replace the paper's trained Transformer-Big baselines. We use small, document-disjoint TICO subsets and OPUS-100 replay, not the original 40-million-pair training recipe. This is a method reproduction, NOT an exact score reproduction. All choices are frozen before final testing.
''')
requirements = (ROOT / 'requirements.txt').read_text().splitlines()
packages = [r for r in requirements if r and not r.startswith('#')]
code('%pip install -q ' + ' '.join('"' + r + '"' for r in packages))
sources = {f'src/domainmt/{name}': (ROOT / 'src/domainmt' / name).read_text() for name in ['__init__.py', 'data.py', 'experiment.py']}
sources['configs/pilot.json'] = (ROOT / 'configs/pilot.json').read_text()
sources['tests/test_data.py'] = (ROOT / 'tests/test_data.py').read_text()
code('''import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import sys
from pathlib import Path
EMBEDDED_FILES = json.loads(''' + repr(json.dumps(sources, ensure_ascii=False)) + ''')
PROJECT = Path("/kaggle/working/tmg301_source") if Path("/kaggle").exists() else Path.cwd() / "tmg301_source"
for relative, content in EMBEDDED_FILES.items():
    path = PROJECT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
sys.path.insert(0, str(PROJECT / "src"))
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from datasets import disable_progress_bars
from transformers.utils import logging
disable_progress_bars()
logging.disable_progress_bar()
pd.set_option("display.max_colwidth", 180)
from domainmt.experiment import Experiment
config = json.loads((PROJECT / "configs/pilot.json").read_text())
display(pd.Series(config, name="setting").to_frame())
''')
md('''## 1. Fixed protocol and environment
The pilot runs all five systems: A pretrained baseline; B authentic mixed fine-tuning; C randomly sampled synthetic mixed fine-tuning; D quality-ranked synthetic mixed fine-tuning; E parameter averaging of D and A with fixed weight 0.5. Each adaptation starts independently from A. C and D use equal synthetic-pool sizes and identical training update/example budgets; actual token counts are exported because sequence lengths differ. Training samples in-domain examples with probability 0.9 and generic examples with probability 0.1. This is the sampling probability, not the stored corpus-size ratio.

Checkpoints for B/C/D are selected by **domain dev loss**, never test metrics. The experiment uses one seed and small held-out sets; it can support a pilot result, not a definitive significance claim. An improvement may fail or harm general-domain performance; negative results must be reported.
''')
code('''import subprocess
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(PROJECT / "tests"), "-v"],
               env={**os.environ, "PYTHONPATH": str(PROJECT / "src")}, check=True)
output_root = Path("/kaggle/working/domain_mt_runs") if Path("/kaggle").exists() else Path.cwd() / "domain_mt_runs"
experiment = Experiment(config, output_root)
display(pd.Series(experiment.versions, name="version").to_frame())
''')
md('''## 2. Authentic data and leakage controls
TICO-19 is downloaded from the official archive. Both official partitions are combined and **repartitioned by document ID** before model use; these are custom research splits, not official benchmark scores. Cross-split exact source/target duplicates are removed and deterministic subsets are drawn. Only 200 training pairs provide domain prompts and authentic training data. Test references are excluded from generation/training; they are used only for mechanical overlap exclusion and final scoring, never as generation prompts or quality-ranking features.

OPUS-100 Arabic-English supplies generic replay and separate generic dev/test samples. Exact overlaps with TICO and other sampled partitions are excluded. Pretraining contamination cannot be ruled out for public checkpoints. Raw strings and provenance are retained. No dataset redistribution or blanket license claim is made.
''')
code('''display(experiment.prepare_data())
display(pd.DataFrame(experiment.data["domain_train"]).head(5))
print("Saved document partitions and all data subsets:", experiment.root)
''')
md('''## 3. Generate English target-side domain text
English training sentences prompt a small English causal LM using top-k 50, top-p 0.95 and three sampled continuations per prompt. Prompt tokens are removed before collecting complete sentences. Basic filtering removes duplicates, non-English text, incomplete fragments, and exact/high-Jaccard copies of seed or evaluation sentences. These controls reduce memorization risk but cannot prove absence of all semantic overlap. Prompt provenance and raw continuations are saved; progress is checkpointed per prompt. A small LM may drift off-domain, and synthetic health statements must not be treated as medical advice.
''')
code('''targets = experiment.generate_targets()
print("Accepted English targets:", len(targets))
display(targets.head(10))
''')
md('''## 4. Back-translation and matched quality-filtering ablation
Translate generated English into Arabic with the reverse model (beam 5). The resulting training direction is `(synthetic Arabic -> generated English)`. Reject wrong-script/duplicate/overlength pairs and held-out near-copies. Rank remaining pairs by **0.7 bilingual embedding cosine + 0.3 domain-centroid cosine**; the domain centroid uses authentic training English only. The coefficients are fixed a priori. This is a quality proxy, not a semantic guarantee.

C randomly draws K pairs; D takes the top K from the same eligible pool, where K is at most 400 and at most half the pool. Both have the same K. The minimum-size guard stops an uninformative run rather than fabricating results. If it fails, inspect generation and increase the generation budget in a new run before any final-test evaluation. This recipe is a methodological extension, not the paper's original 0.45 semantic threshold.
''')
code('''display(experiment.backtranslate_and_rank())
quality = pd.read_csv(experiment.root / "quality_audit.csv")
display(quality[["en", "ar", "bilingual_similarity", "domain_similarity", "quality_score"]].head(8))
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(quality.quality_score, bins=25, color="#287f8e", edgecolor="white")
ax.set(title="Synthetic quality-score distribution", xlabel="Heuristic score", ylabel="Pairs")
fig.tight_layout()
fig.savefig(experiment.root / "quality_scores.png", dpi=150)
plt.show()
''')
md('''## 5. Independent mixed fine-tuning and checkpoint averaging
Each of B, C and D starts from the same pretrained weights and uses 300 optimizer steps, batch size 4 with accumulation 4, learning rate 2e-5 and maximum source/target length 128. Validation and checkpointing occur every 100 steps. Full-precision training is chosen for stability; only one GPU is used even when two are available. Interrupted training can resume from a checkpoint in the same run folder. Completed variants are cached.

E averages corresponding parameters of A and D at fixed 0.5/0.5 weight. This is weight averaging between compatible models, not a prediction ensemble and not averaging unrelated architectures. No test score determines the coefficient.
''')
code('''training = experiment.train_all()
display(training)
print("Averaged model:", experiment.average_checkpoint())
fig, ax = plt.subplots(figsize=(9, 4))
for name in ["B_authentic_mix", "C_synthetic_random", "D_synthetic_ranked"]:
    history = pd.read_csv(experiment.root / name / "training_history.csv")
    if "eval_loss" in history:
        points = history.dropna(subset=["eval_loss"])
        ax.plot(points.step, points.eval_loss, marker="o", label=name)
ax.set(title="Domain-development loss (checkpoint selection)", xlabel="Optimizer step", ylabel="Cross-entropy loss")
ax.legend()
fig.tight_layout()
fig.savefig(experiment.root / "dev_loss.png", dpi=150)
plt.show()
''')
md('''## 6. Frozen evaluation on both domains
All five predeclared systems are evaluated on the same domain and generic sets using beam 4. Predictions and scores are cached and reused; inspecting test scores must not lead to tuning and retesting on these same sets. We report corpus BLEU (13a), chrF++ (word order 2), TER, sample counts and synchronized GPU runtime. **BLEU here is not the paper's spBLEU**, and neither metric nor split is identical to the original table. COMET and native-Arabic expert human evaluation are not included in this pilot. Downloads/training are excluded from generation timing.
''')
code('''results = experiment.evaluate_all()
display(results[["model", "split", "n", "BLEU", "chrF++", "TER", "seconds"]])
test_results = results[results.split.isin(["domain_test", "generic_test"])]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, split in zip(axes, ["domain_test", "generic_test"]):
    frame = test_results[test_results.split == split].set_index("model")
    ax.bar([name[:1] for name in frame.index], frame["chrF++"], color=["#65717a", "#3879a6", "#259783", "#b65f74", "#9b843e"])
    ax.set(title=split, xlabel="System A-E", ylabel="Corpus chrF++", ylim=(0, 100))
fig.tight_layout()
fig.savefig(experiment.root / "test_comparison.png", dpi=150)
plt.show()
display(pd.Series(experiment.bootstrap_improvement(), name="result").to_frame())
''')
md('''## 7. Inspect actual errors, not invented explanations
The following table joins real outputs by sentence ID. Low overlap is only a screening signal. Compare terminology, negation, quantities, omissions and hallucinations against the Arabic source and English reference. Consult a competent Arabic speaker for a defensible human evaluation. The exported review sheet intentionally leaves human judgements empty: code cannot truthfully claim that these have already been assessed.
''')
code('''from sacrebleu.metrics import CHRF
review = pd.read_csv(experiment.root / "A_baseline_domain_test_predictions.csv").rename(columns={"prediction_en": "A_prediction"})
for name in ["B_authentic_mix", "C_synthetic_random", "D_synthetic_ranked", "E_ranked_average"]:
    frame = pd.read_csv(experiment.root / f"{name}_domain_test_predictions.csv")
    review = review.merge(frame[["id", "prediction_en"]].rename(columns={"prediction_en": name[0]+"_prediction"}), on="id", validate="one_to_one")
metric = CHRF(word_order=2)
review["D_sentence_chrF++"] = [metric.sentence_score(p, [r]).score for p, r in zip(review.D_prediction, review.reference_en)]
review = review.sort_values("D_sentence_chrF++")
review["human_error_category"] = ""
review["human_explanation"] = ""
review.to_csv(experiment.root / "human_review.csv", index=False)
display(review.head(10))
''')
md('''## 8. Measured conclusion and artifacts
The summary below reports whether ranking helped in this run, including degradation if present. It does not promise the paper's gains. Before presenting the project, add a qualitative discussion using actual outputs and repeat important comparisons with more training seeds when resources permit. Do not use the current test set for further model selection.
''')
code('''lookup = results.set_index(["model", "split"])
c = lookup.loc[("C_synthetic_random", "domain_test"), "chrF++"]
d = lookup.loc[("D_synthetic_ranked", "domain_test"), "chrF++"]
a_general = lookup.loc[("A_baseline", "generic_test"), "chrF++"]
d_general = lookup.loc[("D_synthetic_ranked", "generic_test"), "chrF++"]
summary = f"Quality-ranked D minus random-synthetic C: {d-c:+.3f} domain-test chrF++ points. D minus pretrained A: {d_general-a_general:+.3f} generic-test chrF++ points. These are measured pilot results, not guaranteed improvements. All adaptations use the same update count; token budgets, source data and pretrained models differ from the original paper. Human evaluation remains pending."
display(Markdown(summary))
(experiment.root / "measured_summary.txt").write_text(summary)
assert len(results) == 20 and results[["BLEU", "chrF++", "TER"]].notna().all().all()
assert len(experiment.pools["synthetic_random"]) == len(experiment.pools["synthetic_ranked"])
import zipfile
archive = experiment.root / "reports_and_predictions.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for path in experiment.root.rglob("*"):
        if path.is_file() and path.suffix in [".csv", ".json", ".png", ".txt"] and "checkpoints" not in path.parts and "model" not in path.parts:
            z.write(path, path.relative_to(experiment.root))
print("Reports:", archive)
print("Models: B/C/D/E subfolders under", experiment.root)
print("Save Version with outputs and download this executed notebook. No model weights or dataset files are pushed to GitHub.")
''')
md('''## References and limitations
- Moslem et al. (2022): https://aclanthology.org/2022.amta-research.2/
- Authors' implementation: https://github.com/ymoslem/MT-LM
- TICO-19: https://tico-19.github.io/testset.html
- Generic corpus: https://huggingface.co/datasets/Helsinki-NLP/opus-100
- MT: https://huggingface.co/Helsinki-NLP/opus-mt-ar-en and https://huggingface.co/Helsinki-NLP/opus-mt-en-ar
- English LM: https://huggingface.co/EleutherAI/pythia-410m
- Quality encoder: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Metrics: https://github.com/mjpost/sacrebleu

The code is an independently written, AI-assisted implementation of the method; it does not copy the author's implementation. Cite the paper and disclose assistance according to course rules. Public-data licensing, potential model-pretraining overlap, limited compute, small sample size, one seed, heuristic filtering and lack of native-speaker evaluation constrain the conclusions. Generated health-domain material is experimental synthetic text and must not be used as medical advice.
''')
for i, cell in enumerate(cells):
    cell['id'] = f'domain-mt-{i:02d}'
notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python', 'version': '3.12.0'}}}
path = ROOT / 'notebooks/DE200278_Domain_Adaptation_AR_EN_Kaggle.ipynb'
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
print(path)
