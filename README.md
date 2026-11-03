# TMG301: Domain-Specific Text Generation for Machine Translation

Nguyen Hoang Anh Kiet - DE200278 - AI20B

Resource-constrained, independently written reproduction of **Setup 1, Arabic to English**
from [Moslem et al., AMTA 2022](https://aclanthology.org/2022.amta-research.2/),
with a matched-size quality-filtering ablation. See the
[authors' code](https://github.com/ymoslem/MT-LM) for the original implementation.

## Run on Kaggle

1. Import [`notebooks/DE200278_Domain_Adaptation_AR_EN_Kaggle.ipynb`](notebooks/DE200278_Domain_Adaptation_AR_EN_Kaggle.ipynb).
2. Enable **GPU + Internet**. Start a fresh session. Use the existing CUDA PyTorch installation.
3. Run All. No dataset upload, GitHub credentials, or translation API is required.
4. Save Version with outputs. Download the notebook and `reports_and_predictions.zip`.
5. Keep run directories and checkpoints to resume; actual duration depends on the GPU,
   downloads and generation yield. This is a training experiment, not a seconds-long inference demo.

The notebook embeds the repository modules. It does not download mutable code at execution time.
Its first cell installs dependencies; unrelated Kaggle packages may report dependency conflicts.
Do not upgrade every preinstalled package to suppress warnings. A traceback in a used package
requires diagnosis. Never replace CUDA PyTorch with a CPU wheel.

## Experiments

| System | Description |
|---|---|
| A | Pretrained Marian Arabic-English baseline |
| B | Authentic TICO training subset + generic replay |
| C | Random K synthetic pairs + generic replay |
| D | Quality-ranked K synthetic pairs + generic replay |
| E | Fixed 50/50 weight average of A and D |

C/D have equal pool sizes, update counts and sampled example counts. Report token-budget
differences too; equal updates are not identical token budgets. All adapted models restart
from the same A weights. Generation, reverse MT, ranking and training use only training
data, apart from mechanical held-out overlap exclusion. Checkpoint selection uses domain
development loss. No test result controls settings. Do not rerun tuned experiments against
the same test set and present it as untouched evaluation.

## Differences from the paper

- Pythia-410M, not GPT-J-6B; smaller generation budget.
- Pretrained Marian, not a Transformer-Big trained on 40M pairs.
- Custom document-disjoint TICO split; no test-reference prompting.
- OPUS-100 replay subset; not the original mixture or exact corpus-size ratio.
- 0.9 in-domain **sampling probability**, not a claim that 90% of stored rows are in-domain.
- Fixed-budget full fine-tuning; no Setup 2 and no English-to-Arabic main experiment.
- BLEU 13a / chrF++ / TER, not paper spBLEU / historical COMET.
- New embedding-based ranking ablation and cluster bootstrap.

These changes prevent direct numerical comparison with the paper's headline scores.
See [protocol](docs/protocol.md) and [data/model provenance](docs/data_and_models.md).

## Repository layout

```text
configs/pilot.json        Fixed pilot budget
src/domainmt/data.py      Cleaning, overlap checks, document partitions
src/domainmt/experiment.py Staged generation, BT, training, averaging, evaluation
scripts/build_notebook.py Build standalone notebook from the same source
notebooks/               Kaggle deliverable (unexecuted until a real GPU run)
tests/                   Dependency-light data tests
docs/                    Protocol, source and licensing notes
```

## Local checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
python scripts/build_notebook.py
```

For full execution, install requirements in a fresh CUDA environment with PyTorch >=2.6.
Do not claim successful GPU execution merely because unit tests pass.
The initial repository contains no invented metrics or pretrained output tables.

## Resume and artifacts

Outputs: `/kaggle/working/domain_mt_runs/<config-code-version-hash>/`.
Completed generation/data/evaluation stages are cached. Generation checkpoints every prompt;
training checkpoints every 100 updates. Reverse translation currently checkpoints at the end
of the stage. Across Kaggle sessions, attach the previous output and copy the run folder back
to the working location. Package/config/code changes create a different run identity. Remote
model/data revisions can change; commits and dataset hashes are recorded where available,
and strict long-term reproduction requires preserving those revisions and downloaded assets.

The final report archive includes input subsets, synthetic examples, configuration, predictions
and metrics. Treat it as research data, review source licenses before redistributing it.
Models/checkpoints and data are intentionally ignored by Git. Never commit tokens or secrets.

## Reporting honestly

Generated health text is experimental and may be false. No clinical deployment is intended.
Embedding ranking is not proof of translation correctness. A pretrained model may already have
seen public evaluation sentences. One seed and a small document sample are not conclusive.
Human evaluation is left pending for a qualified Arabic reader; it is not fabricated by code.
The project was developed with AI assistance; follow course disclosure rules.

```bibtex
@inproceedings{moslem-etal-2022-domain,
  title = {Domain-Specific Text Generation for Machine Translation},
  author = {Moslem, Yasmin and Haque, Rejwanul and Kelleher, John D. and Way, Andy},
  booktitle = {Proceedings of the 15th Biennial Conference of the Association for Machine Translation in the Americas (Volume 1: Research Track)},
  year = {2022}, pages = {14--30},
  url = {https://aclanthology.org/2022.amta-research.2/}
}
```
