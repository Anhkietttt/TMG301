# Sources and redistribution

| Asset | Source | Use |
|---|---|---|
| TICO-19 | https://tico-19.github.io/testset.html | Public-health parallel text, custom document splits |
| OPUS-100 | https://huggingface.co/datasets/Helsinki-NLP/opus-100 | Generic replay and evaluation |
| Arabic-English Marian | https://huggingface.co/Helsinki-NLP/opus-mt-ar-en | Baseline and fine-tuning |
| English-Arabic Marian | https://huggingface.co/Helsinki-NLP/opus-mt-en-ar | Back-translation |
| Pythia-410M | https://huggingface.co/EleutherAI/pythia-410m | English target generation |
| multilingual MiniLM | https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Heuristic quality ranking |

The Marian and Pythia cards identify Apache-2.0 at inspection time. Verify upstream licenses
and revisions when distributing derived models. The OPUS-100 card lists its overall license
as unknown; constituent corpora have their own conditions. TICO rows include a license column,
which is retained. This repo distributes code and an unexecuted notebook, not raw corpora or
model weights. Downloading an asset for research does not establish unrestricted redistribution
rights. Do not automatically upload the report archive to a public repo without checking it.

All source code here is independently implemented from the described method. Cite upstream
authors and models. The original MT-LM repo is a reference, not vendored source.
