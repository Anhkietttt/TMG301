"""Staged, resumable Kaggle experiment; no fabricated scores or silent fallbacks."""
import gc
import hashlib
import importlib.metadata
import json
import math
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sacrebleu.metrics import BLEU, CHRF, TER
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, set_seed)

from .data import (assert_disjoint, clean_pairs, continuation_sentences, is_arabic,
                   normalize, read_tico, split_documents, text_hash, token_jaccard)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temp.replace(path)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Experiment:
    def __init__(self, config, output_root):
        self.cfg = dict(config)
        if config['train_steps'] % config['eval_steps']:
            raise ValueError('train_steps must be divisible by eval_steps for final checkpoint evaluation.')
        if not 0 < config['in_domain_probability'] < 1:
            raise ValueError('Mixed sampling requires a probability strictly between zero and one.')
        if not 0 < config['quality_keep_fraction'] <= 0.5:
            raise ValueError('Matched ranking ablation keeps at most half the eligible pool.')
        if config['prompt_count'] > config['authentic_pairs']:
            raise ValueError('prompt_count cannot exceed the authentic training pool.')
        if not torch.cuda.is_available():
            raise RuntimeError('Enable a Kaggle GPU in Settings and restart. This training notebook requires CUDA.')
        self.device = 'cuda:0'
        self.seed = config['seed']
        set_seed(self.seed)
        self.versions = {p: importlib.metadata.version(p) for p in
                         ['torch', 'transformers', 'datasets', 'sacrebleu', 'sentence-transformers']}
        code_digest = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name('data.py').read_bytes()).hexdigest()
        identity = {'config': config, 'versions': self.versions, 'code_digest': code_digest}
        run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        self.root = Path(output_root) / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / 'run_config.json', identity)
        revisions = self.root / 'model_revisions.json'
        self.model_revisions = json.loads(revisions.read_text()) if revisions.exists() else {}
        print('Run:', self.root, '| GPU:', torch.cuda.get_device_name(0), flush=True)

    def _record_model(self, name, model):
        self.model_revisions[name] = getattr(model.config, '_commit_hash', None)
        write_json(self.root / 'model_revisions.json', self.model_revisions)

    def _subset(self, rows, n):
        if len(rows) < n:
            raise ValueError(f'Need {n} rows, only {len(rows)} remain. Review counts; do not silently change the protocol.')
        return random.Random(self.seed).sample(rows, n)

    def prepare_data(self):
        cache = self.root / 'prepared_data.json'
        if cache.exists():
            self.data = json.loads(cache.read_text())
            assert_disjoint(self.data)
            return self.data_summary()
        archive = self.root / 'tico19-testset.zip'
        if not archive.exists():
            urllib.request.urlretrieve(self.cfg['tico_url'], archive)
        tico = read_tico(archive)
        partitions = split_documents(tico, self.seed)
        data = {f'domain_{key}': self._subset(partitions[key], self.cfg[size])
                for key, size in [('train', 'authentic_pairs'), ('dev', 'domain_dev'), ('test', 'domain_test')]}
        assert_disjoint(data)
        # All TICO strings are excluded from the general-domain pools, independent of outcomes.
        forbidden_ar = {text_hash(r['ar']) for r in tico}
        forbidden_en = {text_hash(r['en']) for r in tico}
        tokenizer = AutoTokenizer.from_pretrained(self.cfg['mt_model'])
        for split, key in [('train', 'generic_train'), ('validation', 'generic_dev'), ('test', 'generic_test')]:
            print('Loading OPUS-100', split, flush=True)
            stream = load_dataset('Helsinki-NLP/opus-100', 'ar-en', split=split, streaming=True)
            stream = stream.shuffle(seed=self.seed, buffer_size=5000)
            selected = []
            for i, item in enumerate(stream):
                pair = item['translation']
                if not clean_pairs([pair]):
                    continue
                if text_hash(pair['ar']) in forbidden_ar or text_hash(pair['en']) in forbidden_en:
                    continue
                if min(len(pair['ar'].split()), len(pair['en'].split())) < 3:
                    continue
                if len(tokenizer(pair['ar'])['input_ids']) > self.cfg['max_length']:
                    continue
                if len(tokenizer(text_target=pair['en'])['input_ids']) > self.cfg['max_length']:
                    continue
                selected.append({'ar': pair['ar'], 'en': pair['en'], 'id': f'opus100-{split}-{i}'})
                forbidden_ar.add(text_hash(pair['ar']))
                forbidden_en.add(text_hash(pair['en']))
                if len(selected) == self.cfg[key]:
                    break
            if len(selected) != self.cfg[key]:
                raise RuntimeError(f'{key}: not enough eligible examples ({len(selected)})')
            data[key] = selected
        assert_disjoint(data)
        write_json(self.root / 'document_partitions.json', {k: sorted({r['document'] for r in v}) for k, v in partitions.items()})
        write_json(self.root / 'data_provenance.json', {
            'tico_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'tico_rows_after_cleaning': len(tico), 'generic': 'Helsinki-NLP/opus-100 ar-en',
            'note': 'Custom document-disjoint TICO partitions, not official benchmark or paper splits.'})
        write_json(cache, data)
        self.data = data
        for name, rows in data.items():
            pd.DataFrame(rows).to_csv(self.root / f'{name}.csv', index=False)
        return self.data_summary()

    def data_summary(self):
        return pd.DataFrame([{'partition': name, 'pairs': len(rows)} for name, rows in self.data.items()])

    def generate_targets(self):
        cache = self.root / 'generated_targets.json'
        if cache.exists():
            self.targets = json.loads(cache.read_text())
            return pd.DataFrame(self.targets)
        import langid
        tokenizer = AutoTokenizer.from_pretrained(self.cfg['lm_model'], padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        lm = AutoModelForCausalLM.from_pretrained(self.cfg['lm_model'], torch_dtype=torch.float16).to(self.device).eval()
        self._record_model(self.cfg['lm_model'], lm)
        prompts = self._subset(self.data['domain_train'], self.cfg['prompt_count'])
        blocked = [r['en'] for name in ('domain_train', 'domain_dev', 'domain_test', 'generic_dev', 'generic_test')
                   for r in self.data[name]]
        blocked_hashes = {text_hash(t) for t in blocked}
        seen, targets, raw_records = set(), [], []
        resume = self.root / 'generation_progress.json'
        if resume.exists():
            progress = json.loads(resume.read_text())
            raw_records = progress['raw']
            targets = progress['targets']
            seen = {text_hash(t['en']) for t in targets}
        completed = len(raw_records)
        for index, row in enumerate(prompts):
            if index < completed:
                continue
            set_seed(self.seed + index)
            inputs = tokenizer(row['en'].strip() + '\n', return_tensors='pt', truncation=True, max_length=128).to(self.device)
            with torch.inference_mode():
                outputs = lm.generate(**inputs, do_sample=True, top_k=50, top_p=0.95,
                                      max_new_tokens=self.cfg['lm_new_tokens'],
                                      num_return_sequences=self.cfg['sequences_per_prompt'],
                                      pad_token_id=tokenizer.eos_token_id)
            continuations = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            raw_records.append({'prompt_id': row['id'], 'prompt': row['en'], 'continuations': continuations})
            for text in continuations:
                for sentence in continuation_sentences(text):
                    key = text_hash(sentence)
                    if key in seen or key in blocked_hashes or langid.classify(sentence)[0] != 'en':
                        continue
                    if any(token_jaccard(sentence, other) >= 0.8 for other in blocked):
                        continue
                    seen.add(key)
                    targets.append({'en': sentence, 'prompt_id': row['id'], 'id': key[:20]})
            write_json(resume, {'raw': raw_records, 'targets': targets})
            if (index + 1) % 10 == 0:
                print(f'Prompts {index+1}/{len(prompts)}; accepted target sentences {len(targets)}', flush=True)
        del lm
        free_memory()
        write_json(cache, targets)
        self.targets = targets
        return pd.DataFrame(targets)

    def _translate(self, model, tokenizer, texts, beams=4):
        outputs = []
        model.eval()
        for start in range(0, len(texts), self.cfg['batch_size']):
            inputs = tokenizer(texts[start:start+self.cfg['batch_size']], return_tensors='pt',
                               padding=True, truncation=True, max_length=self.cfg['max_length']).to(self.device)
            with torch.inference_mode():
                predicted = model.generate(**inputs, num_beams=beams, do_sample=False,
                                           max_new_tokens=self.cfg['max_new_tokens'])
            outputs.extend(tokenizer.batch_decode(predicted, skip_special_tokens=True))
        return outputs

    def backtranslate_and_rank(self):
        cache = self.root / 'synthetic_pools.json'
        if cache.exists():
            self.pools = json.loads(cache.read_text())
            return pd.DataFrame([{'pool': k, 'pairs': len(v)} for k, v in self.pools.items()])
        if not self.targets:
            raise RuntimeError('No target sentences generated. Inspect generation_progress.json.')
        raw_cache = self.root / 'backtranslations.json'
        if raw_cache.exists():
            pairs = json.loads(raw_cache.read_text())
        else:
            tok = AutoTokenizer.from_pretrained(self.cfg['reverse_model'])
            model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg['reverse_model']).to(self.device)
            self._record_model(self.cfg['reverse_model'], model)
            ar = self._translate(model, tok, [r['en'] for r in self.targets], beams=5)
            pairs = [{**r, 'ar': text} for r, text in zip(self.targets, ar)]
            write_json(raw_cache, pairs)
            del model
            free_memory()
        pairs = clean_pairs(pairs)
        mt_tokenizer = AutoTokenizer.from_pretrained(self.cfg['mt_model'])
        heldout = [r for key in ('domain_dev', 'domain_test', 'generic_dev', 'generic_test') for r in self.data[key]]
        forbidden = {text_hash(r['ar']) for r in heldout}
        eligible = []
        for r in pairs:
            if text_hash(r['ar']) in forbidden:
                continue
            if any(token_jaccard(r['ar'], v['ar']) >= 0.8 for v in heldout):
                continue
            if len(mt_tokenizer(r['ar'])['input_ids']) > self.cfg['max_length']:
                continue
            if len(mt_tokenizer(text_target=r['en'])['input_ids']) > self.cfg['max_length']:
                continue
            eligible.append(r)
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(self.cfg['quality_model'], device=self.device)
        if not eligible:
            raise RuntimeError('No eligible back-translated pairs; inspect raw generation.')
        ar_vectors = encoder.encode([r['ar'] for r in eligible], normalize_embeddings=True, show_progress_bar=False)
        en_vectors = encoder.encode([r['en'] for r in eligible], normalize_embeddings=True, show_progress_bar=False)
        seed_vectors = encoder.encode([r['en'] for r in self.data['domain_train']], normalize_embeddings=True, show_progress_bar=False)
        centroid = seed_vectors.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        bilingual = (ar_vectors * en_vectors).sum(axis=1)
        relevance = en_vectors @ centroid
        for i, row in enumerate(eligible):
            row.update({'bilingual_similarity': float(bilingual[i]), 'domain_similarity': float(relevance[i]),
                        'quality_score': float(0.7*bilingual[i] + 0.3*relevance[i])})
        del encoder
        free_memory()
        k = min(self.cfg['synthetic_pairs'], math.floor(len(eligible)*self.cfg['quality_keep_fraction']))
        if k < self.cfg['minimum_synthetic_pairs']:
            raise RuntimeError(f'Only {len(eligible)} eligible pairs, matched experiment size {k}. '
                               'Increase prompt_count/sequences_per_prompt in a NEW run; no results fabricated.')
        random_pool = random.Random(self.seed).sample(eligible, k)
        ranked_pool = sorted(eligible, key=lambda r: r['quality_score'], reverse=True)[:k]
        self.pools = {'synthetic_random': random_pool, 'synthetic_ranked': ranked_pool}
        write_json(cache, self.pools)
        pd.DataFrame(eligible).to_csv(self.root / 'quality_audit.csv', index=False)
        write_json(self.root / 'quality_protocol.json', {'eligible_pairs': len(eligible), 'matched_k': k,
                   'scoring': '0.7 cross-lingual cosine + 0.3 cosine to training English centroid',
                   'note': 'Heuristic proxy, not a guarantee of semantic correctness. No dev/test reference used in ranking.'})
        return pd.DataFrame([{'pool': key, 'pairs': len(value)} for key, value in self.pools.items()])

    def _tokenized(self, rows, tokenizer):
        dataset = Dataset.from_list([{'ar': r['ar'], 'en': r['en']} for r in rows])
        def encode(batch):
            inputs = tokenizer(batch['ar'], max_length=self.cfg['max_length'], truncation=True)
            inputs['labels'] = tokenizer(text_target=batch['en'], max_length=self.cfg['max_length'], truncation=True)['input_ids']
            return inputs
        return dataset.map(encode, batched=True, remove_columns=dataset.column_names)

    def train_variant(self, name, domain_rows):
        directory = self.root / name
        marker = directory / 'complete.json'
        if marker.exists():
            return json.loads(marker.read_text())
        directory.mkdir(exist_ok=True)
        set_seed(self.seed)
        tokenizer = AutoTokenizer.from_pretrained(self.cfg['mt_model'])
        model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg['mt_model']).to(self.device)
        self._record_model(self.cfg['mt_model'], model)
        # A finite sampled schedule gives all adaptations identical update/example budgets.
        rng = random.Random(self.seed)
        draws = self.cfg['train_steps'] * self.cfg['batch_size'] * self.cfg['gradient_accumulation']
        mixed = []
        domain_draws = 0
        for _ in range(draws):
            use_domain = rng.random() < self.cfg['in_domain_probability']
            mixed.append(rng.choice(domain_rows if use_domain else self.data['generic_train']))
            domain_draws += int(use_domain)
        write_json(directory / 'sampling.json', {'examples': draws, 'domain_draws': domain_draws,
                   'domain_probability': self.cfg['in_domain_probability'], 'unique_domain_pool': len(domain_rows),
                   'unique_generic_pool': len(self.data['generic_train'])})
        training = self._tokenized(mixed, tokenizer)
        write_json(directory / 'token_budget.json', {
            'source_tokens_in_sampled_schedule': sum(len(x) for x in training['input_ids']),
            'target_tokens_in_sampled_schedule': sum(len(x) for x in training['labels']),
            'note': 'Same updates and examples, but variable-length text means token budgets can differ.'})
        development = self._tokenized(self.data['domain_dev'], tokenizer)
        args = Seq2SeqTrainingArguments(
            output_dir=str(directory / 'checkpoints'), max_steps=self.cfg['train_steps'],
            learning_rate=self.cfg['learning_rate'], per_device_train_batch_size=self.cfg['batch_size'],
            per_device_eval_batch_size=self.cfg['batch_size'], gradient_accumulation_steps=self.cfg['gradient_accumulation'],
            eval_strategy='steps', eval_steps=self.cfg['eval_steps'], save_strategy='steps', save_steps=self.cfg['eval_steps'],
            save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
            predict_with_generate=False, logging_steps=25, report_to='none', seed=self.seed, data_seed=self.seed,
            fp16=False, dataloader_num_workers=0, warmup_ratio=0.05, disable_tqdm=True)
        trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=training, eval_dataset=development,
                                data_collator=DataCollatorForSeq2Seq(tokenizer, model=model), processing_class=tokenizer)
        checkpoints = sorted((directory / 'checkpoints').glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
        torch.cuda.synchronize()
        trainer.save_model(str(directory / 'model'))
        tokenizer.save_pretrained(directory / 'model')
        pd.DataFrame(trainer.state.log_history).to_csv(directory / 'training_history.csv', index=False)
        summary = {'variant': name, 'best_checkpoint': trainer.state.best_model_checkpoint,
                   'best_dev_loss': trainer.state.best_metric, 'steps': trainer.state.global_step,
                   'this_session_seconds': time.perf_counter()-started, 'train_metrics': result.metrics}
        write_json(marker, summary)
        del trainer, model
        free_memory()
        return summary

    def train_all(self):
        variants = [('B_authentic_mix', self.data['domain_train']),
                    ('C_synthetic_random', self.pools['synthetic_random']),
                    ('D_synthetic_ranked', self.pools['synthetic_ranked'])]
        summaries = []
        for name, rows in variants:
            print('Training', name, flush=True)
            summaries.append(self.train_variant(name, rows))
        return pd.DataFrame(summaries)

    def average_checkpoint(self):
        path = self.root / 'E_ranked_average'
        if (path / 'complete.json').exists():
            return str(path / 'model')
        model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg['mt_model'])
        adapted = AutoModelForSeq2SeqLM.from_pretrained(self.root / 'D_synthetic_ranked' / 'model')
        left, right = dict(model.named_parameters()), dict(adapted.named_parameters())
        assert left.keys() == right.keys()
        with torch.no_grad():
            for name in left:
                assert left[name].shape == right[name].shape
                left[name].mul_(0.5).add_(right[name], alpha=0.5)
        path.mkdir(exist_ok=True)
        model.save_pretrained(path / 'model')
        AutoTokenizer.from_pretrained(self.cfg['mt_model']).save_pretrained(path / 'model')
        write_json(path / 'complete.json', {'alpha': 0.5, 'parents': ['A_baseline', 'D_synthetic_ranked'],
                                         'note': 'Fixed coefficient, no test-set tuning.'})
        del model, adapted
        free_memory()
        return str(path / 'model')

    def evaluate_all(self):
        models = {'A_baseline': self.cfg['mt_model']}
        models.update({name: str(self.root / name / 'model') for name in
                       ['B_authentic_mix', 'C_synthetic_random', 'D_synthetic_ranked', 'E_ranked_average']})
        rows = []
        for name, checkpoint in models.items():
            pending = [split for split in ('domain_dev', 'generic_dev', 'domain_test', 'generic_test')
                       if not (self.root / f'{name}_{split}_scores.json').exists()]
            if pending:
                model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(self.device).eval()
                tok = AutoTokenizer.from_pretrained(checkpoint)
                # Warm up with training text; no test outcome influences any configuration.
                self._translate(model, tok, [self.data['domain_train'][0]['ar']])
                for split in pending:
                    data = self.data[split]
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    predictions = self._translate(model, tok, [r['ar'] for r in data])
                    torch.cuda.synchronize()
                    seconds = time.perf_counter()-started
                    refs = [r['en'] for r in data]
                    bleu, chrf, ter = BLEU(), CHRF(word_order=2), TER()
                    score = {'model': name, 'split': split, 'n': len(data),
                             'BLEU': bleu.corpus_score(predictions, [refs]).score,
                             'chrF++': chrf.corpus_score(predictions, [refs]).score,
                             'TER': ter.corpus_score(predictions, [refs]).score, 'seconds': seconds,
                             'BLEU_signature': str(bleu.get_signature()), 'chrF_signature': str(chrf.get_signature())}
                    frame = pd.DataFrame({'id': [r['id'] for r in data], 'source_ar': [r['ar'] for r in data],
                                          'reference_en': refs, 'prediction_en': predictions})
                    frame.to_csv(self.root / f'{name}_{split}_predictions.csv', index=False)
                    write_json(self.root / f'{name}_{split}_scores.json', score)
                    print(name, split, score['BLEU'], score['chrF++'], flush=True)
                del model
                free_memory()
            for split in ('domain_dev', 'generic_dev', 'domain_test', 'generic_test'):
                rows.append(json.loads((self.root / f'{name}_{split}_scores.json').read_text()))
        results = pd.DataFrame(rows)
        results.to_csv(self.root / 'results.csv', index=False)
        return results

    def bootstrap_improvement(self, repeats=300):
        baseline = pd.read_csv(self.root / 'C_synthetic_random_domain_test_predictions.csv')
        improved = pd.read_csv(self.root / 'D_synthetic_ranked_domain_test_predictions.csv')
        assert baseline.id.tolist() == improved.id.tolist()
        docs = [r['document'] for r in self.data['domain_test']]
        groups = sorted(set(docs))
        rng = np.random.default_rng(self.seed)
        metric, differences = CHRF(word_order=2), []
        # Cluster bootstrap respects within-document correlation.
        for _ in range(repeats):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            ids = [i for group in sampled for i, doc in enumerate(docs) if doc == group]
            refs = baseline.reference_en.iloc[ids].tolist()
            c = metric.corpus_score(baseline.prediction_en.iloc[ids].tolist(), [refs]).score
            d = metric.corpus_score(improved.prediction_en.iloc[ids].tolist(), [refs]).score
            differences.append(d-c)
        interval = np.percentile(differences, [2.5, 97.5])
        summary = {'comparison': 'D minus C, domain test chrF++', 'repeats': repeats,
                   'document_clusters': len(groups), 'ci95_low': interval[0], 'ci95_high': interval[1],
                   'caution': 'Pilot cluster bootstrap, few documents and one training seed; not robust evidence across runs.'}
        write_json(self.root / 'bootstrap.json', summary)
        return summary
