"""Pure-Python data validation and document-disjoint partitions."""
import csv
import hashlib
import io
import random
import re
import unicodedata
import zipfile


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def text_hash(text):
    return hashlib.sha256(normalize(text).encode('utf-8')).hexdigest()


def is_arabic(text):
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum('\u0600' <= c <= '\u06ff' for c in letters) / len(letters) >= 0.5


def clean_pairs(rows):
    result, seen = [], set()
    for row in rows:
        ar, en = row.get('ar'), row.get('en')
        if not isinstance(ar, str) or not isinstance(en, str):
            continue
        key = (normalize(ar), normalize(en))
        if not all(key) or key in seen or key[0] == key[1]:
            continue
        if not is_arabic(ar) or is_arabic(en):
            continue
        seen.add(key)
        result.append(dict(row))
    return result


def read_tico(path):
    rows = []
    with zipfile.ZipFile(path) as archive:
        for split in ('dev', 'test'):
            name = f'tico19-testset/{split}/{split}.en-ar.tsv'
            reader = csv.DictReader(io.StringIO(archive.read(name).decode('utf-8-sig')), delimiter='\t')
            for row in reader:
                if row['sourceLang'] != 'en' or row['targetLang'] != 'ar':
                    raise ValueError('Unexpected TICO direction')
                rows.append({'ar': row['targetString'], 'en': row['sourceString'],
                             'id': row['stringID'], 'document': row['stringID'].split(':')[0],
                             'license': row['license'], 'official_split': split})
    return clean_pairs(rows)


def split_documents(rows, seed=42):
    documents = sorted({r['document'] for r in rows})
    random.Random(seed).shuffle(documents)
    n = max(1, round(len(documents) * 0.2))
    groups = {'test': set(documents[:n]), 'dev': set(documents[n:2*n]), 'train': set(documents[2*n:])}
    splits = {name: [r for r in rows if r['document'] in docs] for name, docs in groups.items()}
    # Reserve evaluation strings first, removing cross-partition exact overlap.
    forbidden_ar, forbidden_en = set(), set()
    for name in ('test', 'dev', 'train'):
        splits[name] = [r for r in splits[name] if text_hash(r['ar']) not in forbidden_ar
                        and text_hash(r['en']) not in forbidden_en]
        forbidden_ar.update(text_hash(r['ar']) for r in splits[name])
        forbidden_en.update(text_hash(r['en']) for r in splits[name])
    return splits


def assert_disjoint(splits):
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            for language in ('ar', 'en'):
                a = {text_hash(r[language]) for r in splits[left]}
                b = {text_hash(r[language]) for r in splits[right]}
                if a & b:
                    raise AssertionError(f'{left}/{right}: overlapping {language} text')


def continuation_sentences(text):
    # Keep only complete punctuation-terminated sentences, not the trailing fragment.
    return [s.strip() for s in re.findall(r'[^.!?\n]+[.!?](?:["\u201d\u2019])?', text)
            if 6 <= len(s.split()) <= 80 and not re.search(r'https?://|<[^>]+>', s)]


def token_jaccard(a, b):
    a, b = set(normalize(a).split()), set(normalize(b).split())
    return len(a & b) / max(1, len(a | b))
