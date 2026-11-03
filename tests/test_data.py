import unittest
from domainmt.data import (assert_disjoint, clean_pairs, continuation_sentences, normalize,
                           split_documents, token_jaccard)


class DataTests(unittest.TestCase):
    def test_clean(self):
        pair = {'ar': 'هذا مثال عربي واضح', 'en': 'This is an English example.'}
        rows = [pair, pair.copy(), {'ar': ' ', 'en': 'hello'},
                {'ar': 'English', 'en': 'English'}, {'ar': None, 'en': 'x'}]
        self.assertEqual(clean_pairs(rows), [pair])

    def test_disjoint(self):
        with self.assertRaises(AssertionError):
            assert_disjoint({'a': [{'ar': 'نص', 'en': ' A sentence '}],
                             'b': [{'ar': 'آخر', 'en': 'a sentence'}]})

    def test_document_split(self):
        rows = [{'document': str(i), 'ar': f'نص {i}', 'en': f'English {i}'} for i in range(30)]
        first = split_documents(rows)
        self.assertEqual(first, split_documents(rows))
        self.assertEqual(sum(map(len, first.values())), 30)
        assert_disjoint(first)
        sets = [{r['document'] for r in rows} for rows in first.values()]
        self.assertTrue(sets[0].isdisjoint(sets[1]))

    def test_continuations(self):
        text = 'People should wash their hands with soap. This trailing fragment has no final stop'
        self.assertEqual(continuation_sentences(text), ['People should wash their hands with soap.'])

    def test_normalization(self):
        self.assertEqual(normalize('  Test\nCASE '), 'test case')
        self.assertEqual(token_jaccard('a b c', 'c b a'), 1)
        self.assertEqual(token_jaccard('', ''), 0)


if __name__ == '__main__':
    unittest.main()
