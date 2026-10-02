"""The train / test split of the training corpus (split v2), for selecting the test side.

Rule: per dataset 2 of every 100 clips go to test (deterministic, seed 0), sampled by *take group*, never by clip, so
no retarget, crop or window of a test take is in train; the dog datasets are held out entirely (the cross-species test);
the dev and benchmark clips are forced to test; duplicates and unverified overlaps are excluded. Datasets prefixed
'fc_' are the full-corpus versions; the unprefixed ones are earlier samples / windows mapped to their source take
('test' when that take is test, else 'superseded').

  python -m curvecodec.split [--side test] [--dataset fc_cmu]      # the clips of a side (dataset<TAB>clip), then a check
"""
import argparse
import gzip
import json
import os
import sys

SPLIT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'split_v2.json.gz')


class Split:
    def __init__(self, data):
        self.data = data
        self.sides = data['sides']; self.groups = data['groups']
        self.legacy_sides = data['legacy_sides']; self.legacy_groups = data['legacy_groups']
        self.test_groups = set(data['test_groups'])

    def side(self, dataset, clip):
        """'train' | 'test' | 'excluded:<why>' | 'superseded' | None (not in the split)."""
        if dataset in self.sides:
            return self.sides[dataset].get(clip)
        return self.legacy_sides.get(dataset, {}).get(clip)

    def group(self, dataset, clip):
        return self.groups.get(dataset, {}).get(clip) or self.legacy_groups.get(dataset, {}).get(clip)

    def clips(self, side='test', datasets=None):
        return sorted((ds, c) for ds, m in self.sides.items() if datasets is None or ds in datasets for c, x in m.items() if x == side)

    def check(self):
        """No take group on both sides, no dog clip in train, every forced clip's take is test -> counts."""
        dog = set(self.data['dog_datasets'])
        train_g, test_g = set(), set()
        for ds, m in self.sides.items():
            for c, x in m.items():
                g = self.groups[ds][c]
                if x == 'train':
                    assert ds not in dog, f'dog clip in train: {ds}/{c}'
                    train_g.add(g)
                elif x == 'test':
                    test_g.add(g)
        assert not train_g & (test_g | self.test_groups), 'a take group on both sides'
        for ds, m in self.legacy_sides.items():
            for c, x in m.items():
                assert x in ('test', 'superseded') and (x == 'test' or self.legacy_groups[ds][c] not in self.test_groups), (ds, c)
        for ds, c, _ in self.data['forced']:
            g = self.group(ds, c)
            assert g is None or g in self.test_groups or ds in dog, f'forced clip not test: {ds}/{c}'
        n = {k: sum(1 for m in self.sides.values() for x in m.values() if x.split(':')[0] == k) for k in ('train', 'test', 'excluded')}
        return dict(n, train_groups=len(train_g), test_groups=len(test_g))


def load():
    with gzip.open(SPLIT_FILE, 'rt') as f:
        return Split(json.load(f))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--side', default='test'); ap.add_argument('--dataset', action='append')
    a = ap.parse_args(argv)
    s = load()
    for ds, c in s.clips(a.side, a.dataset):
        print(f'{ds}\t{c}')
    print('check:', s.check(), file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
