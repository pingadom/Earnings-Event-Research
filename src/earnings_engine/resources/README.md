# Bundled lexical resources

`lm_lite.txt` is a compact, hand-checked subset of the Loughran-McDonald
Master Dictionary categories, bundled so the text features work out of the box
and in CI with no download.

**For research you intend to report, use the full dictionary.** It is
maintained by Tim Loughran and Bill McDonald at Notre Dame and is free for
academic use:

    https://sraf.nd.edu/loughranmcdonald-master-dictionary/

Download the master dictionary CSV and point the config at it:

```yaml
features:
  lm_dictionary_path: data/raw/Loughran-McDonald_MasterDictionary.csv
```

The loader reads the CSV's `Word`, `Negative`, `Positive`, `Uncertainty`,
`Litigious`, `Strong_Modal` and `Weak_Modal` columns (a non-zero value in a
category column means the word belongs to that category).

Why a finance-specific dictionary at all: general-purpose sentiment lexicons
misclassify financial language badly. "Liability", "cost", "depreciation" and
"tax" are all negative in a general lexicon and are simply accounting terms in
a filing. Loughran and McDonald (2011) showed roughly three quarters of the
Harvard-IV negative words in 10-Ks are of this type.
