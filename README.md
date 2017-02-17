# py-phrasematcher

Fast and resource-friendly Python phrase matcher.

## Usage

It takes a plain pattern file as input.

```
sepak bola
pencetak gol terbanyak
sir bobby charlton
bobby charlton
musim lalu
musim ini
satu di antara
kesalahan defensif
kesalahan defensif terbesar
...
```

### Initial usage.

```python
from phrasematcher import PhraseMatcher

matcher = PhraseMatcher('pmdb', pattern_file='patterns.txt')

text = '''menurut analisa squawka , mu adalah satu di antara lima kesebelasan dengan kesalahan defensif terbesar di epl musim lalu -- walau hanya tiga gol yang masuk ke gawang mereka dari sejumlah kesalahan itu .'''

for match in matcher.match(text):
    print(match)
```

### Reusing database

```python
from phrasematcher import PhraseMatcher

matcher = PhraseMatcher('pmdb')

```

## Why?

Short answer: I'm bored.

Long answer: Doing n-gram lookups is a waste of time and resources. Here we rejecting candidates that has OOV, only lookup first and last tokens and then check if the candidate pattern is in the hashtable.

In previous version, it requires 12GB, using normal Python dict, to store 150K vocab and 30M patterns. To reduce memory usage, we now use `vedis` to store dictionary on disk.


