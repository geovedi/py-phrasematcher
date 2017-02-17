# py-phrasematcher

Simple, fast and resource-friendly Python phrase matcher by looking for known sequences.

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

Common usage.

```python
from phrasematcher import PhraseMatcher

matcher = PhraseMatcher('patterns.txt')

text = '''menurut analisa squawka , mu adalah satu di antara lima kesebelasan dengan kesalahan defensif terbesar di epl musim lalu -- walau hanya tiga gol yang masuk ke gawang mereka dari sejumlah kesalahan itu .'''

for match in matcher.match(text):
    print(match)
```

