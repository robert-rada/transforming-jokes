import json
from tqdm import tqdm

with open('data/jokes.json') as f:
    data = json.load(f)

authors = {}

for sub in tqdm(data):
    author = sub['author']

    if author == 'None':
        continue

    if author not in authors:
        authors[author] = []

    authors[author].append(sub['score'])

sorted_auth = sorted(authors.items(), key=lambda item: len(item[1]), reverse=True)
sorted_auth = [(a, sorted(l, reverse=True)) for a, l in sorted_auth]

pct = []
for i in range(100):
    # print(sorted_auth[i])
    good = len([x for x in sorted_auth[i][1] if x > 85])
    bad = len([x for x in sorted_auth[i][1] if x <= 85])
    pct.append(good / (good + bad))

pct.sort()
print(pct[:10])
print(pct[90:])

data.sort(key=lambda sub: sub['score'], reverse=True)

o5 = 0
u5 = 0
for i in range(1000):
    if data[i]['author'] == 'None':
        continue
    if len(authors[data[i]['author']]) > 10:
        o5 += 1
    else:
        u5 += 1

print(o5, u5)

nsfw_good = 0
nsfw_nr = 0
norm_good = 0
norm_nr = 0
for sub in data:
    if sub['nsfw?']:
        nsfw_nr += 1
        if sub['score'] > 85:
            nsfw_good += 1
    else:
        norm_nr += 1
        if sub['score'] > 85:
            norm_good += 1

print(nsfw_good / nsfw_nr, nsfw_nr)
print(norm_good / norm_nr, norm_nr)
