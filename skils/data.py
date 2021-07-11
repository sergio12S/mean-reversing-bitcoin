

df = [{
    "key1": 1,
    "key2": 2,
    "key3": 3
},
    {
        "key1": 2,
        "key2": 2,
}]

# AWS  S3 - data base

df1 = {
    "sdfsdf": 1,
    'key1': 2
}

df2 = {
    "key1": 10,
    "key2": 222,
    "key3": 30
}

df.append(df2)

# Manipulation data of dictionary
data = []
for i in df:
    if not i.get('key2'):
        print('Missed value')
    if i.get('key2'):
        data.append(i.get('key2'))
len(data)


# Check keys
data_keys_of_all_flles = []
for i in df:
    data_keys_of_all_flles.append(
        list(i.keys())
    )
print(data_keys_of_all_flles)

uniq_values = []
flat_list = []
for i in data_keys_of_all_flles:
    uniq_values.append(set(i))


# Intersection
data1 = [1, 2, 3, 334, 3434343]
data2 = [10, 20, 3, 334, 3434343]

uniq_values1 = set(data1)
uniq_values2 = set(data2)

uniq_values1.intersection(uniq_values2)

number = 2
[i + 2 for i in data1]

bigData = {
    'sdf': 1,
    "sdfsdf": 234,
    "sfak": 334,
    "dfsdfsf": 8833,
    "alfgdfg": 222
}

for i in bigData.keys():
    if i == 'sdf':
        print(bigData.get(i) + 25)

for i in bigData.values():
    print(i)

# Filter
list(filter(lambda i: i == 2, data1))

text = 'sfdsdf dsfsdf  sdfsdfsdf llsdla all'
for i in text:
    print(i)


list(filter(lambda i: i == 's', text))

for i in text:
    if i == 's':
        print(i)

# zip

for i, k in zip(data1, data2):
    print(i)
    print(k)

# iterate items
for i, k in bigData.items():
    if i == 'sdf':
        print(k + 20)
    # print('i', i)
    # print('k', k)

# List manipulation

df1 = [1, 2, 3, 4, 5]
df1[0]
df1[:1]
df1[1:]
df1[-1]
df1[-2]
df1[::-1]

# Home work
# 1. woking with map
# 2. Free Acount AWS, READ: s3, LAMBDA, STEP FUNCTION


# 3 Developing method
def dict_compare(d1, d2):
    d1_keys = set(i.lower() for i in d1.keys())
    d2_keys = set(i.lower() for i in d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    removed = d1_keys - d2_keys
    added = d2_keys - d1_keys
    modified = {o: (d1[o], d2[o]) for o in shared_keys if d1[o] != d2[o]}
    # same = set(o for o in shared_keys if d1[o] == d2[o])
    return added, removed, modified
