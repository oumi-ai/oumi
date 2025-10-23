import json
import random

data = json.load(
    open("/home/shanghong/oumi/enterprise_experiments/data/open_thoughts_train.json")
)

random.shuffle(data)
print(data[0])

# create micro (0.3k), small (1k, 3k), medium (10k, 30k), large (100k), and xlarge (1.2M) datasets
micro_data = data[:300]
small_1k_data = data[:1000]
small_3k_data = data[:3000]
medium_10k_data = data[:10000]
medium_30k_data = data[:30000]
large_100k_data = data[:100000]
xlarge_data = data

# save to json
json.dump(
    micro_data,
    open(
        "/home/shanghong/oumi/enterprise_experiments/data/open_thoughts_micro_train.json",
        "w",
    ),
    indent=4,
)
json.dump(
    small_1k_data,
    open(
        "/home/shanghong/oumi/enterprise_experiments/data/open_thoughts_small_1k_train.json",
        "w",
    ),
    indent=4,
)
json.dump(
    small_3k_data,
    open(
        "/home/shanghong/oumi/enterprise_experiments/data/open_thoughts_small_3k_train.json",
        "w",
    ),
    indent=4,
)
json.dump(
    medium_10k_data,
    open(
        "/home/shanghong/oumi/enterprise_experiments/data/open_thoughts_medium_10k_train.json",
        "w",
    ),
    indent=4,
)
json.dump(
    medium_30k_data,
    open(
        "/home/shanghong/oumi/enterprise_experiments/data/open_thoughts_medium_30k_train.json",
        "w",
    ),
    indent=4,
)
json.dump(
    large_100k_data,
    open(
        "/home/shanghong/oumi/enterprise_experiments/data/open_thoughts_large_100k_train.json",
        "w",
    ),
    indent=4,
)
json.dump(
    xlarge_data,
    open(
        "/home/shanghong/oumi/enterprise_experiments/data/open_thoughts_xlarge_train.json",
        "w",
    ),
    indent=4,
)
