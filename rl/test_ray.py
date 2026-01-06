import ray

ray.init(
    num_cpus=4,
    include_dashboard=False,
)

@ray.remote
def f(x):
    return x + 1

print(ray.get(f.remote(1)))
