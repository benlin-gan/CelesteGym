# coding: utf-8
import pandas as pd
import os
methods = ["funcprox", "funcproxwheels", "tabular", "random", "dqn"]
def load_method_data(method):
    data_files = [i for i in os.listdir(method) if ".json" in i]
    a = [pd.read_json(f"{method}/{n}", lines=True).set_index("episode") for n in data_files]
    b = [int(i.split('_')[0]) for i in data_files]
    u = pd.concat(a, keys=b).sort_index()
    u.index.names = ['seed', 'episode']
    return u
df = pd.concat([load_method_data(m) for m in methods], keys=methods)
df.index.names = ['method', 'seed', 'episode']

