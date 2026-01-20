import torch


def build_graph(df):
    nodes = {u:i for i,u in enumerate(df.user.unique())}
    edges = []

    for _, row in df.iterrows():
        edges.append([nodes[row.user], nodes[row.user]])

    return torch.tensor(edges).t(), len(nodes)
