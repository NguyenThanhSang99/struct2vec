from struct2vec import Struc2Vec
import networkx as nx

if __name__ == '__main__':
    graph = nx.read_edgelist('data/data.edgelist', create_using=nx.DiGraph(), nodetype=None,
                             data=[('weight', int)])
    walk_length = 10
    walk_numbers = 50
    workers = 4
    verbose = 40
    window_size = 5
    iter = 3

    model = Struc2Vec(graph, walk_length, walk_numbers,
                      workers, verbose, )
    model.train(window_size=window_size, iter=iter)
    embeddings = model.get_embeddings()
    print(embeddings)
