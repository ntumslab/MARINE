import numpy as np
import torch
import torch.nn.functional
import torch.utils.data


class Analyzer(torch.utils.data.Dataset):

    def __init__(self, Q):
        super(Analyzer, self).__init__()
        self.Q = Q
        self.edgeList = []
        self.relaOvserved = {}
        self.nodeCount = 0
        self.relaCount = 0
        self.attribute = None
        self.attriDim = 0

    def readFiles(self, undirected, edgeFile, attriFile):
        with open(edgeFile) as f:
            nodeCount, relaCount = (int(x) for x in f.readline().split())

            for line in f:
                index_i, index_j, index_k = (int(x) for x in line.split())
                if index_i == index_j:
                    continue
                observedSet = self.relaOvserved.setdefault(index_k, set())
                observedSet.add((index_i, index_j))
                self.edgeList.append((index_i, index_j, index_k))
                if not undirected:
                    continue
                observedSet.add((index_j, index_i))
                self.edgeList.append((index_j, index_i, index_k))

        self.nodeCount = nodeCount
        self.relaCount = relaCount
        print("edges: {}".format(len(self.edgeList)))

        if not attriFile:
            return self

        with open(attriFile) as f:
            count, dimension = (int(x) for x in f.readline().split())
            assert count == nodeCount
            embedding = np.empty((nodeCount, dimension))

            for index, line in enumerate(f, start=0):
                embedding[index,:] = np.array(line.split(), dtype=float)

        self.attriDim = dimension
        self.attribute = torch.tensor(embedding, dtype=torch.float)

        return self

    def __len__(self):
        return len(self.edgeList) * self.Q

    def __getitem__(self, index):
        i, j, k = self.edgeList[index // self.Q]
        inputVector = [k, i, j]
        observedSet = self.relaOvserved[k]

        if np.random.random() < 0.5:  # corrupt tail
            while True:
                corrupt = np.random.randint(self.nodeCount)
                if corrupt != i and (i, corrupt) not in observedSet:
                    break
            inputVector.extend([i, corrupt])
        else:                         # corrupt head
            while True:
                corrupt = np.random.randint(self.nodeCount)
                if corrupt != j and (corrupt, j) not in observedSet:
                    break
            inputVector.extend([corrupt, j])

        return torch.tensor(inputVector, dtype=torch.int64, requires_grad=False)


class Marine(torch.nn.Module):

    def __init__(self, edgeFile, Q=5, dimension=128, undirected=False,
                 alpha=0.0, attriFile=None):
        super(Marine, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device: {}".format(self.device))

        self.analyzer = Analyzer(Q).readFiles(undirected, edgeFile, attriFile)
        self.nodeEmbedding = torch.nn.Embedding(self.analyzer.nodeCount, dimension)
        self.relaEmbedding = torch.nn.Embedding(self.analyzer.relaCount, dimension)
        self.linkEmbedding = torch.nn.Embedding(self.analyzer.relaCount, dimension)
        self.alpha = alpha
        if alpha:
            print("alpha: {} ({})".format(alpha, attriFile))
            self.attribute = torch.nn.Embedding.from_pretrained(
                self.analyzer.attribute, freeze=True)
            self.transform = torch.nn.Linear(dimension, self.analyzer.attriDim)

    def forward(self, batchVector):
        idx_k = batchVector[:,0]
        link_k = self.linkEmbedding(idx_k)
        rela_k = self.relaEmbedding(idx_k)
        idx_i = batchVector[:,1]
        idx_j = batchVector[:,2]
        pos_i = self.nodeEmbedding(idx_i)
        pos_j = self.nodeEmbedding(idx_j)
        neg_i = self.nodeEmbedding(batchVector[:,3])
        neg_j = self.nodeEmbedding(batchVector[:,4])

        # softplus(corrupt - correct)
        relaError = torch.sum((neg_j - neg_i - pos_j + pos_i) * rela_k, dim=1)
        linkError = torch.sum((neg_i * neg_j - pos_i * pos_j) * link_k, dim=1)
        loss = torch.nn.functional.softplus(relaError + linkError)

        if not self.alpha:
            return loss

        diff_i = self.transform(pos_i) - self.attribute(idx_i)
        diff_j = self.transform(pos_j) - self.attribute(idx_j)
        return loss + self.alpha * (torch.norm(diff_i, p=2, dim=1) +
                                    torch.norm(diff_j, p=2, dim=1))

    def train(self, epoches=500):
        self.to(device=self.device)
        optimizer = torch.optim.Adam(self.parameters())
        generator = torch.utils.data.DataLoader(
            self.analyzer, batch_size=1024, shuffle=True, num_workers=2)

        for epoch in range(1, epoches + 1):
            loss = 0.0
            for batchData in generator:
                optimizer.zero_grad()
                batchData = batchData.to(device=self.device)
                batchLoss = self(batchData).sum()
                loss += float(batchLoss)
                batchLoss.backward()
                optimizer.step()
            print("Epoch{:4d}/{}   Loss: {:e}".format(epoch, epoches, loss))

    @staticmethod
    def dumpTensor(tensor, filePath):
        matrix = tensor.weight
        size = matrix.size()
        with open(filePath, 'w') as f:
            print("{} {}".format(size[0], size[1]), file=f)
            for vec in matrix:
                print(' '.join(['{:e}'.format(x) for x in vec]), file=f)

    def saveWeights(self, path='.'):
        import os
        self.dumpTensor(self.nodeEmbedding, os.path.join(path, 'node.txt'))
        self.dumpTensor(self.relaEmbedding, os.path.join(path, 'rela.txt'))
        self.dumpTensor(self.linkEmbedding, os.path.join(path, 'link.txt'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Marine')
    parser.add_argument('edgeFile')
    parser.add_argument('-q', '--Q',
                        action='store', default=5, type=int,
                        help="negative samples per instance")
    parser.add_argument('-d', '--dimension',
                        action='store', default=128, type=int,
                        help="output embeddings' dimension")
    parser.add_argument('-u', '--undirected',
                        action='store_true',
                        help="whether the graph is undirected")
    parser.add_argument('-a', '--alpha',
                        action='store', default=0.0, type=float,
                        help="loss ratio of attributes")
    parser.add_argument('-A', '--attribute',
                        action='store', default=None, type=str,
                        help="attribute file (with a positive alpha)")
    parser.add_argument('-e', '--epoches',
                        action='store', default=500, type=int,
                        help="training epoches")
    parser.add_argument('-p', '--path',
                        action='store', default=".", type=str,
                        help="output path")

    args = parser.parse_args()
    if args.alpha <= 0.0 or not args.attribute:
        args.alpha = 0.0
        args.attribute = None

    module = Marine(edgeFile=args.edgeFile,
                    Q=args.Q,
                    dimension=args.dimension,
                    undirected=args.undirected,
                    alpha=args.alpha,
                    attriFile=args.attribute)
    module.train(epoches=args.epoches)
    module.saveWeights(path=args.path)

