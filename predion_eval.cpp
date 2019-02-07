#include <stdio.h>
#include <string>
#include <vector>
#include <valarray>
#include <unordered_map>
#include <unordered_set>

typedef long long int HASH_TYPE;

inline HASH_TYPE hashTuple(const int &a, const int &b) {
    // the number of entities should be less than 1048576 (2 ** 20)
    return (static_cast<HASH_TYPE>(a) << 20) | static_cast<HASH_TYPE>(b);
}

inline double score(const std::valarray<double> &head,
                    const std::valarray<double> &tail,
                    const std::valarray<double> &rela,
                    const std::valarray<double> &link) {
    return ((tail - head) * rela).sum() + ((head * tail) * link).sum();
}

std::vector<std::valarray<double>> readEmbedding(const char *fileName) {
    std::vector<std::valarray<double>> embedding;
    int count, dimension;
    FILE *fp = fopen(fileName, "r");
    if (fscanf(fp, "%d %d\n", &count, &dimension) != 2) {
        fprintf(stderr, "file format error: %s\n", fileName);
        return embedding;
    }
    embedding.assign(count, std::valarray<double>(0.0, dimension));
    for (int e = 0; e < count; ++e) {
        for (int i = 0; i < dimension; ++i) {
            fscanf(fp, "%lf", &embedding[e][i]);
        }
    }
    fclose(fp);
    return embedding;
}

/** args:
        path of the folder containing train.txt, valid.txt, test.txt
        file of node embeddings
        file of relation embeddings
        file of link embeddings
*/
int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "incorrect arguments\n");
        return 0;
    }

    std::unordered_map<int, std::unordered_set<HASH_TYPE>> exist;
    int h, t, r;
    for (const char *s: { "train.txt", "valid.txt", "test.txt" }) {
        FILE *fp = fopen((std::string(argv[1]) + s).data(), "r");
        fscanf(fp, "%*d %*d\n");  // consume first line
        while (fscanf(fp, "%d %d %d\n", &h, &t, &r) == 3)
            exist[r].emplace(hashTuple(h, t));
        fclose(fp);
    }
    std::vector<std::valarray<double>> nodeEmbedding = readEmbedding(argv[2]);
    std::vector<std::valarray<double>> relaEmbedding = readEmbedding(argv[3]);
    std::vector<std::valarray<double>> linkEmbedding = readEmbedding(argv[4]);
    const int&& nodeCount = static_cast<int>(nodeEmbedding.size());

    int testCount = 0;
    int sumRank = 0;
    int hits10 = 0;
    double mrr = 0.0;
    std::vector<int> rank;
    rank.reserve(10000);

    FILE *fp = fopen((std::string(argv[1]) + "test.txt").data(), "r");
    fscanf(fp, "%*d %*d\n");  // consume first line
    while (fscanf(fp, "%d %d %d\n", &h, &t, &r) == 3) {
        const std::unordered_set<HASH_TYPE> &currReExist = exist[r];
        const std::valarray<double> &headVector = nodeEmbedding[h];
        const std::valarray<double> &tailVector = nodeEmbedding[t];
        const std::valarray<double> &relaVector = relaEmbedding[r];
        const std::valarray<double> &linkVector = linkEmbedding[r];
        const double &&origin = score(headVector, tailVector,
                                      relaVector, linkVector);
        int headIndex = 1;
        int tailIndex = 1;
        #pragma omp parallel num_threads(4)
        {
            #pragma omp for reduction(+ : headIndex) schedule(guided, 128) nowait
            for (int hi = 0; hi < nodeCount; ++hi) {
                if (currReExist.count(hashTuple(hi, t)) == 0) {
                    if (origin <= score(nodeEmbedding[hi], tailVector,
                                        relaVector, linkVector))
                        ++headIndex;
                }
            }
            #pragma omp for reduction(+ : tailIndex) schedule(guided, 128)
            for (int ti = 0; ti < nodeCount; ++ti) {
                if (currReExist.count(hashTuple(h, ti)) == 0) {
                    if (origin <= score(headVector, nodeEmbedding[ti],
                                        relaVector, linkVector))
                        ++tailIndex;
                }
            }
        }
        rank.push_back(headIndex);
        rank.push_back(tailIndex);
        testCount += 2;
        sumRank += headIndex + tailIndex;
        hits10 += (headIndex <= 10 ? 1 : 0) + (tailIndex <= 10 ? 1 : 0);
        mrr += 1.0 / headIndex + 1.0 / tailIndex;
    }
    fclose(fp);

    mrr /= testCount;
    printf("MeanRank %d\nHitRate  %.4f\nMRR      %.4f\n",
           sumRank / testCount,
           static_cast<double>(hits10) / testCount,
           mrr);
    return 0;
}
