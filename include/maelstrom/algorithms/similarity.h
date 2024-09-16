#include "maelstrom/containers/vector.h"
#include <vector>

namespace maelstrom {

    enum similarity_t {
        COSINE = 0,
        JACCARD = 1
    };

    /*
        Computes the similarity between all vectors in src_embeddings pointed to by src_offsets
        and the vectors in target embeddings by taking the maximum similarity to each target
        embedding.
        If src_offsets is empty, then all vectors in src_embeddings are searched.
    */
    maelstrom::vector similarity(similarity_t metric, maelstrom::vector& src_embeddings, maelstrom::vector& src_offsets, maelstrom::vector& target_embeddings, size_t emb_stride);

}