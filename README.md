# Reading-CVPR2021
1) [Learning Position and Target Consistency for Memory-based Video Object Segmentation
](https://arxiv.org/abs/2104.04329): Matching-based methods do not consider any prior about the sequential order of the frames and how pixels of an object move together.
This paper addresses this problem by introducing 1) global retrieval module, 2) position guidance module, 3) object relation module. Global retrival mainly follows the architecture in STM. For position guidance module, additional local keys are extracted from the query embedding and the previous adjacent memory embedding.
This module adds positional encoding to both aforementioned embeddings, making the local keys position-sensitive. 
