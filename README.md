# Reading-CVPR2021
1) [Learning Position and Target Consistency for Memory-based Video Object Segmentation
](https://arxiv.org/abs/2104.04329): Matching-based methods do not consider any prior about the sequential order of the frames and how pixels of an object move together.
This paper addresses this problem by introducing 1) global retrieval module, 2) position guidance module, 3) object relation module. Global retrival mainly follows the architecture in STM. For position guidance module, additional local keys are extracted from the query embedding and the previous adjacent memory embedding.
This module adds positional encoding to both aforementioned embeddings, making the local keys position-sensitive. 
Finally, the object relation module brings the object-level information from the first frame to improve the target consistency.
This way, we specifically pay attention to the first frame, unlike previous methods that treat the first frame the same as others stored in the memory bank. <br/>
Similar idea for temporal consistency, [Blazingly Fast Video Object Segmentation with Pixel-Wise Metric Learning
](https://arxiv.org/abs/1804.03131), [One-Shot Object Detection with Co-Attention and Co-Excitation](https://github.com/timy90022/One-Shot-Object-Detection). <br/>
Q: Why the position embedding is added only to the previous frame?
2) [Delving Deep into Many-to-many Attention for Few-shot Video Object Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Delving_Deep_Into_Many-to-Many_Attention_for_Few-Shot_Video_Object_Segmentation_CVPR_2021_paper.pdf)
