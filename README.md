# Paper Reading
## CVPR2021
1) [Learning Position and Target Consistency for Memory-based Video Object Segmentation
](https://arxiv.org/abs/2104.04329): Matching-based methods do not consider any prior about the sequential order of the frames and how pixels of an object move together.
This paper addresses this problem by introducing 1) global retrieval module, 2) position guidance module, 3) object relation module. Global retrival mainly follows the architecture in STM. For position guidance module, additional local keys are extracted from the query embedding and the previous adjacent memory embedding.
This module adds positional encoding to both aforementioned embeddings, making the local keys position-sensitive. 
Finally, the object relation module brings the object-level information from the first frame to improve the target consistency.
This way, we specifically pay attention to the first frame, unlike previous methods that treat the first frame the same as others stored in the memory bank. <br/>
* Similar idea for temporal consistency, [Blazingly Fast Video Object Segmentation with Pixel-Wise Metric Learning
](https://arxiv.org/abs/1804.03131), [One-Shot Object Detection with Co-Attention and Co-Excitation](https://github.com/timy90022/One-Shot-Object-Detection). <br/>
* Why the position embedding is added only to the previous frame?
2) [Delving Deep into Many-to-many Attention for Few-shot Video Object Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Delving_Deep_Into_Many-to-Many_Attention_for_Few-Shot_Video_Object_Segmentation_CVPR_2021_paper.pdf): In few-shot VOS, a support set for multiple appearances of a target object is provided; given the query images containing an object instance from the same class, the model should segment objects of the same category.
Two main approaches are either computing a prototype feature vector from the support set and detecting the object in the query via comparison to the prototype or perform many-to-many attention between the support set and the query frames, which is computationally expensive.
This paper considers the latter and proposes a solution for reducing the exponential cost in many-to-many attention operation to linear without performance loss. 
* A limitation is the naive way of choosing the agent frame from the video (middle frame), which could be the subject of future work. <br/>

3) [Group Collaborative Learning for Co-Salient Object Detection](https://arxiv.org/abs/2104.01108): <br/>

4) [Inception Convolution with Efficient Dilation Search](https://arxiv.org/abs/2012.13587):
5) [Improving Multiple Object Tracking with Single Object Tracking](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Improving_Multiple_Object_Tracking_With_Single_Object_Tracking_CVPR_2021_paper.pdf): This paper proposes the SOTMOT architecture for multiple object tracking to bring the single object tracking advances to MOT setup!
The training pipeline consists of offline and online phases. 
During the online training, the SOT branch (which is based on CenterNet) is trained via minimizing the ridge regression loss.
CenterNet produces the heatmap of the objects as well as an offset value for the object center and bounding box sizes.
In online inference, an association algorithm (DeepSORT) is used to find the optimal trajectory for each object. <br/>
* Additional references: [Learning Feature Embeddings for Discriminant Model based Tracking](https://arxiv.org/pdf/1906.10414.pdf), [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)


## Neurips2020
[Dual-Resolution Correspondence Networks](https://arxiv.org/abs/2006.08844):
