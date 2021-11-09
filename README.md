# Paper Reading
## ICLR2021
1) [In Defense of Pseudo-Labeling: An Uncertainty-Aware Pseudo-label Selection Framework for Semi-Supervised Learning](https://openreview.net/forum?id=-ODN6SbiUU): This paper studies Semi-Suervised Learning (SSL). The paper suggests that consistency regularization, a popular approach in SSL has limitations such as requiring domain-specific data augmentation.Psudo-labeling on the other hand does not have these lmimitation but underperforms relative to consistency regularization. They suggest that this is due to high amount of noise in the pseudo labels which resulst from over-confident models. They make a connection between network calibration and uncertainty estimation and by including model uncertainty in the process of pseudo label selection, reduce the noise level and improve the overall performance. They experiment with multiple methods for uncertainty estimation and show that all this methos achieve similar results.
## ICCV2021
1) [COTR: Correspondence Transformer for Matching Across Images](https://github.com/ubc-vision/COTR)
2) [Warp Consistency for Unsupervised Learning of Dense Correspondences](https://arxiv.org/abs/2104.03308)
3) [Learning Target Candidate Association to Keep Track of What Not to Track](https://arxiv.org/abs/2103.16556)

## [CVPR2021](https://openaccess.thecvf.com/CVPR2021?day=2021-06-24)
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
* Task: Co-salient object detection targets at detecting common salient objects sharing the same attributes given a group of relevant images. <br/>
* Why:  Instead of only using images from the same group (similar things), teach the network dissimilar things using images from the other group. Therefore, the goal is to increase the intra-group compactness and the inter-group distinctiveness.<br/>
* How: The Group Affinity module brings the embeddings of the objects from the same category closer by computing a general group consensus from a group of images containing the same object (using correlation ops). The Group Collaborative Learning Network improves the inter-group separability by similar operations, only adding cross-group correlation. The consensus computed from this operation should not be able to detect the common object.<br/>
* Reference: [SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf)

4) [Inception Convolution with Efficient Dilation Search](https://arxiv.org/abs/2012.13587):
* Task: Any architecture using Conv layer. <br/>
* Why: To have an adaptive receptive field by searching optimal dilation rates across spatial and channel dimensions instead of using a fixed manual dilation.
* How: Using a search algorithm referred to as EDO (efficient dilation optimization). The statistical optimization minimizes the L1 error between the expectation of the output of the pre-trained weights (from the so-called supernet) and the expectation of the output from the sampled dilation weights. For more information about the role of the pre-trained weights refer to DARTS method. <br/>
Question: why should the dilation pattern give us the same expected value as the pre-trained supernet? Does this optimization happen together with the actual training of the backbone weights?<br/>

5) [Improving Multiple Object Tracking with Single Object Tracking](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Improving_Multiple_Object_Tracking_With_Single_Object_Tracking_CVPR_2021_paper.pdf): This paper proposes the SOTMOT architecture for multiple object tracking to bring the single object tracking advances to MOT setup!
The training pipeline consists of offline and online phases. 
During the offline training, the SOT branch (which is based on CenterNet) is trained via minimizing the ridge regression loss.
CenterNet produces the heatmap of the objects as well as an offset value for the object center and bounding box sizes.
In online inference, an association algorithm (DeepSORT) is used to find the optimal trajectory for each object. <br/>
* Additional references: [Learning Feature Embeddings for Discriminant Model based Tracking](https://arxiv.org/pdf/1906.10414.pdf), [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)

6) [Spatial Feature Calibration and Temporal Fusion for Effective One-stage Video Instance Segmentation](https://arxiv.org/abs/2104.05606)
7) [Self-supervised Augmentation Consistency for Adapting Semantic Segmentation](https://arxiv.org/abs/2105.00097)
8) [Less is More: CLIPBERT for Video-and-Language Learning via Sparse Sampling](https://openaccess.thecvf.com/content/CVPR2021/papers/Lei_Less_Is_More_ClipBERT_for_Video-and-Language_Learning_via_Sparse_Sampling_CVPR_2021_paper.pdf)
9) [Adversarial Generation of Continuous Images](https://openaccess.thecvf.com/content/CVPR2021/papers/Skorokhodov_Adversarial_Generation_of_Continuous_Images_CVPR_2021_paper.pdf)
10) [Region-aware Adaptive Instance Normalization for Image Harmonization](https://openaccess.thecvf.com/content/CVPR2021/papers/Ling_Region-Aware_Adaptive_Instance_Normalization_for_Image_Harmonization_CVPR_2021_paper.pdf)
11) [DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution](https://openaccess.thecvf.com/content/CVPR2021/papers/Qiao_DetectoRS_Detecting_Objects_With_Recursive_Feature_Pyramid_and_Switchable_Atrous_CVPR_2021_paper.pdf)
12) [Discriminative Appearance Modeling with Multi-track Pooling for Real-time Multi-object Tracking](https://openaccess.thecvf.com/content/CVPR2021/papers/Kim_Discriminative_Appearance_Modeling_With_Multi-Track_Pooling_for_Real-Time_Multi-Object_Tracking_CVPR_2021_paper.pdf)
13) [Multiple Object Tracking with Correlation Learning (https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Multiple_Object_Tracking_With_Correlation_Learning_CVPR_2021_paper.pdf)
14) [InverseForm: A Loss Function for Structured Boundary-Aware Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Borse_InverseForm_A_Loss_Function_for_Structured_Boundary-Aware_Segmentation_CVPR_2021_paper.pdf)
15) [Adaptive Consistency Regularization for Semi-Supervised Transfer Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Abuduweili_Adaptive_Consistency_Regularization_for_Semi-Supervised_Transfer_Learning_CVPR_2021_paper.pdf)
16) [Interactive Self-Training with Mean Teachers for Semi-supervised Object Detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Interactive_Self-Training_With_Mean_Teachers_for_Semi-Supervised_Object_Detection_CVPR_2021_paper.pdf)
17) [Convolutional Hough Matching Networks](https://openaccess.thecvf.com/content/CVPR2021/papers/Min_Convolutional_Hough_Matching_Networks_CVPR_2021_paper.pdf)
18) [Detector-Free Local Feature Matching with Transformers]()


## [Neurips2020](https://proceedings.neurips.cc/paper/2020)
1) [Dual-Resolution Correspondence Networks](https://arxiv.org/abs/2006.08844):
* Additional references: [Neighbourhood Consensus Networks](https://arxiv.org/abs/1810.10510)
2) [Partial Optimal Tranport with applications on Positive-Unlabeled Learning](https://proceedings.neurips.cc/paper/2020/hash/1e6e25d952a0d639b676ee20d0519ee2-Abstract.html)
3) [Adversarial Self-Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/1f1baa5b8edac74eb4eaa329f14a0361-Abstract.html)
4) [Normalizing Kalman Filters for Multivariate Time Series Analysis](https://proceedings.neurips.cc/paper/2020/hash/1f47cef5e38c952f94c5d61726027439-Abstract.html)
5) [Domain Generalization for Medical Imaging Classification with Linear-Dependency Regularization](https://proceedings.neurips.cc/paper/2020/hash/201d7288b4c18a679e48b31c72c30ded-Abstract.html)
6) [Removing Bias in Multi-modal Classifiers: Regularization by Maximizing Functional Entropies](https://proceedings.neurips.cc/paper/2020/hash/20d749bc05f47d2bd3026ce457dcfd8e-Abstract.html)
7) [Domain Generalization via Entropy Regularization](https://proceedings.neurips.cc/paper/2019/file/2974788b53f73e7950e8aa49f3a306db-Paper.pdf)
8) [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://bmild.github.io/fourfeat/)

## [ECCV2020](https://www.ecva.net/papers.php), [highlights](https://www.paperdigest.org/2020/08/eccv-2020-highlights/)
1) [Embedding Propagation: Smoother Manifold for Few-Shot Classification](https://arxiv.org/abs/2003.04151):
* Additional references: [Learning from labeled and unlabeled data with label propagation](http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf)
2) [ BorderDet: Border Feature for Dense Object Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2211_ECCV_2020_paper.php)
3) [Suppress and Balance: A Simple Gated Network for Salient Object Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2852_ECCV_2020_paper.php)
4) [Mining Cross-Image Semantics for Weakly Supervised Semantic Segmentation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3387_ECCV_2020_paper.php)
5) [Conditional Convolutions for Instance Segmentation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460273.pdf)
