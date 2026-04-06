# Weekly ToC Digest (week of 2026-04-06)

This week's papers include advances in high-dimensional signal compression and applications of convolutional surrogates for tensor upscaling. Several also explore extensions of neural networks for PDE solutions. Prioritized significant methodological contributions linking classical signal analysis and modern deep learning approaches, along with novel applications to signal processing. Out of the provided papers, none directly align with the highly specific interests in harmonic analysis, wavelet theory, or deep learning combined with classical signal processing methods. Papers are ranked based on their potential relevance to broader interests in signal processing and deep learning. Selected papers for the researcher's interests in harmonic analysis, wavelet theory, and modern deep learning.

**Included:** 14 (score ≥ 0.35)  
**Scored:** 16 total items

---

## [CIPHER: Conformer-based Inference of Phonemes from High-density EEG](https://arxiv.org/abs/2604.02362)
*arXiv AI*  
Score: **0.95**  
Published: 2026-04-06T04:00:00+00:00
Tags: EEG, signal-processing, methods

This paper introduces a dual-pathway model using high-density EEG representations, integrating signal processing for EEG analysis. Relevant to neural signal processing and potentially sophisticated decomposition methods.

<details>
<summary>RSS summary</summary>

arXiv:2604.02362v1 Announce Type: cross Abstract: Decoding speech information from scalp EEG remains difficult due to low SNR and spatial blurring. We present CIPHER (Conformer-based Inference of Phonemes from High-density EEG Representations), a dual-pathway model using (i) ERP features and (ii) broadband DDA coefficients. On OpenNeuro ds006104 (24 participants, two studies with concurrent TMS), binary articulatory tasks reach near-ceiling performance but are highly confound-vulnerable (acousti…

</details>

---

## [A Spectral Framework for Multi-Scale Nonlinear Dimensionality Reduction](https://arxiv.org/abs/2604.02535)
*arXiv Machine Learning*  
Score: **0.90**  
Published: 2026-04-06T04:00:00+00:00
Tags: wavelets, MRA, sparse, theory

This paper presents a spectral framework for dimensionality reduction that balances global-local preservation, involving multi-scale analysis. It parallels multiresolution approaches in wavelet theory.

<details>
<summary>RSS summary</summary>

arXiv:2604.02535v1 Announce Type: new Abstract: Dimensionality reduction (DR) is characterized by two longstanding trade-offs. First, there is a global-local preservation tension: methods such as t-SNE and UMAP prioritize local neighborhood preservation, yet may distort global manifold structure, while methods such as Laplacian Eigenmaps preserve global geometry but often yield limited local separation. Second, there is a gap between expressiveness and analytical transparency: many nonlinear DR …

</details>

---

## [High-Dimensional Signal Compression: Lattice Point Bounds and Metric Entropy](https://arxiv.org/abs/2604.03178)
*arXiv Math*  
Score: **0.85**  
Published: 2026-04-06T04:00:00+00:00
Tags: signal-compression, sparse

This paper studies worst-case signal compression under energy constraints, involving counting lattice points in ellipsoids, which aligns with the interest in low-rank and sparse coding.

---

## [ROMAN: A Multiscale Routing Operator for Convolutional Time Series Models](https://arxiv.org/abs/2604.02577)
*arXiv Machine Learning*  
Score: **0.85**  
Published: 2026-04-06T04:00:00+00:00
Tags: time-frequency, MRA, CNN, filter-bank

ROMAN introduces a multiscale routing operator that reduces sequence length and enhances convolutional classifiers. The focus on multiscale analysis resonant with wavelet-based approaches is valuable.

<details>
<summary>RSS summary</summary>

arXiv:2604.02577v1 Announce Type: new Abstract: We introduce ROMAN (ROuting Multiscale representAtioN), a deterministic operator for time series that maps temporal scale and coarse temporal position into an explicit channel structure while reducing sequence length. ROMAN builds an anti-aliased multiscale pyramid, extracts fixed-length windows from each scale, and stacks them as pseudochannels, yielding a compact representation on which standard convolutional classifiers can operate. In this way,…

</details>

---

## [Fast NF4 Dequantization Kernels for Large Language Model Inference](https://arxiv.org/abs/2604.02556)
*arXiv Machine Learning*  
Score: **0.80**  
Published: 2026-04-06T04:00:00+00:00
Tags: sparse, methods, theory

The paper discusses optimization of quantization techniques, which are analogous to sparse coding in signal processing, for improving efficiency in neural networks.

<details>
<summary>RSS summary</summary>

arXiv:2604.02556v1 Announce Type: new Abstract: Large language models (LLMs) have grown beyond the memory capacity of single GPU devices, necessitating quantization techniques for practical deployment. While NF4 (4-bit NormalFloat) quantization enables 4$\times$ memory reduction, inference on current NVIDIA GPUs (e.g., Ampere A100) requires expensive dequantization back to FP16 format, creating a critical performance bottleneck. This paper presents a lightweight shared memory optimization that a…

</details>

---

## [Convolutional Surrogate for 3D Discrete Fracture-Matrix Tensor Upscaling](https://arxiv.org/abs/2604.02335)
*arXiv Machine Learning*  
Score: **0.75**  
Published: 2026-04-06T04:00:00+00:00
Tags: CNN, MRA, methods

Focus is on multiscale simulations using a Convolutional Neural Network framework, relevant to applications of CNNs in multiresolution signal processing.

<details>
<summary>RSS summary</summary>

arXiv:2604.02335v1 Announce Type: new Abstract: Modeling groundwater flow in three-dimensional fractured crystalline media requires accounting for strong spatial heterogeneity induced by fractures. Fine-scale discrete fracture-matrix (DFM) simulations can capture this complexity but are computationally expensive, especially when repeated evaluations are needed. To address this, we aim to employ a multilevel Monte Carlo (MLMC) framework in which numerical homogenization is used to upscale sub-res…

</details>

---

## [Jump Start or False Start? A Theoretical and Empirical Evaluation of LLM-initialized Bandits](https://arxiv.org/abs/2604.02527)
*arXiv Machine Learning*  
Score: **0.70**  
Published: 2026-04-06T04:00:00+00:00
Tags: methods, theory

Exploring LLM setups for bandits may involve methodologies similar to adaptive representations in supervised learning.

<details>
<summary>RSS summary</summary>

arXiv:2604.02527v1 Announce Type: new Abstract: The recent advancement of Large Language Models (LLMs) offers new opportunities to generate user preference data to warm-start bandits. Recent studies on contextual bandits with LLM initialization (CBLI) have shown that these synthetic priors can significantly lower early regret. However, these findings assume that LLM-generated choices are reasonably aligned with actual user preferences. In this paper, we systematically examine how LLM-generated p…

</details>

---

## [ECG Foundation Models and Medical LLMs for Agentic Cardiovascular Intelligence at the Edge: A Review and Outlook](https://arxiv.org/abs/2604.02501)
*arXiv Signal Processing*  
Score: **0.65**  
Published: 2026-04-06T04:00:00+00:00
Tags: ECG, foundation-models, signals

This review of foundation models for ECG signals highlights advanced learning architectures for signal processing, relevant to neural signal processing interests.

<details>
<summary>RSS summary</summary>

arXiv:2604.02501v1 Announce Type: new Abstract: Electrocardiogram (ECG) foundation models represent a paradigm shift from task-specific pipelines to generalizable architectures pre-trained on large-scale unlabeled waveform data. This survey presents a unified and deployment-aware review of foundation models and medical large language models (LLMs) for ECG intelligence in cardiovascular disease (CVD) diagnosis, monitoring, and clinical decision support. The central thesis of this survey paper is …

</details>

---

## [Learning Contractive Integral Operators with Fredholm Integral Neural Operators](https://arxiv.org/abs/2604.03034)
*arXiv Math*  
Score: **0.64**  
Published: 2026-04-06T04:00:00+00:00
Tags: neural-operators, theory

Introduces neural operators for FIEs, providing novel theoretical insights connecting integral operators with learnable neural architectures.

<details>
<summary>RSS summary</summary>

arXiv:2604.03034v1 Announce Type: new Abstract: We generalize the framework of Fredholm Neural Networks, to learn non-expansive integral operators arising in Fredholm Integral Equations (FIEs) of the second kind in arbitrary dimensions. We first present the proposed Fredholm Integral Neural Operators (FREDINOs), for FIEs and prove that they are universal approximators of linear and non-linear integral operators and corresponding solution operators. We furthermore prove that the learned operators…

</details>

---

## [Learning interacting particle systems from unlabeled data](https://arxiv.org/abs/2604.02581)
*arXiv Math*  
Score: **0.63**  
Published: 2026-04-06T04:00:00+00:00
Tags: machine-learning

Introduces a self-test loss function for learning from particle systems, touching on interests in novel learning techniques.

<details>
<summary>RSS summary</summary>

arXiv:2604.02581v1 Announce Type: cross Abstract: Learning the potentials of interacting particle systems is a fundamental task across various scientific disciplines. A major challenge is that unlabeled data collected at discrete time points lack trajectory information due to limitations in data collection methods or privacy constraints. We address this challenge by introducing a trajectory-free self-test loss function that leverages the weak-form stochastic evolution equation of the empirical d…

</details>

---

## [On the detection of medium inhomogeneity by contrast agent: wave scattering models and numerical implementations](https://arxiv.org/abs/2507.05773)
*arXiv Math*  
Score: **0.60**  
Published: 2026-04-06T04:00:00+00:00
Tags: wave-scattering, models

Engages wave scattering models, potentially connecting wavelet transformations to signal processing tasks in complex media.

<details>
<summary>RSS summary</summary>

arXiv:2507.05773v2 Announce Type: replace Abstract: We consider the wave scattering and inverse scattering in an inhomogeneous medium embedded a homogeneous droplet with a small size, which is modeled by a constant mass density and a small bulk modulus. Based on the Lippmann-Schwinger integral equation for scattering wave in inhomogeneous medium, we firstly develop an efficient approximate scheme for computing the scattered wave as well as its far-field pattern for any droplet located in the inh…

</details>

---

## [Convolutional Surrogate for 3D Discrete Fracture-Matrix Tensor Upscaling](https://arxiv.org/abs/2604.02335)
*arXiv Math*  
Score: **0.58**  
Published: 2026-04-06T04:00:00+00:00
Tags: convolutional, tensor-upscaling

Applies convolutional methods to tensor upscaling, relevant to interests in convolutional neural networks.

<details>
<summary>RSS summary</summary>

arXiv:2604.02335v1 Announce Type: cross Abstract: Modeling groundwater flow in three-dimensional fractured crystalline media requires accounting for strong spatial heterogeneity induced by fractures. Fine-scale discrete fracture-matrix (DFM) simulations can capture this complexity but are computationally expensive, especially when repeated evaluations are needed. To address this, we aim to employ a multilevel Monte Carlo (MLMC) framework in which numerical homogenization is used to upscale sub-r…

</details>

---

## [A hybrid high-order method for the biharmonic problem](https://arxiv.org/abs/2504.16608)
*arXiv Math*  
Score: **0.55**  
Published: 2026-04-06T04:00:00+00:00
Tags: high-order, methods

Proposes high-order discretizations, touching on multiscale and wavelet analysis connections.

<details>
<summary>RSS summary</summary>

arXiv:2504.16608v3 Announce Type: replace Abstract: This paper proposes a new hybrid high-order discretization for the biharmonic problem and the corresponding eigenvalue problem. The discrete ansatz space includes degrees of freedom in $n-2$ dimensional submanifolds (e.g., nodal values in 2D and edge values in 3D), in addition to the typical degrees of freedom in the mesh and on the hyperfaces in the HHO literature. This approach enables the characteristic commuting property of the hybrid high-…

</details>

---

## [Beyond Fixed Inference: Quantitative Flow Matching for Adaptive Image Denoising](https://arxiv.org/abs/2604.02392)
*arXiv Computer Vision*  
Score: **0.45**  
Published: 2026-04-06T04:00:00+00:00
Tags: methods

Applies flow-based generative models for image denoising, aligns with interest in adaptive signal processing.

<details>
<summary>RSS summary</summary>

arXiv:2604.02392v1 Announce Type: new Abstract: Diffusion and flow-based generative models have shown strong potential for image restoration. However, image denoising under unknown and varying noise conditions remains challenging, because the learned vector fields may become inconsistent across different noise levels, leading to degraded restoration quality under mismatch between training and inference. To address this issue, we propose a quantitative flow matching framework for adaptive image d…

</details>

---
