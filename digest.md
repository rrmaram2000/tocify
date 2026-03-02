# Weekly ToC Digest (week of 2026-03-02)

This week's highlights include novel neural operators for inverse scattering, optimization for multiscale PDEs, and interesting frameworks at the intersection of signal processing and deep learning. This week's selection includes strong connections between signal processing methods and modern machine learning techniques, particularly focusing on wavelets and time-series analysis. The top-ranked papers integrate novel multiresolution analysis and adaptive representation techniques. Relevant topics include novel signal processing techniques, wavelets, time-frequency analysis, signal processing in neural data, mathematical and computational frameworks linking to deep learning. Focus was placed on papers with methodological contributions related to wavelets, signal processing, and their connection to machine learning.

**Included:** 16 (score ≥ 0.35)  
**Scored:** 18 total items

---

## [A neural operator framework for solving inverse scattering problems](https://arxiv.org/abs/2602.24147)
*arXiv Math*  
Score: **0.95**  
Published: 2026-03-02T05:00:00+00:00
Tags: neural-operators, scattering, deep-learning, wavelets, signal-processing

Uses a neural operator combining deep learning with traditional inverse scattering approaches, important for signal processing and deep learning intersection.

<details>
<summary>RSS summary</summary>

arXiv:2602.24147v1 Announce Type: new Abstract: We present a neural operator framework for solving inverse scattering problems. A neural operator produces a preliminary indicator function for the scatterer, which, after appropriate rescaling, is used as a regularization parameter within the Linear Sampling Method to validate the initial reconstruction. The neural operator is implemented as a DeepONet with a fixed radial-basis-function trunk, while the noise level required for rescaling is estima…

</details>

---

## [Selective Denoising Diffusion Model for Time Series Anomaly Detection](https://arxiv.org/abs/2602.23662)
*arXiv Machine Learning*  
Score: **0.92**  
Published: 2026-03-02T05:00:00+00:00
Tags: time-series, methods, multiresolution, anomaly-detection

This paper introduces a diffusion model for time series anomaly detection, using generative approaches that align with modern signal processing, serving as an advanced multiresolution analysis tool for denoising and anomaly detection.

<details>
<summary>RSS summary</summary>

arXiv:2602.23662v1 Announce Type: new Abstract: Time series anomaly detection (TSAD) has been an important area of research for decades, with reconstruction-based methods, mostly based on generative models, gaining popularity and demonstrating success. Diffusion models have recently attracted attention due to their advanced generative capabilities. Existing diffusion-based methods for TSAD rely on a conditional strategy, which reconstructs input instances from white noise with the aid of the con…

</details>

---

## [Locally Subspace-Informed Neural Operators for Efficient Multiscale PDE Solving](https://arxiv.org/abs/2505.16030)
*arXiv Math*  
Score: **0.90**  
Published: 2026-03-02T05:00:00+00:00
Tags: neural-operators, multiscale, deep-learning, PDE

Utilizes neural operators with spectral basis functions for multiscale PDEs, linking traditional methods with modern machine learning.

<details>
<summary>RSS summary</summary>

arXiv:2505.16030v2 Announce Type: replace Abstract: Neural operators (NOs) struggle with high-contrast multiscale partial differential equations (PDEs), where fine-scale heterogeneities cause large errors. To address this, we use the Generalized Multiscale Finite Element Method (GMsFEM) that constructs localized spectral basis functions on coarse grids. This approach efficiently captures dominant multiscale features while solving heterogeneous PDEs accurately at reduced computational cost. Howev…

</details>

---

## [SDMixer: Sparse Dual-Mixer for Time Series Forecasting](https://arxiv.org/abs/2602.23581)
*arXiv Machine Learning*  
Score: **0.89**  
Published: 2026-03-02T05:00:00+00:00
Tags: time-series, sparse, frequency-domain, multiresolution

This paper proposes a model for time series forecasting, combining frequency and temporal domain analysis, which could be of interest in multiresolution and sparse coding techniques.

<details>
<summary>RSS summary</summary>

arXiv:2602.23581v1 Announce Type: new Abstract: Multivariate time series forecasting is widely applied in fields such as transportation, energy, and finance. However, the data commonly suffers from issues of multi-scale characteristics, weak correlations, and noise interference, which limit the predictive performance of existing models. This paper proposes a dual-stream sparse Mixer prediction framework that extracts global trends and local dynamic features from sequences in both the frequency a…

</details>

---

## [Brain-OF: An Omnifunctional Foundation Model for fMRI, EEG and MEG](https://arxiv.org/abs/2602.23410)
*arXiv Signal Processing*  
Score: **0.87**  
Published: 2026-03-02T05:00:00+00:00
Tags: EEG, MEG, fMRI, neural-networks, signal-analysis

Integrates neuroscientific modalities fMRI, EEG, and MEG, relevant due to its advanced signal processing methods for neural data.

<details>
<summary>RSS summary</summary>

arXiv:2602.23410v1 Announce Type: cross Abstract: Brain foundation models have achieved remarkable advances across a wide range of neuroscience tasks. However, most existing models are limited to a single functional modality, restricting their ability to exploit complementary spatiotemporal dynamics and the collective data scale across imaging techniques. To address this limitation, we propose Brain-OF, the first omnifunctional brain foundation model jointly pretrained on fMRI, EEG and MEG, capa…

</details>

---

## [Edge-based discretizations on triangulations in $\mathbb{R}^d$, with special attention to four-dimensional space](https://arxiv.org/abs/2412.02555)
*arXiv Math*  
Score: **0.85**  
Published: 2026-03-02T05:00:00+00:00
Tags: discretization, methods, space-time, wavelets

Focuses on space-time methods for computational challenges, touching on discretization and signal processing concepts.

<details>
<summary>RSS summary</summary>

arXiv:2412.02555v2 Announce Type: replace Abstract: Many time-dependent problems in the field of computational fluid dynamics can be solved using space-time methods. However, such methods can encounter issues with computational cost and robustness. In order to address these issues, efficient, node-centered edge-based schemes are currently being developed. In these schemes, a median-dual tessellation of the space-time domain is constructed based on an initial triangulation. These methods are node…

</details>

---

## [pathsig: A GPU-Accelerated Library for Truncated and Projected Path Signatures](https://arxiv.org/abs/2602.24066)
*arXiv Machine Learning*  
Score: **0.85**  
Published: 2026-03-02T05:00:00+00:00
Tags: methods, time-frequency, sparse

Introduces a PyTorch-native library for scalable, gradient-based learning in path signatures, linking sequential data representation to machine learning tasks, which could relate to signal processing.

<details>
<summary>RSS summary</summary>

arXiv:2602.24066v1 Announce Type: new Abstract: Path signatures provide a rich representation of sequential data, with strong theoretical guarantees and good performance in a variety of machine-learning tasks. While signatures have progressed from fixed feature extractors to trainable components of machine-learning models, existing libraries often lack the required scalability for large-scale, gradient-based learning. To address this gap, this paper introduces pathsig, a PyTorch-native library t…

</details>

---

## [ULW-SleepNet: An Ultra-Lightweight Network for Multimodal Sleep Stage Scoring](https://arxiv.org/abs/2602.23852)
*arXiv Signal Processing*  
Score: **0.82**  
Published: 2026-03-02T05:00:00+00:00
Tags: EEG, deep-learning, neural-networks, signal-processing

Relevance in lightweight neural network design for EEG analysis, illustrating novel deep learning applications in signal processing.

<details>
<summary>RSS summary</summary>

arXiv:2602.23852v1 Announce Type: cross Abstract: Automatic sleep stage scoring is crucial for the diagnosis and treatment of sleep disorders. Although deep learning models have advanced the field, many existing models are computationally demanding and designed for single-channel electroencephalography (EEG), limiting their practicality for multimodal polysomnography (PSG) data. To overcome this, we propose ULW-SleepNet, an ultra-lightweight multimodal sleep stage scoring framework that efficien…

</details>

---

## [Neural Operators Can Discover Functional Clusters](https://arxiv.org/abs/2602.23528)
*arXiv Machine Learning*  
Score: **0.82**  
Published: 2026-03-02T05:00:00+00:00
Tags: methods, theory, adaptive, functional-analysis

Focuses on operator learning for clustering in function spaces, relevant to adaptive and learnable basis functions in signal processing frameworks.

<details>
<summary>RSS summary</summary>

arXiv:2602.23528v1 Announce Type: new Abstract: Operator learning is reshaping scientific computing by amortizing inference across infinite families of problems. While neural operators (NOs) are increasingly well understood for regression, far less is known for classification and its unsupervised analogue: clustering. We prove that sample-based neural operators can learn any finite collection of classes in an infinite-dimensional reproducing kernel Hilbert space, even when the classes are neithe…

</details>

---

## [Hybrid Quantum Temporal Convolutional Networks](https://arxiv.org/abs/2602.23578)
*arXiv Machine Learning*  
Score: **0.80**  
Published: 2026-03-02T05:00:00+00:00
Tags: CNN, quantum-computing, methods

The hybridization of quantum computing with convolutional networks in sequential data processing is analogous to advanced signal transformation techniques, offering novel insights into structured neural network architectures.

---

## [WiLoc: Massive Measured Dataset of Wi-Fi Channel State Information with Application to Machine-Learning Based Localization](https://arxiv.org/abs/2602.09115)
*arXiv Signal Processing*  
Score: **0.78**  
Published: 2026-03-02T05:00:00+00:00
Tags: signal-processing, machine-learning, localization

This work leverages channel state information for machine learning localization, aligning with learnable representation concepts in signal processing and offering potential cross-domain applications.

<details>
<summary>RSS summary</summary>

arXiv:2602.09115v2 Announce Type: replace Abstract: Localization is a key component of the wireless ecosystem. Machine learning (ML)-based localization using channel state information (CSI) is one of the most popular methods for achieving high-accuracy localization with low cost. However, to be accurate and robust, ML-based algorithms need to be trained and tested with large amounts of data, covering not only many user equipment (UE)/target locations, but also many different access points (APs) …

</details>

---

## [Towards Source-Aware Object Swapping with Initial Noise Perturbation](https://arxiv.org/abs/2602.23697)
*arXiv Computer Vision*  
Score: **0.70**  
Published: 2026-03-02T05:00:00+00:00
Tags: methods, sparse

Focuses on noise perturbation methods which can relate to multiresolution and signal processing in visual data applications.

<details>
<summary>RSS summary</summary>

arXiv:2602.23697v1 Announce Type: new Abstract: Object swapping aims to replace a source object in a scene with a reference object while preserving object fidelity, scene fidelity, and object-scene harmony. Existing methods either require per-object finetuning and slow inference or rely on extra paired data that mostly depict the same object across contexts, forcing models to rely on background cues rather than learning cross-object alignment. We propose SourceSwap, a self-supervised and source-…

</details>

---

## [Foundation World Models for Agents that Learn, Verify, and Adapt Reliably Beyond Static Environments](https://arxiv.org/abs/2602.23997)
*arXiv Machine Learning*  
Score: **0.65**  
Published: 2026-03-02T05:00:00+00:00
Tags: methods, theory

Mentions compositional representations that could be relevant for adaptive signal processing and multiresolution analysis.

<details>
<summary>RSS summary</summary>

arXiv:2602.23997v1 Announce Type: new Abstract: The next generation of autonomous agents must not only learn efficiently but also act reliably and adapt their behavior in open worlds. Standard approaches typically assume fixed tasks and environments with little or no novelty, which limits world models' ability to support agents that must evolve their policies as conditions change. This paper outlines a vision for foundation world models: persistent, compositional representations that unify reinf…

</details>

---

## [LE-NeuS: Latency-Efficient Neuro-Symbolic Video Understanding via Adaptive Temporal Verification](https://arxiv.org/abs/2602.23553)
*arXiv Computer Vision*  
Score: **0.60**  
Published: 2026-03-02T05:00:00+00:00
Tags: time-frequency, methods

Incorporates neuro-symbolic methods and temporal verification, potentially relevant for time-series analysis in vision tasks.

<details>
<summary>RSS summary</summary>

arXiv:2602.23553v1 Announce Type: new Abstract: Neuro-symbolic approaches to long-form video question answering (LVQA) have demonstrated significant accuracy improvements by grounding temporal reasoning in formal verification. However, existing methods incur prohibitive latency overheads, up to 90x slower than base VLM prompting, rendering them impractical for latency-sensitive edge deployments. We present LE-NeuS, a latency-efficient neuro-symbolic framework that preserves the accuracy benefits…

</details>

---

## [Long Range Frequency Tuning for QML](https://arxiv.org/abs/2602.23409)
*arXiv AI*  
Score: **0.60**  
Published: 2026-03-02T05:00:00+00:00
Tags: QML, spectral, methods

The paper discusses quantum machine learning models employing Fourier series approximations, which relate to spectral methods and signal approximation.

<details>
<summary>RSS summary</summary>

arXiv:2602.23409v1 Announce Type: cross Abstract: Quantum machine learning models using angle encoding naturally represent truncated Fourier series, providing universal function approximation capabilities with sufficient circuit depth. For unary fixed-frequency encodings, circuit depth scales as O(omega_max * (omega_max + epsilon^{-2})) with target frequency magnitude omega_max and precision epsilon. Trainable-frequency approaches theoretically reduce this to match the target spectrum size, requ…

</details>

---

## [EgoGraph: Temporal Knowledge Graph for Egocentric Video Understanding](https://arxiv.org/abs/2602.23709)
*arXiv Computer Vision*  
Score: **0.55**  
Published: 2026-03-02T05:00:00+00:00
Tags: time-frequency, methods

Utilizes temporal modeling in video understanding, which ties back to signal decomposition and time-frequency analysis.

<details>
<summary>RSS summary</summary>

arXiv:2602.23709v1 Announce Type: new Abstract: Ultra-long egocentric videos spanning multiple days present significant challenges for video understanding. Existing approaches still rely on fragmented local processing and limited temporal modeling, restricting their ability to reason over such extended sequences. To address these limitations, we introduce EgoGraph, a training-free and dynamic knowledge-graph construction framework that explicitly encodes long-term, cross-entity dependencies in e…

</details>

---
