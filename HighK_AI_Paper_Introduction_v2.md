# Process-Conditioned Transfer Learning for High-κ Dielectric Materials Discovery: Coupling Atomistic Structure with Deposition and Annealing Parameters

## 1. Introduction

### 1.1 Background and Motivation

Continued scaling of complementary metal-oxide-semiconductor (CMOS) and
emerging logic and memory technologies has placed increasingly stringent
requirements on gate dielectric materials. Since the introduction of
high-κ/metal-gate stacks, the experimentally realized dielectric
response of candidate oxides has been governed by a coupled set of
factors that include composition, crystal structure and phase, dopant
chemistry, interfaces, and the fabrication conditions under which the
film is formed and subsequently treated. For HfO₂-family dielectrics in
particular, atomic layer deposition (ALD) conditions and post-deposition
annealing (PDA) can influence phase formation, microstructure,
interfacial evolution, and consequently measured dielectric behavior.
Thus, identifying a material with favorable intrinsic properties is only
one component of the materials-discovery problem; the property
ultimately realized in a fabricated film can also depend strongly on its
processing history.

Density functional theory (DFT), together with density functional
perturbation theory (DFPT), has long provided a first-principles basis
for evaluating structural, thermodynamic, electronic, and dielectric
properties of candidate materials. Large computational repositories such
as the Materials Project and JARVIS-DFT have substantially expanded the
accessible materials space and enabled high-throughput screening across
broad compositional and structural domains. However, direct
first-principles evaluation of dielectric properties remains
computationally demanding when extended to large candidate spaces,
motivating machine-learned surrogate models capable of substantially
faster property inference.

A second and conceptually distinct limitation arises when predictions
derived from computational materials databases are compared with
experimentally measured thin-film properties. Computational databases
predominantly characterize intrinsic material behavior for specified
atomic structures under defined computational conditions, whereas
semiconductor experiments observe materials after fabrication and
thermal processing. Consequently, the measured response can contain both
an intrinsic structure-dependent component and process-associated
variation. Although this process dependence is well established
experimentally, it remains weakly represented in large-scale
computational materials-discovery pipelines, which predominantly learn
intrinsic structure--property relationships from computational
databases.

### 1.2 Machine Learning for Materials Property Prediction and Discovery

Machine learning has increasingly been used to bridge the computational
cost of first-principles screening and the scale of modern materials
databases. Graph neural networks (GNNs) are particularly well suited to
crystalline materials because atomic structures can be represented as
graphs containing atomic and geometric information. The Atomistic Line
Graph Neural Network (ALIGNN), for example, incorporates both
interatomic bond and bond-angle information through atomistic graphs and
their corresponding line graphs and has demonstrated strong predictive
performance across a range of materials-property tasks. ALIGNN serves as
the atomistic representation-learning backbone adopted in this work.

In parallel with predictive surrogate models, generative materials
approaches have expanded the scope of machine-learning-assisted
discovery. Crystal generative frameworks such as CDVAE and related
diffusion-based approaches have demonstrated the generation of candidate
structures subject to structural or property constraints. More recent
large-scale efforts, including GNoME and MatterGen, have further
demonstrated the potential of learned models for broad exploration of
inorganic materials space, stability screening, and property-conditioned
candidate generation. These developments establish that transferable
representations learned from large computational datasets can
substantially accelerate exploration beyond conventional
first-principles screening alone.

Machine-learning models have also been developed specifically for
dielectric-property prediction using compositional descriptors, ensemble
methods, graph-based representations, and more recently equivariant
atomistic models. Such models can provide efficient surrogates for
expensive dielectric calculations and can be incorporated into
high-throughput screening or inverse-design workflows. However, these
approaches predominantly predict properties from composition and/or
atomic structure. The synthesis and fabrication conditions associated
with experimentally realized thin films are generally not represented
with the same fidelity as the intrinsic material structure.

Related efforts in synthesis-aware materials informatics and data-driven
process optimization have begun to incorporate experimental conditions,
synthesis parameters, or literature-derived processing information into
machine-learning workflows. These studies establish the importance of
process information but also expose a broader representation challenge:
transferring knowledge learned from large atomistic datasets into
substantially smaller, heterogeneous, process-specific experimental
datasets without discarding the underlying structural information.

### 1.3 From Structure-Only Prediction to Process-Conditioned Materials Learning

A large class of computational materials-learning pipelines implicitly
assumes that the target property can be represented primarily as a
function of composition and atomic structure. This assumption is
appropriate for learning computational labels evaluated for specified
structures, but it becomes less complete when the target is an
experimentally measured property of a processed thin film. In ALD-grown
high-κ dielectrics, nominally similar compositions can exhibit different
measured properties depending on deposition temperature, precursor and
oxidant chemistry, cycle conditions, dopant incorporation, annealing
temperature and ambient, crystal phase, and interface evolution.

This distinction creates what we refer to here as a **process-blindness
gap** between intrinsic computational materials prediction and
fabrication-conditioned experimental behavior. Structure-only surrogates
trained on computational databases may therefore exhibit an additional
source of prediction uncertainty when applied directly to experimentally
measured, process-dependent properties. Likewise, candidate materials
identified from intrinsic stability or property criteria may require
further assessment of whether the desired response can be realized
within practical fabrication conditions.

Addressing this gap is not equivalent to simply appending process
variables to an atomistic feature vector. Intrinsic material structure
and fabrication conditions represent related but physically distinct
information sources and differ substantially in data availability.
Atomistic databases can contain tens or hundreds of thousands of
structures, whereas carefully curated process-specific experimental
datasets may contain only hundreds of observations. Training the
complete material-property relationship directly from the latter risks
losing the transferable structural knowledge available from large
computational datasets.

A central design principle of the framework proposed here is therefore
to preserve material structure and fabrication conditions as **distinct
but observation-aligned representations**. For each experimental
observation (i), the material is represented by a crystal graph (G_i),
while selected fabrication and process-conditioning information is
transformed into a fixed-dimensional process representation (P_i).
Numerical and categorical process variables are encoded into a
reproducible feature space, and the material graph and process vector
are subsequently paired according to the same experimental observation.
This produces an aligned representation ((G_i, P_i)) while preserving
the semantic distinction between intrinsic structural information and
process-conditioning information.

The process representation is therefore not intended to replace the
atomistic representation or require the experimental dataset to relearn
intrinsic materials physics from scratch. Instead, it provides a
separate information pathway through which fabrication-associated
deviations from the structure-derived response can be learned. Such a
formulation provides a natural mechanism for transferring broad
intrinsic materials knowledge into a process-specific prediction problem
while retaining the ability to analyze the contribution of process
conditioning independently.

### 1.4 Proposed Framework and Contributions

In this work, we develop a hierarchical, process-conditioned
transfer-learning framework for high-κ dielectric materials that
connects large-scale atomistic learning with sparse experimental
fabrication data. The framework employs an ALIGNN-based atomistic
backbone and proceeds through staged learning: broad pretraining using
computational materials data, domain specialization toward
dielectric-relevant properties, and subsequent process-conditioned
learning using experimentally reported high-κ observations.

At the process-conditioned stage, the structure-derived representation
is coupled to the fixed-dimensional process representation through a
distinct model pathway. A bounded process-conditioned residual
formulation and lightweight structural adaptation are used to learn
experimentally observed deviations while limiting uncontrolled
modification of the transferred atomistic representation. The resulting
architecture is designed to retain knowledge acquired from substantially
larger computational datasets while allowing comparatively sparse
process observations to condition the final property prediction.

The principal contributions of this work are:

1.  **Process-conditioned material representation.** A representation
    framework that preserves crystal structure and fabrication
    conditions as separate but row-aligned inputs, pairing an atomistic
    material graph (G_i) with a fixed-dimensional process vector (P_i)
    corresponding to the same experimental observation.

2.  **Hierarchical transfer from intrinsic to process-conditioned
    prediction.** A staged learning architecture that transfers
    atomistic structure--property knowledge learned from large
    computational datasets into a dielectric-specific and subsequently
    fabrication-conditioned prediction problem, using a bounded residual
    pathway and lightweight structural adaptation rather than requiring
    sparse process data to relearn the complete material-property
    relationship.

3.  **Leakage-resistant evaluation methodology.** A material-disjoint
    train/validation/test evaluation protocol with locked holdout
    assessment, provenance tracking, and overlap controls designed to
    reduce information leakage and provide a more rigorous estimate of
    generalization for small, correlated experimental materials
    datasets.

4.  **Process-feature assessment with controlled validation.** A
    methodology for identifying useful process-conditioning variables
    while excluding metadata and leakage-adjacent information, combining
    feature-importance assessment, stability analysis, expert-reviewed
    feature inclusion, and controlled ablation to evaluate the
    predictive contribution of the retained process variables.

5.  **Reproducible internal and external validation.** A quantitative
    evaluation framework reporting prediction-error and
    dispersion-normalized metrics under locked evaluation, supplemented
    where applicable by independent external datasets to assess
    transferability beyond the data used for model development.

The central scientific question addressed by this study is therefore not
simply whether a graph neural network can predict dielectric properties,
nor whether process parameters alone can be fitted to a small
experimental dataset. Rather, we investigate whether **intrinsic
structure--property knowledge learned from large atomistic datasets can
be retained and transferred to sparse semiconductor process data while
separately representing process-associated variation in experimentally
realized properties**. High-κ dielectrics provide a particularly
relevant test case because their functional behavior is strongly coupled
to both intrinsic material characteristics and fabrication history. The
underlying formulation, however, is applicable more broadly to materials
systems in which experimentally observed properties emerge from the
interaction between atomic structure and processing conditions.

### 1.5 Paper Organization

The remainder of this paper is organized as follows. Section 2 describes
the hierarchical architecture and staged transfer-learning methodology,
including atomistic pretraining, dielectric-domain specialization,
process-conditioned representation, residual learning, and structural
adaptation. Section 3 describes the experimental process database,
data-quality controls, process representation generation, feature
assessment, and feature-selection methodology. Section 4 presents the
evaluation methodology and results, including material-disjoint
validation, locked holdout assessment, ablation studies, uncertainty
analysis, and external validation. Section 5 discusses the scientific
interpretation of the process-conditioned model, limitations associated
with sparse experimental data, observed failure modes, and implications
for generalization. Section 6 summarizes the principal findings and
discusses extensions toward process-aware materials screening, candidate
discovery, and fabrication-condition optimization.

------------------------------------------------------------------------

## Author note --- citation completion before submission

The manuscript version should attach formal references to claims
concerning high-κ/metal-gate technology, HfO₂ phase/process dependence,
DFT/DFPT dielectric calculations, Materials Project, JARVIS-DFT, ALIGNN,
CDVAE, GNoME, MatterGen, dielectric-property ML, synthesis-aware
materials informatics, and ALD/process optimization. Claims of novelty
around the process-blindness gap should be finalized only after a
focused literature review of process-conditioned and
synthesis-conditioned materials-learning work.
