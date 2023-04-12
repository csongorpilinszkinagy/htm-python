# Sparse HTM
Sparse matrix implementation of the HTM CLA algorithm in Python.

## The HTM network
Hierarchical Temporal Memory (HTM) is a type of neural network that is modeled after the structure and function of the human neocortex. The neocortex is the part of the brain responsible for many of our higher cognitive functions, such as perception, language, and reasoning. The HTM network is designed to perform tasks such as pattern recognition, anomaly detection, and prediction, among others. There are several advantages to using HTM networks over traditional neural networks. One of the main advantages is that HTM networks are highly scalable and can handle large and complex datasets with ease. Additionally, they are adaptive and can learn in real-time, making them ideal for applications that require continuous learning and adaptation. HTM networks are also robust to noise and can perform well even when the data is noisy or incomplete. Finally, HTM networks are highly interpretable, which means that they can provide insights into how they arrive at their decisions, making them useful for tasks where transparency and explainability are important.

## Layers of the network
* [SDR encoder](#sdr-encoder)
* [Spatial Pooler](#spatial-pooler)
* [Temporal Memory](#temporal-memory)
* [SDR decoder](#sdr-decoder)

### SDR encoder

Key features of an SDR:
* Sparse binary vector
* Similarity is retained by overlapping bits

Sparse Distributed Representation (SDR) encoder is a key component of the Hierarchical Temporal Memory (HTM) network. The SDR encoder is responsible for transforming the input data into a sparse binary representation, which is then processed by the HTM network. The SDR encoder works by mapping the input data onto a high-dimensional space, where each dimension corresponds to a feature or attribute of the input. The SDR encoder then identifies which dimensions are active, or "on," for a given input. By representing the input data in a sparse binary form, the SDR encoder can effectively capture the underlying patterns and relationships in the data, while minimizing redundancy and noise. This sparse binary representation is then passed on to the HTM network for further processing and analysis. The SDR encoder is a critical component of the HTM network, as it enables the network to handle large and complex datasets with high accuracy and efficiency.



