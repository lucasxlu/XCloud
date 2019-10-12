# Machine Learning in Production
## Serving ML/DL Models
In most of the use cases, ``RESTful API`` is always the preference of deploying machine learning models. In this section, I will describe some popular approaches.


### Clipper
[Clipper](http://clipper.ai/) is a low-latency prediction serving system for machine learning. Clipper makes it simple to integrate machine learning into user-facing serving systems.


### GraphPipe
[GraphPipe](https://github.com/oracle/graphpipe) is a protocol and collection of software designed to simplify machine learning model deployment and decouple it from framework-specific model implementations.


### NVIDIA TensorRT Inference Server
[TensorRT Inference Server (TRTIS)](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/index.html) provides a cloud inferencing solution optimized for NVIDIA GPUs. The server provides an inference service via an HTTP or gRPC endpoint, allowing remote clients to request inferencing for any model being managed by the server.


### Use Nginx Proxy to Enable Multi-services
See [README.md](README.md) __Upgrade Django Built-in Server__ section.
