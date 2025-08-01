# ADR-0001: Gaudi 3 Hardware Choice

## Status
Accepted

## Context
The project needs to select a hardware platform for large-scale machine learning training. The primary candidates are:

1. **Intel Gaudi 3**: Latest HPU with 2.7x better performance/dollar than H100
2. **NVIDIA H100**: Industry standard with mature ecosystem
3. **AMD MI300X**: Emerging alternative with competitive specs
4. **Google TPU v5**: Cloud-only solution with custom runtime

Key considerations:
- **Cost Efficiency**: Budget constraints require optimal performance per dollar
- **Performance**: Must handle large models (70B+ parameters) efficiently
- **Ecosystem Maturity**: Need stable software stack and community support
- **Availability**: Hardware must be accessible across multiple cloud providers
- **Future Roadmap**: Platform should have long-term viability

## Decision
We will standardize on **Intel Gaudi 3** as the primary hardware platform for the following reasons:

### Performance Advantages
- **2.7x better performance/dollar** compared to H100 in benchmark testing
- **Native BF16 support** with hardware-accelerated mixed precision
- **Matrix engine optimizations** for transformer workloads
- **High-bandwidth memory (HBM3)** with efficient memory subsystem

### Cost Benefits
- **Lower TCO**: Significant cost savings for large-scale training workloads
- **Energy efficiency**: Reduced power consumption and cooling requirements
- **Competitive pricing**: Better economics for extended training runs

### Technical Capabilities
- **Graph compiler optimizations**: Habana SynapseAI provides advanced optimization
- **Distributed training support**: Native multi-node scaling capabilities
- **PyTorch integration**: First-class support through habana-torch-plugin
- **Container ecosystem**: Production-ready Docker images and Kubernetes support

## Consequences

### Positive
- **Cost Reduction**: Estimated 60-70% reduction in training costs compared to H100
- **Performance Gains**: Higher throughput for large language models
- **Innovation Opportunity**: Early adoption of cutting-edge hardware
- **Differentiation**: Competitive advantage through cost-efficient training

### Negative
- **Ecosystem Maturity**: Smaller community compared to NVIDIA ecosystem
- **Tool Integration**: Some third-party tools may require additional integration work
- **Documentation**: Less extensive documentation compared to established platforms
- **Market Risk**: Intel HPU adoption uncertainty in long term

### Mitigation Strategies
- **Hybrid Approach**: Maintain compatibility with NVIDIA hardware for fallback
- **Community Engagement**: Active participation in Habana developer community
- **Documentation**: Comprehensive internal documentation and knowledge sharing
- **Monitoring**: Regular performance and cost monitoring to validate decision

## Alternatives Considered

### NVIDIA H100
- **Pros**: Mature ecosystem, extensive community, broad tool support
- **Cons**: 2.7x higher cost, limited availability, vendor lock-in concerns
- **Verdict**: Ruled out due to cost considerations

### AMD MI300X
- **Pros**: Competitive performance, ROCm ecosystem, price competitiveness
- **Cons**: Limited cloud availability, smaller ecosystem, early stage tooling
- **Verdict**: Considered for future evaluation but not ready for production

### Google TPU v5
- **Pros**: Excellent ML performance, integrated with Google Cloud
- **Cons**: Cloud vendor lock-in, limited flexibility, JAX/TensorFlow focus
- **Verdict**: Ruled out due to vendor lock-in and PyTorch preference

## Related Decisions
- [ADR-0002](./adr-0002-pytorch-lightning-framework.md): PyTorch Lightning Framework
- [ADR-0003](./adr-0003-terraform-infrastructure.md): Terraform for Infrastructure

## Review Date
This decision should be reviewed quarterly to assess:
- Hardware availability and pricing changes
- Ecosystem maturity progress
- Performance benchmark updates
- Alternative platform developments