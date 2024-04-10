# candle-deberta

### sentence embeddings 
```bash
cargo run --release -- --prompt "isn't the moon lovely"

> [[[ 0.1374,  0.2507,  0.0314, ..., -0.0353,  0.1919,  0.0098],
> [ 0.2572,  1.3495, -0.3776, ...,  0.5259, -0.2559,  0.1176],
> [ 0.4925,  0.2192, -0.4584, ..., -0.7313, -0.1503, -0.9733],
> ...
> [ 0.5388,  0.2678, -0.0243, ..., -0.3001, -0.4251,  0.8693],
> [-0.0410, -0.4019,  0.2414, ..., -0.1704, -0.4913,  1.0118],
> [ 0.1529,  0.2455,  0.0346, ..., -0.0607,  0.2182,  0.0370]]]
> Tensor[[1, 8, 768], f32]
```

#### ToDo ! 

- Optimize Attention block.

#### Things to Consider 

- Bottleneck in the Attention block.
- Sentence similarity scores are way off.
