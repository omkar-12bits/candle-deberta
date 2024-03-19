
use candle_transformers::models::with_tracing::{layer_norm, linear, LayerNorm, Linear};
use candle_core::{ Device, IndexOp, Module, Result, Tensor, D };
use candle_nn::{embedding, ops::softmax, Embedding, VarBuilder};

use serde::Deserialize;


#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    GeluApproximate,
    Relu,
}

struct HiddenActLayer {
    act: HiddenAct,
    span: tracing::Span,
}

impl HiddenActLayer {
    fn new(act: HiddenAct) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "hidden-act");
        Self { act, span }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        match self.act {
            // https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/activations.py#L213
            HiddenAct::Gelu => xs.gelu_erf(),
            HiddenAct::GeluApproximate => xs.gelu(),
            HiddenAct::Relu => xs.relu(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum PositionEmbeddingType {
    #[default]
    Absolute,
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/configuration_bert.py#L1
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    pub hidden_act: HiddenAct,
    hidden_dropout_prob: f64,
    attention_probs_dropout_prob: f64,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    initializer_range: f64,
    relative_attention: bool,
    max_relative_positions: i32,
    position_biased_input: bool,
    pos_att_type: Option<String>,
    position_buckets: usize,
    #[serde(default)]
    layer_norm_eps: f64,
    #[serde(default)]
    pad_token_id: usize,
    #[serde(default)]
    position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    use_cache: bool,
    classifier_dropout: Option<f64>,
    model_type: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 128100,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 0,
            initializer_range: 0.02,
            layer_norm_eps: 1e-7,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("deberta".to_string()),
            attention_probs_dropout_prob: 0.1,
            relative_attention: false,
            max_relative_positions: -1,
            position_biased_input: true,
            pos_att_type: None,
            position_buckets: 256,
        }
    }
}


struct Dropout {
    #[allow(dead_code)]
    pr: f64,
}

impl Dropout {
    fn new(pr: f64) -> Self {
        Self { pr }
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO
        Ok(x.clone())
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L180
pub struct DebertaV2Embeddings {
    word_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl DebertaV2Embeddings {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;

        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        Ok(Self {
            word_embeddings,
            layer_norm,
            dropout: Dropout::new(config.hidden_dropout_prob),
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let embeddings = self.word_embeddings.forward(input_ids)?;
        
        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings)?;
        Ok(embeddings)
    }
}

struct DisentangledSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
    span: tracing::Span,
    span_softmax: tracing::Span,
    pos_ebd_size : usize,
    position_buckets: usize,
    max_relative_positions: usize,
}

impl DisentangledSelfAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {

        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let hidden_size = config.hidden_size;

        let query = linear(hidden_size, all_head_size, vb.pp("query_proj"))?;
        let value = linear(hidden_size, all_head_size, vb.pp("value_proj"))?;
        let key = linear(hidden_size, all_head_size, vb.pp("key_proj"))?;

        let max_relative_positions = config.max_position_embeddings;
        let position_buckets = config.position_buckets;
        let pos_ebd_size = config.position_buckets;

        Ok(Self {
            query,
            key,
            value,
            dropout,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            pos_ebd_size,
            position_buckets,
            max_relative_positions,
            span: tracing::span!(tracing::Level::TRACE, "self-attn"),
            span_softmax: tracing::span!(tracing::Level::TRACE, "softmax"),
        })
    }

    
    fn make_log_bucket_position(&self,relative_pos:&Tensor, bucket_size: usize, max_position: usize)->Result<Tensor>{

    let sign  = relative_pos.to_dtype(candle_core::DType::I64)?.to_dtype(candle_core::DType::F32)?;
    let mid = (bucket_size /2)as f32;
    let (B,T) = relative_pos.dims2()?;
    let rlv_pos: Vec<f32> = relative_pos.flatten_all()?.to_vec1()?;
    let mut nw_pos = vec![];
    for elem in rlv_pos.iter(){
        if *elem > mid  && *elem < -mid {
            nw_pos.push(mid-1.0)
        }
        else {
            nw_pos.push(elem.abs());
        }
    }
    let abs_pos = Tensor::from_vec(nw_pos, (B,T), relative_pos.device())?;
    let log_pos = abs_pos.broadcast_sub(&Tensor::new(&[mid], relative_pos.device())?)?.log()?
        .broadcast_sub(&Tensor::new(&[(max_position as f32 - 1.0)/mid], relative_pos.device())?
        .mul(&Tensor::new(&[mid - 1.0], relative_pos.device())?)?)?.ceil()?.broadcast_add(&Tensor::new(&[mid], relative_pos.device())?)?;

    let i: Vec<f32> = abs_pos.flatten_all()?.to_vec1()?;
    let j: Vec<f32> = relative_pos.flatten_all()?.to_vec1()?;
    let k: Vec<f32> = log_pos.mul(&sign)?.flatten_all()?.to_vec1()?;

    let mut bucket_pos: Vec<f32> = vec![];
    for ((i_1,j_1),k_1) in i.iter().zip(j).zip(k){
        if *i_1 <= mid {
            bucket_pos.push(j_1)
        }
        else {
            bucket_pos.push(k_1)
        }
    }
    let bucket_pos = Tensor::from_vec(bucket_pos, (B,T), relative_pos.device())?;

    Ok(bucket_pos)
}

    fn build_relative_position(&self, query_size: usize, key_size: usize, bucket_size: usize, max_position: usize, device: &Device)->Result<Tensor>{

        let q_ids = Tensor::arange(0f32, query_size as f32, device)?;
        let k_ids = Tensor::arange(0f32, key_size as f32, device)?;
        let rel_pos_ids = q_ids.unsqueeze(1)?.broadcast_sub(&k_ids.unsqueeze(0)?)?;

        let rel_pos_ids = self.make_log_bucket_position(&rel_pos_ids, bucket_size, max_position)?;
    
        let rel_pos_ids = rel_pos_ids.to_dtype(candle_core::DType::I64)?.to_dtype(candle_core::DType::F32)?;
        let rel_pos_ids = rel_pos_ids.i((..query_size,..))?;
        let rel_pos_ids = rel_pos_ids.unsqueeze(0)?;

        Ok(rel_pos_ids)
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let (B,T,_) = xs.dims3()?;
        let xs = xs.reshape((B,T,self.num_attention_heads,()))?;
        let xs = xs.transpose(1, 2)?.reshape(((),xs.dim(1)? as usize, xs.dim(D::Minus1)? as usize))?;
        xs.contiguous()
    }

    fn x_softmax(&self, attention_scores: &Tensor, attention_mask: &Tensor) -> Result<Tensor>{

        let softmax_values = softmax(&attention_scores, D::Minus1)?;

        let masked_softmask_values = softmax_values.broadcast_mul(&attention_mask.to_dtype(candle_core::DType::F32)?)?;
        let normalized_softmask_values = masked_softmask_values.broadcast_div(&masked_softmask_values.sum(D::Minus1)?.unsqueeze(2)?)?;

        Ok(normalized_softmask_values)
    }

    fn disentangled_attention_bias(&self,query_layer:&Tensor, key_layer:&Tensor, rel_embeddings:&Tensor, scale_factor:&Tensor)->Result<Tensor>{

        let (_,q,_) = query_layer.dims3()?;
        let relative_pos = self.build_relative_position(q, key_layer.dim(D::Minus2)?,self.position_buckets, self.max_relative_positions, query_layer.device())?;

        let att_span = self.pos_ebd_size;
        let relative_pos = relative_pos.to_dtype(candle_core::DType::I64)?.to_device(query_layer.device())?.to_dtype(candle_core::DType::F32)?;

        let rel_embeddings = rel_embeddings.i((0..att_span*2,..))?.unsqueeze(0)?;
        let (Q,_,_) = query_layer.dims3()?;
        let pos_query_layer = self.transpose_for_scores(&self.query.forward(&rel_embeddings)?)?.repeat((Q/self.num_attention_heads,1,1))?;
        let pos_key_layer = self.transpose_for_scores(&self.key.forward(&rel_embeddings)?)?.repeat((Q/self.num_attention_heads,1,1))?;

        let (_,_,key_D) = pos_key_layer.dims3()?;
        let(_,_,query_D) = pos_query_layer.dims3()?;

        // content -> position 
        let scale = Tensor::new(&[key_D as f32], pos_key_layer.device())?.mul(scale_factor)?.sqrt()?;
        let c2p_att = query_layer.matmul(&pos_key_layer.transpose(D::Minus1, D::Minus2)?)?;
        let c2p_pos = relative_pos.broadcast_add(&Tensor::new(&[att_span as f32], relative_pos.device())?)?.clamp(0f32, (att_span * 2 - 1) as f32)?;
        let c2p_att = c2p_att.gather(&c2p_pos.squeeze(0)?.to_dtype(candle_core::DType::U32)?.expand((query_layer.dim(0)?,query_layer.dim(1)?, relative_pos.dim(D::Minus1)?))?.contiguous()?, D::Minus1)?;
        let score = c2p_att.broadcast_div(&scale)?;

        // position -> content 
        let scale = Tensor::new(&[pos_query_layer.dim(D::Minus1)? as f32], pos_query_layer.device())?.mul(scale_factor)?.sqrt()?;
        let r_pos ;
        if key_layer.dim(D::Minus2)? != query_layer.dim(D::Minus2)?{
            let t_pos = self.build_relative_position(key_layer.dim(D::Minus2)?,key_layer.dim(D::Minus2)?,self.position_buckets, self.max_relative_positions, query_layer.device())?;
            r_pos = t_pos.unsqueeze(0)?;
        }
        else {
            r_pos = relative_pos;
        }

        let p2c_pos = r_pos.neg()?.broadcast_add(&Tensor::new(&[att_span as f32], r_pos.device())?)?.clamp(0 as f32, (att_span*2-1)as f32)?;
        let p2c_att = key_layer.matmul(&pos_query_layer.transpose(D::Minus1, D::Minus2)?)?;
        let p2c_att = p2c_att.gather(&p2c_pos.squeeze(0)?.to_dtype(candle_core::DType::U32)?.expand((query_layer.dim(0)?, key_layer.dim(D::Minus2)?, key_layer.dim(D::Minus2)?))?.contiguous()?, D::Minus1)?.transpose(D::Minus1, D::Minus2)?;
        let score = score.add(&(p2c_att.broadcast_div(&scale)?))?;

        Ok(score)
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor, rel_embeddings:&Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        // scale factor = 1 (default) + 1 (c2p) + 1 (p2c)  check -> config.pos_att_type 
        let scale_factor = Tensor::new(&[3f32],hidden_states.device())?; 

        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        // let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        // let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;

        let q_l = query_layer.dim(D::Minus1)?;
        let scale = Tensor::new(&[q_l as f32 ], hidden_states.device())?.mul(&scale_factor)?.sqrt()?;

        let attention_scores = query_layer.matmul(&key_layer.transpose(D::Minus1, D::Minus2)?.broadcast_div(&scale)?)?;
        let rel_embeddings = self.disentangled_attention_bias(&query_layer, &key_layer,rel_embeddings, &scale_factor)?;
        let attention_scores = attention_scores.add(&rel_embeddings)?;
        let attention_scores = attention_scores.reshape(((),self.num_attention_heads, attention_scores.dim(D::Minus2)? as usize, attention_scores.dim(D::Minus1)? as usize))?;
        
        let attention_probs = {
            let _enter_sm = self.span_softmax.enter();
            self.x_softmax(&attention_scores, attention_mask)?
        };
        let attention_probs = self.dropout.forward(&attention_probs)?;
        
        let context_layer = attention_probs.reshape(((),attention_scores.dim(D::Minus2)? as usize,attention_scores.dim(D::Minus1)? as usize))?.matmul(&value_layer)?;
        let context_layer = context_layer
            .reshape(((),self.num_attention_heads,context_layer.dim(D::Minus2)?as usize,context_layer.dim(D::Minus1)?as usize))?
            .transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(D::Minus2)?;
        
        Ok(context_layer)
    }
}


struct DebertaV2SelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl DebertaV2SelfOutput {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
            span: tracing::span!(tracing::Level::TRACE, "self-out"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L392
struct DebertaV2Attention {
    self_attention: DisentangledSelfAttention,
    self_output: DebertaV2SelfOutput,
    span: tracing::Span,
}

impl DebertaV2Attention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let self_attention = DisentangledSelfAttention::load(vb.pp("self"), config)?;
        let self_output = DebertaV2SelfOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            self_attention,
            self_output,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
        })
    }
    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor, rel_embeddings: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let self_outputs = self.self_attention.forward(hidden_states, attention_mask, rel_embeddings)?;
        let attention_output = self.self_output.forward(&self_outputs, hidden_states)?;
        Ok(attention_output)
    }
}


// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L441
struct DebertaV2Intermediate {
    dense: Linear,
    intermediate_act: HiddenActLayer,
    span: tracing::Span,
}

impl DebertaV2Intermediate {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            intermediate_act: HiddenActLayer::new(config.hidden_act),
            span: tracing::span!(tracing::Level::TRACE, "inter"),
        })
    }
}

impl Module for DebertaV2Intermediate {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let ys = self.intermediate_act.forward(&hidden_states)?;
        Ok(ys)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L456
struct DebertaV2Output {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl DebertaV2Output {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
            span: tracing::span!(tracing::Level::TRACE, "out"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L470
struct DebertaV2Layer {
    attention: DebertaV2Attention,
    intermediate: DebertaV2Intermediate,
    output: DebertaV2Output,
    span: tracing::Span,
}

impl DebertaV2Layer {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention = DebertaV2Attention::load(vb.pp("attention"), config)?;
        let intermediate = DebertaV2Intermediate::load(vb.pp("intermediate"), config)?;
        let output = DebertaV2Output::load(vb.pp("output"), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor, rel_embeddings: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let attention_output = self.attention.forward(hidden_states, attention_mask, rel_embeddings)?;
        // TODO: Support cross-attention?
        // https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L523
        // TODO: Support something similar to `apply_chunking_to_forward`?
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;
        Ok(layer_output)
    }
}


// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L556
struct DebertaV2Encoder {
    layers: Vec<DebertaV2Layer>,
    span: tracing::Span,
    rel_embeddings: Embedding,
    layer_norm: LayerNorm
}

impl DebertaV2Encoder {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| DebertaV2Layer::load(vb.pp(&format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        let pos_ebd_size = config.position_buckets*2;
        let rel_embeddings = embedding(pos_ebd_size, config.hidden_size, vb.pp("rel_embeddings"))?;
        let layer_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("LayerNorm"))?;
        let span = tracing::span!(tracing::Level::TRACE, "encoder");
        Ok(DebertaV2Encoder { layers, span , rel_embeddings,layer_norm})
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.clone();
        let rel_embeddings = self.layer_norm.forward(self.rel_embeddings.embeddings())?;
        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, attention_mask, &rel_embeddings)?
        }
        Ok(hidden_states)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L874
pub struct DebertaV2Model {
    embeddings: DebertaV2Embeddings,
    encoder: DebertaV2Encoder,
    pub device: Device,
    span: tracing::Span,
}

impl DebertaV2Model {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let (embeddings, encoder) = match (
            DebertaV2Embeddings::load(vb.pp("deberta.embeddings"), config),
            DebertaV2Encoder::load(vb.pp("deberta.encoder"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                if let Some(model_type) = &config.model_type {
                    if let (Ok(embeddings), Ok(encoder)) = (
                        DebertaV2Embeddings::load(vb.pp(&format!("{model_type}.embeddings")), config),
                        DebertaV2Encoder::load(vb.pp(&format!("{model_type}.encoder")), config),
                    ) {
                        (embeddings, encoder)
                    } else {
                        return Err(err);
                    }
                } else {
                    return Err(err);
                }
            }
        };
        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, input_ids: &Tensor,attention_mask:&Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let embedding_output = self.embeddings.forward(input_ids)?;
        let sequence_output = self.encoder.forward(&embedding_output, attention_mask)?;
        Ok(sequence_output)
    }
}

