mod deberta_v2;


use anyhow::{Error as E,Ok,Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use clap::Parser;
use tokenizers::{PaddingParams, Tokenizer};
use hf_hub::{api::sync::Api, Repo, RepoType};
use deberta_v2::{Config,DebertaV2Model};

#[derive(Parser, Debug)]
struct Args{

    #[arg(long, default_value = "true")]
    cpu: bool,

    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    prompt: Option<String>,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    n: usize,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "false")]
    normalize_embeddings: bool,

    /// Use tanh based approximation for Gelu instead of erf implementation.
    #[arg(long, default_value = "false")]
    approximate_gelu: bool,

    /// Use the pytorch weights rather than the safetensors ones
    #[arg(long, default_value="true")]
    use_pth: bool,
}

impl Args {

    fn build_model(&self)->Result<(tokenizers::Tokenizer, DebertaV2Model)>{
        
        let device = if self.cpu{
            // candle_core::Device::new_cuda(0)?
            candle_core::Device::Cpu // uncomment above line if cuda is supported ,add cuda support in features 
            } else {
            candle_core::Device::Cpu
        };
        let default_model = "microsoft/deberta-v3-base".to_string();
        let model_id = match self.model_id.to_owned() {
            Some(model_id) => model_id,
            None => default_model,
        };

        let config_filename = Api::new()?
            .repo(Repo::new(
                model_id.clone(),
                RepoType::Model,
            ))
        .get("config.json")?;
    
        let weights = if self.use_pth{ Api::new()?
            .repo(Repo::new(
                model_id,
                RepoType::Model,
            ))
        .get("pytorch_model.bin")?}
        else{
            Api::new()?
            .repo(Repo::new(
                model_id,
                RepoType::Model,
            ))
        .get("model.safetensors")?
        };
    
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        
        let vb = if self.use_pth {
            VarBuilder::from_pth(weights, candle_core::DType::F32, &device).unwrap()}
        else{
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights], candle_core::DType::F32, &device)? }
        };
        let model = deberta_v2::DebertaV2Model::load(vb.clone(), &config)?;
    
        let mut tokenizer = Tokenizer::from_file("tokenizer.json").map_err(E::msg)?;
        let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            direction: tokenizers::TruncationDirection::Right, 
            max_length: 512, 
            ..Default::default()
        }));
    
        Ok((tokenizer, model))
    }

}   



fn main() -> Result<()>{

    let args = Args::parse();

    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let _guard = if args.tracing {
        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    
    let start = std::time::Instant::now();
    let (mut tokenizer , model) = args.build_model()?;
    let device = &model.device;

    if let Some(prompt) = args.prompt {
        let tokenizer = tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)?;
        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.ones_like()?;
        println!("Loaded and encoded {:?}", start.elapsed());
        for idx in 0..args.n {
            let start = std::time::Instant::now();
            let ys = model.forward(&token_ids, &token_type_ids)?;
            if idx == 0 {
                println!("{ys}");
            }
            println!("Took {:?}", start.elapsed());
        }
    } else {
        let sentences = [
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
            "The new movie is awesome",
            "The cat plays in the garden",
            "A woman watches TV",
            "The new movie is so great",
            "Do you like pizza?",
        ];
        let n_sentences = sentences.len();
        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }
        let tokens = tokenizer
            .encode_batch(sentences.to_vec(), true)
            .map_err(E::msg)?;
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let token_type_ids = token_ids.ones_like()?;
        println!("running inference on batch {:?}", token_ids.shape());
        let embeddings = model.forward(&token_ids, &token_type_ids)?;
        println!("generated embeddings {:?}", embeddings.shape());
        // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = if args.normalize_embeddings {
            normalize_l2(&embeddings)?
        } else {
            embeddings
        };
        println!("pooled embeddings {:?}", embeddings.shape());

        let mut similarities = vec![];
        for i in 0..n_sentences {
            let e_i = embeddings.get(i)?;
            for j in (i + 1)..n_sentences {
                let e_j = embeddings.get(j)?;
                let sum_ij = (&e_i * &e_j)?.sum_all()?.to_scalar::<f32>()?;
                let sum_i2 = (&e_i * &e_i)?.sum_all()?.to_scalar::<f32>()?;
                let sum_j2 = (&e_j * &e_j)?.sum_all()?.to_scalar::<f32>()?;
                let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
                similarities.push((cosine_similarity, i, j))
            }
        }
        similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
        for &(score, i, j) in similarities[..5].iter() {
            println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
        }
    }

    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}