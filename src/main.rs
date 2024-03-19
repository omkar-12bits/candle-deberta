mod debertaV2;

use anyhow::{Error as E,Ok,Result};

use candle_core::Tensor;
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;
use hf_hub::{api::sync::Api, Repo, RepoType};
use debertaV2::{Config,DebertaV2Model};

fn build_model(model:String)->Result<(tokenizers::Tokenizer, DebertaV2Model)>{

    let config_filename = Api::new()?
            .repo(Repo::new(
                model.clone(),
                RepoType::Model,
            ))
            .get("config.json")?;
    
    let model = Api::new()?
    .repo(Repo::new(
        model,
        RepoType::Model,
    ))
    .get("pytorch_model.bin")?;

    let config = std::fs::read_to_string(config_filename)?;
    let config: Config = serde_json::from_str(&config)?;
    
    let vb = VarBuilder::from_pth(model, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
    let model = debertaV2::DebertaV2Model::load(vb.clone(), &config)?;

    let mut tokenizer = Tokenizer::from_file("tokenizer.json").map_err(E::msg)?;

    let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
        direction: tokenizers::TruncationDirection::Right, 
        max_length: 512, 
        ..Default::default()
    }));

    Ok((tokenizer, model))
}



fn main() -> Result<()>{

    let model_name = "microsoft/deberta-v3-base".to_string();
    let (tokenizer , model) = build_model(model_name)?;

    let encoded = tokenizer.encode("the dog jumped over the cat", true).expect("failed encoding");
    let input_ids = encoded.get_ids().to_vec();
    let ids_length = input_ids.len();
    let input_ids = Tensor::from_vec(input_ids,(1,ids_length),&candle_core::Device::Cpu)?;
    let mask = Tensor::ones_like(&input_ids)?;
    let embeddings = model.forward(&input_ids, &mask)?;
    println!("embeddings {}",embeddings);

    Ok(())
}
