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

    let mut tokenizer = Tokenizer::from_file("tokenizer/tokenizer.json").map_err(E::msg)?;

    let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
        direction: tokenizers::TruncationDirection::Right, 
        max_length: 512, 
        ..Default::default()
    }));

    Ok((tokenizer, model))
}



fn main() -> Result<()>{

    // use tracing_chrome::ChromeLayerBuilder;
    // use tracing_subscriber::prelude::*;

    // let _guard =  {
    //     println!("tracing...");
    //     let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
    //     tracing_subscriber::registry().with(chrome_layer).init();
    //     Some(guard)
    // };

    let model_name = "microsoft/deberta-v3-base".to_string();
    let (tokenizer , model) = build_model(model_name)?;

    let text = "Over the past several years, management of advanced melanoma has been transformed by the development and approval of novel therapeutic approaches. Genetically targeted therapies are now effective treatment options for the approximately 50% of patients whose melanomas harbor activating point mutations in BRAF. Combination regimens of small-molecule inhibitors have been developed to delay the onset of acquired resistance. Specifically, combined BRAF and MEK inhibition improves response rates and survival compared with single-agent BRAF inhibitors and has now received regulatory approval. During the same time frame, excitement has surrounded the development of immunotherapy with checkpoint inhibitors. New immune checkpoint inhibitors blocking cytotoxic T lymphocyte antigen-4 (CTLA4) or programmed death-1 receptor/ligand (PD-1/PD-L1) improve patient outcomes by promoting an antitumor immune response. These agents have been associated with an increasing number of durable responses and are being developed in various combinations. In this review, we discuss the development of these targeted and immune therapies, review current patient management, and highlight future directions. Therapeutic Advances and Treatment Options in Metastatic Melanoma.";
    println!("text \n{}",text);
    let encoded = tokenizer.encode(text, true).expect("failed encoding");
    let input_ids = encoded.get_ids().to_vec();
    let ids_length = input_ids.len();
    let input_ids = Tensor::from_vec(input_ids,(1,ids_length),&candle_core::Device::Cpu)?;
    let mask = Tensor::ones_like(&input_ids)?;
    let embeddings = model.forward(&input_ids, &mask)?;

    println!("embeddings {}",embeddings);

    
    Ok(())
}
