// derive the jsons stuff

use serde::{Deserialize, Serialize};

use crate::target::target::TargetSettings;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Config {
    name: String,
    id: String,
    description: String,
    version: String,
    pipeline: Pipeline,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Pipeline {
    source: serde_json::Value,
    modules: serde_json::Value,
    target: TargetSettings,
}

impl Pipeline {
    pub fn new(json_file_path: &str) -> Result<Self, String> {
        let file = std::fs::File::open(json_file_path)
            .map_err(|e| format!("Unable to open file: {}", e))?;
        let reader = std::io::BufReader::new(file);
        let config: Config =
            serde_json::from_reader(reader).map_err(|e| format!("Unable to parse JSON: {}", e))?;

        Ok(Self {
            source: config.pipeline.source,
            modules: config.pipeline.modules,
            target: config.pipeline.target,
        })
    }

    pub fn run(&self) -> Result<(), String> {
        // Here you would implement the logic to run the pipeline
        // For example, you could call the source, modules, and target in order
        


        tracing::info!("Running pipeline with target: {:?}", self.target);

        Ok(())
    }
}
