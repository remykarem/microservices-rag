use std::io;
use std::io::Write;
use tokio_stream::StreamExt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    system_fingerprint: String,
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Choice {
    index: u32,
    pub delta: Delta,
    logprobs: Option<()>,      // null in JSON
    finish_reason: Option<()>, // null in JSON
}

#[derive(Debug, Deserialize, Serialize)]
struct DeltaContent {
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Delta {
    String(DeltaContent),
    Object(Value),
}

pub async fn ask_llm(api_url: &str, api_key: &str, model_name: &str, prompt: &str) {
    let client = reqwest::Client::new();

    // Prepare the request body
    let json_body = serde_json::json!({
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": true
    });

    // Make the request
    let response = client
        .post(api_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&json_body)
        .send()
        .await
        .unwrap();

    handle_streaming_response(response).await.unwrap();
}

async fn handle_streaming_response(
    response: reqwest::Response,
) -> Result<(), Box<dyn std::error::Error>> {
    // Read the response as bytes
    let mut stream = response.bytes_stream();

    // Process the stream
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let Ok(s) = std::str::from_utf8(&chunk) {
            // Handle SSE format (data: prefix)
            if s.starts_with("data:") {
                let stuff: ChatCompletionChunk =
                    serde_json::from_str(s.trim_start_matches("data:").trim())?;
                if let Delta::String(delta) = &stuff.choices[0].delta {
                    print!("{}", delta.content);
                    io::stdout().flush().unwrap();
                }
            } else {
                println!("{}", s);
            }
        }
    }

    Ok(())
}
