use tokio_stream::StreamExt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    system_fingerprint: String,
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Choice {
    index: u32,
    delta: Delta,
    logprobs: Option<()>, // null in JSON
    finish_reason: Option<()>, // null in JSON
}

#[derive(Debug, Deserialize, Serialize)]
struct Delta {
    content: String,
}


pub async fn call_lm_studio_model(
    api_url: &str,
    api_key: &str,
    model_name: &str,
    prompt: &str,
    stream_response: bool,
) -> Result<(), Box<dyn std::error::Error>> {
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
        "stream": stream_response
    });

    // Print the request body
    println!("Request Body:");
    println!("{}", serde_json::to_string_pretty(&json_body)?);

    // Make the request
    let response = client
        .post(api_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&json_body)
        .send()
        .await?;

    if stream_response {
        // Handle streaming response
        handle_streaming_response(response).await?;
    } else {
        // Handle regular response
        let response_text = response.text().await?;
        println!("Response:\n{}", response_text);
    }

    Ok(())
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
                let stuff: ChatCompletionChunk = serde_json::from_str(s.trim_start_matches("data:").trim())?;
                print!("{}", stuff.choices[0].delta.content);
            } else {
                println!("{}", s);
            }
        }
    }

    Ok(())
}
