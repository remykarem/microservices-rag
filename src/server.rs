use crate::client::llm_client::{ChatCompletionChunk, Delta};
use async_stream::stream;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::{
    Router,
    extract::Json,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use bytes::Bytes;
use futures_util::stream::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{convert::Infallible, io, time::Duration};
use std::{
    net::SocketAddr,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio_stream::StreamExt;

pub async fn server() -> Result<(), Box<dyn std::error::Error>> {
    let app = Router::new()
        .route("/v1/models", get(get_models))
        .route("/v1/chat/completions", post(post_chat_completions));

    let addr: SocketAddr = "0.0.0.0:8090".parse().unwrap();
    println!("Listening on http://{addr}");
    Ok(axum::serve(tokio::net::TcpListener::bind(addr).await.unwrap(), app).await?)
}

// -------- /v1/models --------

#[derive(Serialize)]
struct Model {
    id: String,
    object: String,
    created: i64,
    owned_by: String,
}

async fn get_models() -> impl IntoResponse {
    let now = unix_ts();
    let models = vec![
        Model {
            id: "qwen/qwen3-coder-30b".into(),
            object: "model".into(),
            created: now,
            owned_by: "you".into(),
        },
        Model {
            id: "pro-echo".into(),
            object: "model".into(),
            created: now,
            owned_by: "you".into(),
        },
    ];

    Json(json!({
        "object": "list",
        "data": models
    }))
}

// -------- /v1/chat/completions --------

#[derive(Deserialize, Debug)]
struct ChatRequest {
    model: Option<String>,
    stream: Option<bool>,
    // we accept but ignore the payload below
    messages: Option<serde_json::Value>,
    // … add other OpenAI fields if you want to parse them
}

async fn post_chat_completions(
    axum::extract::Json(req): axum::extract::Json<ChatRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    if req.stream != Some(true) {
        return Err((StatusCode::BAD_REQUEST, "set `stream: true`".into()));
    }

    let text = "This is a tiny Rust server streaming a static response, chunk by chunk, \
                using OpenAI’s chat.completions SSE format. ✅";
    let tokens: Vec<&str> = text.split_inclusive([' ', ',', '.', '✅']).collect();

    let model = req.model.unwrap_or_else(|| "mini-echo".to_string());
    let created = unix_ts();
    let id = format!("chatcmpl-{}", created);

    let stream = stream! {

        let client = reqwest::Client::new();

        // Prepare the request body
        let json_body = serde_json::json!({
            "model": "qwen/qwen3-coder-30b",
            "messages": [
                {
                    "role": "user",
                    "content": "hello"
                }
            ],
            "stream": true
        });

        // Make the request
        let response = client
            .post("http://localhost:1234/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", "lm-studio"))
            .header("Content-Type", "application/json")
            .json(&json_body)
            .send()
            .await
            .unwrap();
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
                        // print!("{}", delta.content);
                        // io::stdout().flush().unwrap();
                    }
                } else {
                    // println!("{}", s);
                }
            }
        }

        // // First chunk: include role + first content piece
        // if let Some(first) = tokens.first() {
        //     let chunk = serde_json::json!({
        //         "id": id,
        //         "object": "chat.completion.chunk",
        //         "created": created,
        //         "model": model,
        //         "choices": [{
        //             "index": 0,
        //             "delta": { "role": "assistant", "content": first },
        //             "finish_reason": null
        //         }]
        //     });
        //
        //     let event = Event::default().data(chunk.to_string());
        //     yield Ok(event);
        // }
        //
        // // Remaining content
        // for t in tokens.iter().skip(1) {
        //     let chunk = serde_json::json!({
        //         "id": id,
        //         "object": "chat.completion.chunk",
        //         "created": created,
        //         "model": model,
        //         "choices": [{
        //             "index": 0,
        //             "delta": { "content": t },
        //             "finish_reason": null
        //         }]
        //     });
        //     let event = Event::default().data(chunk.to_string());
        //     yield Ok(event);
        //     // Optional: visible pacing
        //     tokio::time::sleep(Duration::from_millis(50)).await;
        // }
        //
        // // Final chunk with finish_reason
        // let final_chunk = serde_json::json!({
        //     "id": id,
        //     "object": "chat.completion.chunk",
        //     "created": created,
        //     "model": model,
        //     "choices": [{
        //         "index": 0,
        //         "delta": {},
        //         "finish_reason": "stop"
        //     }]
        // });
        // yield Ok(Event::default().data(final_chunk.to_string()));

        // OpenAI sentinel
        yield Ok(Event::default().data("[DONE]"));
    };

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    ))
}
// Helpers

fn line(v: serde_json::Value) -> Bytes {
    let mut s = String::from("data: ");
    s.push_str(&v.to_string());
    s.push_str("\n\n");
    Bytes::from(s)
}

fn unix_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}
