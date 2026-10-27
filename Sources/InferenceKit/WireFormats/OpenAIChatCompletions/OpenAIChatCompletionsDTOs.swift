// Copyright 2026 Link Dupont
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation

/// Request body for `POST /v1/chat/completions`.
struct OpenAIChatCompletionsBody: Encodable {
    var model: String
    var messages: [JSONValue]
    var stream: Bool
    var stream_options: StreamOptions?
    var temperature: Double?
    var top_p: Double?
    var max_completion_tokens: Int?
    var stop: [String]?
    var tools: [JSONValue]?
    var reasoning_effort: String?

    struct StreamOptions: Encodable {
        let include_usage = true
    }
}

/// One SSE chunk from the Chat Completions stream.
struct OpenAIChatChunk: Decodable {
    let id: String?
    let model: String?
    let choices: [Choice]?
    let usage: UsageSummary?

    struct Choice: Decodable {
        let delta: Delta?
        let finish_reason: String?
    }

    struct Delta: Decodable {
        let role: String?
        let content: String?
        let reasoning_content: String?
        let reasoning: String?
        let tool_calls: [ToolCallDelta]?
    }

    struct ToolCallDelta: Decodable {
        let index: Int?
        let id: String?
        let type: String?
        let function: Function?

        struct Function: Decodable {
            let name: String?
            let arguments: String?
        }
    }

    struct UsageSummary: Decodable {
        let prompt_tokens: Int?
        let completion_tokens: Int?
        let completion_tokens_details: Details?

        struct Details: Decodable {
            let reasoning_tokens: Int?
        }
    }
}
