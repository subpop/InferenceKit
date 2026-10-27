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

/// Request body for `POST /v1/messages`.
struct AnthropicRequestBody: Encodable {
    var model: String
    var max_tokens: Int
    var system: [AnthropicTextBlock]?
    var messages: [AnthropicWireMessage]
    var stream: Bool
    var temperature: Double?
    var top_p: Double?
    var stop_sequences: [String]?
    var thinking: AnthropicThinking?
    var output_config: AnthropicOutputConfig?
    var tools: [AnthropicTool]?
    var context_management: AnthropicContextManagement?
}

struct AnthropicTextBlock: Encodable {
    let type = "text"
    var text: String
}

/// `thinking` parameter. `adaptive` (5.x-era adaptive thinking) lets Anthropic
/// decide; `enabled` with `budget_tokens` (4.5-era extended thinking) sets an
/// explicit budget.
struct AnthropicThinking: Encodable {
    var type: String
    var budget_tokens: Int?
}

/// `output_config` parameter; carries the effort that scales adaptive
/// thinking (Claude 4.7 onward).
struct AnthropicOutputConfig: Encodable {
    var effort: String
}

/// `context_management` parameter (context editing).
struct AnthropicContextManagement: Encodable {
    struct Edit: Encodable {
        struct Trigger: Encodable {
            let type = "input_tokens"
            var value: Int
        }
        let type = "clear_tool_uses_20250919"
        var trigger: Trigger
    }
    var edits: [Edit]
}

struct AnthropicTool: Encodable {
    var name: String
    var description: String
    var input_schema: JSONValue
}

/// One wire message. Content blocks are built as `JSONValue` because their
/// shapes vary by role (text, image, document, thinking, tool_use,
/// tool_result); the block builders live in `AnthropicMessagesFormat`.
struct AnthropicWireMessage: Encodable {
    var role: String
    var content: [JSONValue]
}

/// One SSE event from the Messages stream.
struct AnthropicStreamEvent: Decodable {
    let type: String
    let index: Int?
    let message: MessageSummary?
    let content_block: ContentBlockSummary?
    let delta: Delta?
    let usage: UsageSummary?
    let error: ErrorPayload?

    struct MessageSummary: Decodable {
        let id: String?
        let model: String?
        let usage: UsageSummary?
    }

    struct ContentBlockSummary: Decodable {
        let type: String?
        let id: String?
        let name: String?
    }

    struct Delta: Decodable {
        let type: String?
        let text: String?
        let thinking: String?
        let signature: String?
        let partial_json: String?
        let stop_reason: String?
    }

    struct UsageSummary: Decodable {
        let input_tokens: Int?
        let output_tokens: Int?
        let cache_read_input_tokens: Int?
        let cache_creation_input_tokens: Int?
    }

    struct ErrorPayload: Decodable {
        let type: String?
        let message: String?
    }
}

/// One page of `GET /v1/models`.
struct AnthropicModelsPage: Decodable {
    let data: [Item]
    let has_more: Bool?
    let last_id: String?

    struct Item: Decodable {
        let id: String
        let display_name: String?
    }
}
