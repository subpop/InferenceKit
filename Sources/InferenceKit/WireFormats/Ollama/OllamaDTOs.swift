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

/// Request body for `POST /api/chat`.
struct OllamaChatBody: Encodable {
    var model: String
    var messages: [OllamaWireMessage]
    var stream: Bool
    var think: OllamaThink?
    var tools: [JSONValue]?
    var options: Options?

    struct Options: Encodable {
        var num_ctx: Int?
        var temperature: Double?
        var top_p: Double?
        var stop: [String]?
    }
}

/// The `think` parameter: a boolean toggle or a budgeted object on newer
/// servers.
enum OllamaThink: Encodable {
    case enabled(Bool)
    case budget(Int)

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .enabled(let value):
            try container.encode(value)
        case .budget(let tokens):
            try container.encode(JSONValue.object([
                "type": .string("enable"),
                "budget": .int(Int64(tokens)),
            ]))
        }
    }
}

/// One wire message.
struct OllamaWireMessage: Encodable {
    var role: String
    var content: String?
    var images: [String]?
    var thinking: String?
    var tool_calls: [JSONValue]?
    var tool_name: String?
}

/// One NDJSON line from the /api/chat stream.
struct OllamaStreamChunk: Decodable {
    let message: MessageSummary?
    let done: Bool?
    let done_reason: String?
    let prompt_eval_count: Int?
    let eval_count: Int?

    struct MessageSummary: Decodable {
        let role: String?
        let content: String?
        let thinking: String?
        let tool_calls: [ToolCall]?

        struct ToolCall: Decodable {
            let function: Function?

            struct Function: Decodable {
                let name: String?
                let arguments: JSONValue?
            }
        }
    }
}

/// Response from `GET /api/tags`.
struct OllamaTagsResponse: Decodable {
    let models: [Item]?

    struct Item: Decodable {
        let name: String?
        let model: String?
    }
}
