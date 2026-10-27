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

/// Request body for `POST /v1/responses`.
struct OpenAIResponsesBody: Encodable {
    var model: String
    var instructions: String?
    var input: [JSONValue]
    var stream: Bool
    var temperature: Double?
    var top_p: Double?
    var max_output_tokens: Int?
    var tools: [JSONValue]?
    var reasoning: Reasoning?
    var context_management: [Compaction]?

    struct Reasoning: Encodable {
        var effort: String?
        var summary: String?
    }

    struct Compaction: Encodable {
        let type = "compaction"
        var compact_threshold: Int
    }
}

/// One SSE event from the Responses stream.
struct OpenAIResponsesEvent: Decodable {
    let type: String
    let delta: String?
    let item: Item?
    let call_id: String?
    let item_id: String?
    let arguments: String?
    let response: ResponseSummary?
    let error: ErrorPayload?

    struct Item: Decodable {
        let id: String?
        let type: String?
        let call_id: String?
        let name: String?
    }

    struct ResponseSummary: Decodable {
        let id: String?
        let model: String?
        let status: String?
        let usage: UsageSummary?
        let output: [OutputItem]?
        let error: ErrorPayload?
        let incomplete_details: IncompleteDetails?

        struct UsageSummary: Decodable {
            let input_tokens: Int?
            let output_tokens: Int?
            let output_tokens_details: Details?

            struct Details: Decodable {
                let reasoning_tokens: Int?
            }
        }

        struct OutputItem: Decodable {
            let id: String?
            let type: String?
            let call_id: String?
            let name: String?
            let arguments: String?
            let summary: [Summary]?
            let content: [Content]?

            struct Summary: Decodable {
                let type: String?
                let text: String?
            }

            struct Content: Decodable {
                let type: String?
                let text: String?
            }
        }

        struct IncompleteDetails: Decodable {
            let reason: String?
        }
    }

    struct ErrorPayload: Decodable {
        let code: String?
        let message: String?
    }
}

/// One page of `GET /v1/models`.
struct OpenAIModelsPage: Decodable {
    let data: [Item]

    struct Item: Decodable {
        let id: String
    }
}
