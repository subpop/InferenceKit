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

/// Request body for `POST /v1beta/models/{model}:streamGenerateContent`.
struct GeminiRequestBody: Encodable {
    var contents: [GeminiWireContent]
    var systemInstruction: GeminiWireContent?
    var tools: [GeminiTools]?
    var generationConfig: GenerationConfig

    struct GenerationConfig: Encodable {
        var maxOutputTokens: Int?
        var temperature: Double?
        var topP: Double?
        var stopSequences: [String]?
        var thinkingConfig: ThinkingConfig?

        struct ThinkingConfig: Encodable {
            var includeThoughts: Bool
            var thinkingBudget: Int?
        }
    }
}

struct GeminiTools: Encodable {
    var functionDeclarations: [GeminiFunctionDeclaration]
}

struct GeminiFunctionDeclaration: Encodable {
    var name: String
    var description: String
    var parameters: JSONValue
}

/// A content turn. Parts are built as `JSONValue` because their shapes vary
/// (text, inlineData, functionCall, functionResponse, thought signatures).
struct GeminiWireContent: Encodable {
    var role: String?
    var parts: [JSONValue]
}

/// One SSE data line from the streamGenerateContent response: a full
/// `GenerateContentResponse`.
struct GeminiStreamChunk: Decodable {
    let candidates: [Candidate]?
    let usageMetadata: UsageMetadata?
    let modelVersion: String?
    let error: ErrorPayload?

    struct Candidate: Decodable {
        let content: Content?
        let finishReason: String?
    }

    struct Content: Decodable {
        let role: String?
        let parts: [Part]?
    }

    struct Part: Decodable {
        let text: String?
        let thought: Bool?
        let thoughtSignature: String?
        let inlineData: InlineData?
        let functionCall: FunctionCall?

        struct InlineData: Decodable {
            let mimeType: String?
            let data: String?
        }

        struct FunctionCall: Decodable {
            let id: String?
            let name: String?
            let args: JSONValue?
        }
    }

    struct UsageMetadata: Decodable {
        let promptTokenCount: Int?
        let candidatesTokenCount: Int?
        let thoughtsTokenCount: Int?
    }

    struct ErrorPayload: Decodable {
        let code: Int?
        let message: String?
        let status: String?
    }
}

/// One page of `GET /v1beta/models`.
struct GeminiModelsPage: Decodable {
    let models: [Item]?
    let nextPageToken: String?

    struct Item: Decodable {
        let name: String?
        let displayName: String?
        let supportedGenerationMethods: [String]?
    }
}
