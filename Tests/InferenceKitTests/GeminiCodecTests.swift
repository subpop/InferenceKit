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

import Testing
import Foundation

@testable import InferenceKit

@Suite("Gemini generateContent codec")
struct GeminiCodecTests {
    private let format = GeminiGenerateContentFormat()

    private func makeRequest() throws -> InferenceRequest {
        InferenceRequest(
            system: ["You are helpful."],
            messages: [
                .user("Weather?", attachments: [
                    .image(ImageBlock(data: Data([0x89]), mediaType: "image/png"))
                ]),
                .assistant(
                    thinking: "hmm",
                    text: "Let me check.",
                    toolCalls: [ToolCall(id: "call_1", name: "get_weather", arguments: #"{"city": "Cupertino"}"#)]
                ),
                .toolResult(callID: "call_1", name: "get_weather", output: "72F sunny"),
            ],
            tools: [
                try ToolDefinition(
                    name: "get_weather",
                    description: "Get the weather",
                    parametersJSON: #"{"type": "object", "properties": {"city": {"type": "string"}}, "additionalProperties": false}"#
                ),
            ],
            options: GenerationOptions(
                maxOutputTokens: 1024,
                thinking: .budgetedTokens(2048)
            )
        )
    }

    @Test("Encodes a full request to the generateContent wire format")
    func requestGolden() throws {
        let urlRequest = try format.makeChatRequest(
            makeRequest(),
            provider: .gemini(apiKey: "key"),
            model: .custom("gemini-test")
        )

        #expect(urlRequest.url?.absoluteString ==
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:streamGenerateContent?alt=sse")
        #expect(urlRequest.httpMethod == "POST")

        let body = try JSONValue(parsing: #require(urlRequest.httpBody))
        let expected = try JSONValue(parsing: #"""
        {
          "contents": [
            {
              "role": "user",
              "parts": [
                {"inlineData": {"mimeType": "image/png", "data": "iQ=="}},
                {"text": "Weather?"}
              ]
            },
            {
              "role": "model",
              "parts": [
                {"text": "hmm", "thought": true},
                {"text": "Let me check."},
                {"functionCall": {"id": "call_1", "name": "get_weather", "args": {"city": "Cupertino"}}}
              ]
            },
            {
              "role": "user",
              "parts": [
                {"functionResponse": {"id": "call_1", "name": "get_weather", "response": {"output": "72F sunny"}}}
              ]
            }
          ],
          "systemInstruction": {"parts": [{"text": "You are helpful."}]},
          "tools": [
            {
              "functionDeclarations": [
                {
                  "name": "get_weather",
                  "description": "Get the weather",
                  "parameters": {
                    "type": "OBJECT",
                    "properties": {"city": {"type": "STRING"}},
                  }
                }
              ]
            }
          ],
          "generationConfig": {
            "maxOutputTokens": 1024,
            "thinkingConfig": {"includeThoughts": true, "thinkingBudget": 2048}
          }
        }
        """#)
        #expect(body == expected)
    }

    @Test("Thinking configurations map to thinkingConfig")
    func thinkingOptions() throws {
        var request = try makeRequest()
        request.options.thinking = .disabled
        let disabled = try JSONValue(parsing: #require(
            format.makeChatRequest(request, provider: .gemini(apiKey: "k"), model: .custom("m")).httpBody
        ))
        #expect(disabled["generationConfig"]?["thinkingConfig"] == .object([
            "includeThoughts": .bool(false),
            "thinkingBudget": .int(0),
        ]))

        request.options.thinking = .effort(.high)
        #expect(throws: InferenceError.self) {
            _ = try format.makeChatRequest(request, provider: .gemini(apiKey: "k"), model: .custom("m"))
        }
    }

    @Test("Tool results require names")
    func toolResultRequiresName() throws {
        var request = try makeRequest()
        request.options.thinking = nil
        request.messages[2] = .toolResult(callID: "call_1", output: "72F")
        #expect(throws: InferenceError.self) {
            _ = try format.makeChatRequest(request, provider: .gemini(apiKey: "k"), model: .custom("m"))
        }
    }

    @Test("Schema conversion uppercases types and strips unsupported keys")
    func schemaConversion() throws {
        let schema = try JSONValue(parsing: #"""
        {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "number"}},
            "nested": {"type": "object", "properties": {"ok": {"type": "boolean"}}}
          },
          "additionalProperties": false
        }
        """#)
        let converted = GeminiGenerateContentFormat.geminiSchema(schema)
        #expect(converted["type"]?.stringValue == "OBJECT")
        #expect(converted["additionalProperties"] == nil)
        #expect(converted["properties"]?["city"]?["type"]?.stringValue == "STRING")
        #expect(converted["properties"]?["tags"]?["type"]?.stringValue == "ARRAY")
        #expect(converted["properties"]?["tags"]?["items"]?["type"]?.stringValue == "NUMBER")
        #expect(converted["properties"]?["nested"]?["properties"]?["ok"]?["type"]?.stringValue == "BOOLEAN")
    }

    // MARK: - Stream decoding

    private let streamPayloads: [String] = [
        #"{"candidates": [{"content": {"role": "model", "parts": [{"text": "Thinking", "thought": true, "thoughtSignature": "sig1"}]}}]}"#,
        #"{"candidates": [{"content": {"role": "model", "parts": [{"text": "Hello"}]}}]}"#,
        #"{"candidates": [{"content": {"role": "model", "parts": [{"text": " there"}]}}]}"#,
        #"{"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"id": "call_1", "name": "get_weather", "args": {"city": "Cupertino"}}}]}}]}"#,
        #"{"candidates": [{"finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 34, "thoughtsTokenCount": 5}, "modelVersion": "gemini-2.5-flash"}"#,
    ]

    @Test("Decodes a full generateContent stream")
    func streamGolden() throws {
        let decoder = format.makeStreamDecoder()
        var events: [StreamEvent] = []
        for payload in streamPayloads {
            events += try decoder.feed(payload)
        }
        events += try decoder.finish()

        let arguments = #"{"city":"Cupertino"}"#
        let assistant = Message(role: .assistant, content: [
            .thinking(ThinkingBlock(text: "Thinking", signature: "sig1")),
            .text("Hello there"),
            .toolCall(ToolCallBlock(id: "call_1", name: "get_weather", arguments: arguments)),
        ])

        guard events.count == 6 else {
            Issue.record("Unexpected event count: \(events.count)")
            return
        }

        if case .thinkingDelta("Thinking") = events[0] {} else {
            Issue.record("Expected thinking delta first")
        }
        if case .textDelta("Hello") = events[1] {} else {
            Issue.record("Expected text delta")
        }
        if case .textDelta(" there") = events[2] {} else {
            Issue.record("Expected text delta")
        }
        if case .toolCall(let call) = events[3] {
            #expect(call.id == "call_1")
            #expect(call.name == "get_weather")
            let parsed = try JSONValue(parsing: call.arguments)
            let expectedArgs = try JSONValue(parsing: arguments)
            #expect(parsed == expectedArgs)
        } else {
            Issue.record("Expected tool call")
        }
        if case .usage(let usage) = events[4] {
            #expect(usage == Usage(inputTokens: 12, outputTokens: 34, reasoningTokens: 5))
        } else {
            Issue.record("Expected usage")
        }
        if case .completed(let completion) = events[5] {
            #expect(completion.finishReason == .stop)
            #expect(completion.model == "gemini-2.5-flash")
            #expect(completion.assistantMessage?.content.map(Self.blockKind) ==
                assistant.content.map(Self.blockKind))
            // Thought signatures survive the round trip.
            #expect(completion.assistantMessage?.content.first ==
                .thinking(ThinkingBlock(text: "Thinking", signature: "sig1")))
        } else {
            Issue.record("Expected completion")
        }
    }

    private static func blockKind(_ block: ContentBlock) -> String {
        switch block {
        case .thinking: return "thinking"
        case .text: return "text"
        case .toolCall: return "toolCall"
        case .toolResult: return "toolResult"
        case .image: return "image"
        case .file: return "file"
        }
    }

    @Test("Stream errors throw")
    func streamError() {
        let decoder = format.makeStreamDecoder()
        #expect(throws: InferenceError.self) {
            _ = try decoder.feed(#"{"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}}"#)
        }
    }

    @Test("Finish reasons map")
    func finishReasons() throws {
        for (raw, expected) in [
            ("STOP", FinishReason.stop),
            ("MAX_TOKENS", FinishReason.length),
            ("SAFETY", FinishReason.contentFilter),
        ] {
            let decoder = format.makeStreamDecoder()
            var events = try decoder.feed(
                #"{"candidates": [{"finishReason": "\#(raw)"}]}"#
            )
            events += try decoder.finish()
            guard case .completed(let completion) = events.last else {
                Issue.record("Expected a completion event for \(raw)")
                continue
            }
            #expect(completion.finishReason == expected)
        }
    }

    @Test("Model list paginates and filters to generateContent models")
    func modelList() async throws {
        let provider = ProviderConfiguration.gemini(apiKey: "key")
        let log = RequestLog()
        let models: [DiscoveredModel] = try await format.listModels(provider: provider) { request in
            await log.append(request)
            let page: String
            if request.url?.absoluteString.contains("pageToken=tok2") == true {
                page = #"{"models": [{"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent"]}]}"#
            } else {
                page = #"{"models": [{"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]}, {"name": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash", "supportedGenerationMethods": ["generateContent", "countTokens"]}], "nextPageToken": "tok2"}"#
            }
            let response = HTTPURLResponse(
                url: request.url ?? URL(staticString: "https://example.com"),
                statusCode: 200,
                httpVersion: nil,
                headerFields: nil
            )!
            return (Data(page.utf8), response)
        }

        #expect(models.map(\.id) == ["gemini-2.5-flash", "gemini-2.5-pro"])
        #expect(models[0].displayName == "Gemini 2.5 Flash")
        let requests = await log.requests
        #expect(requests.count == 2)
        #expect(requests[0].value(forHTTPHeaderField: "x-goog-api-key") == "key")
    }

    private actor RequestLog {
        var requests: [URLRequest] = []
        func append(_ request: URLRequest) {
            requests.append(request)
        }
    }
}
