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

@Suite("Anthropic Messages codec")
struct AnthropicCodecTests {
    private let format = AnthropicMessagesFormat()

    private func makeRequest() throws -> InferenceRequest {
        InferenceRequest(
            system: ["You are helpful.", "Be concise."],
            messages: [
                .user("Weather?", attachments: [
                    .image(ImageBlock(data: Data([0x89]), mediaType: "image/png"))
                ]),
                .assistant(
                    thinking: "hmm",
                    text: "Let me check.",
                    toolCalls: [ToolCall(id: "toolu_1", name: "get_weather", arguments: #"{"city": "Cupertino"}"#)]
                ),
                .toolResult(callID: "toolu_1", name: "get_weather", output: "72F sunny"),
            ],
            tools: [
                try ToolDefinition(
                    name: "get_weather",
                    description: "Get the weather",
                    parametersJSON: #"{"type": "object", "properties": {"city": {"type": "string"}}}"#
                ),
            ],
            options: GenerationOptions(
                temperature: 0.7,
                maxOutputTokens: 1024,
                thinking: .budgetedTokens(2048)
            )
        )
    }

    @Test("Encodes a full request to the Messages wire format")
    func requestGolden() throws {
        let urlRequest = try format.makeChatRequest(
            makeRequest(),
            provider: .anthropic(apiKey: "key"),
            model: .custom("test-model")
        )

        #expect(urlRequest.url?.absoluteString == "https://api.anthropic.com/v1/messages")
        #expect(urlRequest.httpMethod == "POST")
        #expect(urlRequest.value(forHTTPHeaderField: "anthropic-version") == "2023-06-01")

        let body = try JSONValue(parsing: #require(urlRequest.httpBody))
        let expected = try JSONValue(parsing: #"""
        {
          "model": "test-model",
          "max_tokens": 1024,
          "stream": true,
          "system": [
            {"type": "text", "text": "You are helpful."},
            {"type": "text", "text": "Be concise."}
          ],
          "messages": [
            {
              "role": "user",
              "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iQ=="}},
                {"type": "text", "text": "Weather?"}
              ]
            },
            {
              "role": "assistant",
              "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Cupertino"}}
              ]
            },
            {
              "role": "user",
              "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F sunny"}
              ]
            }
          ],
          "thinking": {"type": "enabled", "budget_tokens": 2048},
          "tools": [
            {
              "name": "get_weather",
              "description": "Get the weather",
              "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
          ]
        }
        """#)
        #expect(body == expected)
    }

    @Test("Thinking disables temperature and top_p; adaptive maps to type")
    func thinkingOptions() throws {
        var request = try makeRequest()
        request.options.thinking = .automatic
        request.options.topP = 0.9
        let urlRequest = try format.makeChatRequest(request, provider: .anthropic(apiKey: "key"), model: .custom("m"))
        let body = try JSONValue(parsing: #require(urlRequest.httpBody))
        #expect(body["thinking"] == .object(["type": .string("adaptive")]))
        #expect(body["temperature"] == nil)
        #expect(body["top_p"] == nil)

        request.options.thinking = nil
        request.options.temperature = 0.5
        let plain = try format.makeChatRequest(request, provider: .anthropic(apiKey: "key"), model: .custom("m"))
        let plainBody = try JSONValue(parsing: #require(plain.httpBody))
        #expect(plainBody["temperature"] == .double(0.5))
    }

    @Test("max_tokens falls back to the model configuration")
    func maxTokensFallback() throws {
        var request = try makeRequest()
        request.options.maxOutputTokens = nil
        request.options.thinking = nil

        let urlRequest = try format.makeChatRequest(
            request,
            provider: .anthropic(apiKey: "key"),
            model: .custom("m", maxOutputTokens: 4096)
        )
        let body = try JSONValue(parsing: #require(urlRequest.httpBody))
        #expect(body["max_tokens"]?.intValue == 4096)

        #expect(throws: InferenceError.self) {
            _ = try format.makeChatRequest(
                request,
                provider: .anthropic(apiKey: "key"),
                model: .custom("m")
            )
        }
    }

    @Test("Reasoning effort maps to adaptive thinking plus output_config")
    func effortMapping() throws {
        var request = try makeRequest()
        request.options.thinking = .effort(.high)
        let urlRequest = try format.makeChatRequest(
            request, provider: .anthropic(apiKey: "key"), model: .custom("m")
        )
        let body = try JSONValue(parsing: #require(urlRequest.httpBody))
        #expect(body["thinking"] == .object(["type": .string("adaptive")]))
        #expect(body["output_config"] == .object(["effort": .string("high")]))
        // Adaptive thinking requires temperature and top_p to be unset.
        #expect(body["temperature"] == nil)
        #expect(body["top_p"] == nil)

        request.options.thinking = .effort(.xhigh)
        let xhigh = try format.makeChatRequest(
            request, provider: .anthropic(apiKey: "key"), model: .custom("m")
        )
        let xhighBody = try JSONValue(parsing: #require(xhigh.httpBody))
        #expect(xhighBody["output_config"]?["effort"] == .string("xhigh"))

        // Anthropic's effort scale starts at low; "none" is OpenAI-only.
        request.options.thinking = .effort(.none)
        #expect(throws: InferenceError.self) {
            _ = try format.makeChatRequest(request, provider: .anthropic(apiKey: "key"), model: .custom("m"))
        }
    }

    @Test("Compaction maps to context management edits")
    func compaction() throws {
        var request = try makeRequest()
        request.options.compaction = Compaction(triggerThresholdTokens: 50_000)
        request.options.thinking = nil
        let urlRequest = try format.makeChatRequest(request, provider: .anthropic(apiKey: "key"), model: .custom("m"))
        let body = try JSONValue(parsing: #require(urlRequest.httpBody))
        #expect(body["context_management"]?["edits"]?[0]?["trigger"]?["value"]?.intValue == 50_000)
    }

    // MARK: - Stream decoding

    private let streamPayloads: [String] = [
        #"{"type": "message_start", "message": {"id": "msg_1", "model": "claude-sonnet-4-5", "usage": {"input_tokens": 10}}}"#,
        #"{"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}}"#,
        #"{"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me think."}}"#,
        #"{"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig1"}}"#,
        #"{"type": "content_block_stop", "index": 0}"#,
        #"{"type": "content_block_start", "index": 1, "content_block": {"type": "text"}}"#,
        #"{"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hello"}}"#,
        #"{"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": " world"}}"#,
        #"{"type": "content_block_stop", "index": 1}"#,
        #"{"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather"}}"#,
        #"{"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}}"#,
        #"{"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "\"Cupertino\"}"}}"#,
        #"{"type": "content_block_stop", "index": 2}"#,
        #"{"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 42}}"#,
        #"{"type": "message_stop"}"#,
    ]

    @Test("Decodes a full Messages stream")
    func streamGolden() throws {
        let decoder = format.makeStreamDecoder()
        var events: [StreamEvent] = []
        for payload in streamPayloads {
            events += try decoder.feed(payload)
        }
        events += try decoder.finish()

        let toolCall = ToolCall(
            id: "toolu_1",
            name: "get_weather",
            arguments: #"{"city":"Cupertino"}"#
        )
        let assistant = Message(role: .assistant, content: [
            .thinking(ThinkingBlock(text: "Let me think.", signature: "sig1")),
            .text("Hello world"),
            .toolCall(ToolCallBlock(id: "toolu_1", name: "get_weather", arguments: #"{"city":"Cupertino"}"#)),
        ])

        var expected: [StreamEvent] = [
            .usage(Usage(inputTokens: 10)),
            .thinkingDelta("Let me think."),
            .textDelta("Hello"),
            .textDelta(" world"),
            .toolCall(toolCall),
            .usage(Usage(inputTokens: 10, outputTokens: 42)),
            .completed(Completion(
                finishReason: .toolCalls,
                usage: Usage(inputTokens: 10, outputTokens: 42),
                responseID: "msg_1",
                model: "claude-sonnet-4-5",
                assistantMessage: assistant
            )),
        ]
        // The decoder assigns fresh message ids; strip them for comparison.
        expected = expected.map(Self.strippingMessageID)
        #expect(events.map(Self.strippingMessageID) == expected)
    }

    private static func strippingMessageID(_ event: StreamEvent) -> StreamEvent {
        guard case .completed(var completion) = event else { return event }
        completion.assistantMessage?.id = UUID(uuidString: "00000000-0000-0000-0000-000000000000")!
        return .completed(completion)
    }

    @Test("Stream errors throw")
    func streamError() {
        let decoder = format.makeStreamDecoder()
        #expect(throws: InferenceError.self) {
            _ = try decoder.feed(#"{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}"#)
        }
    }

    @Test("Truncated streams still complete")
    func truncatedStream() throws {
        let decoder = format.makeStreamDecoder()
        var events: [StreamEvent] = []
        for payload in streamPayloads.prefix(8) {
            events += try decoder.feed(payload)
        }
        events += try decoder.finish()
        // A completion is synthesized with whatever arrived, including the
        // text block that never received content_block_stop.
        guard case .completed(let completion) = events.last else {
            Issue.record("Expected a completion event")
            return
        }
        #expect(completion.finishReason == nil)
        #expect(completion.assistantMessage?.content == [
            .thinking(ThinkingBlock(text: "Let me think.", signature: "sig1")),
            .text("Hello world"),
        ])
    }

    @Test("Model list paginates via after_id")
    func modelListPagination() async throws {
        let provider = ProviderConfiguration.anthropic(apiKey: "key")
        let log = RequestLog()
        let models: [DiscoveredModel] = try await format.listModels(provider: provider) { request in
            await log.append(request)
            let page: String
            if request.url?.absoluteString.contains("after_id=m2") == true {
                page = #"{"data": [{"id": "m3", "display_name": "Model 3"}], "has_more": false}"#
            } else {
                page = #"{"data": [{"id": "m1"}, {"id": "m2", "display_name": "Model 2"}], "has_more": true, "last_id": "m2"}"#
            }
            let response = HTTPURLResponse(
                url: request.url ?? URL(staticString: "https://example.com"),
                statusCode: 200,
                httpVersion: nil,
                headerFields: nil
            )!
            return (Data(page.utf8), response)
        }

        let requestsSeen = await log.requests
        #expect(models.map(\.id) == ["m1", "m2", "m3"])
        #expect(models[1].displayName == "Model 2")
        #expect(requestsSeen.count == 2)
        #expect(requestsSeen[0].value(forHTTPHeaderField: "x-api-key") == "key")
        #expect(requestsSeen[0].value(forHTTPHeaderField: "anthropic-version") == "2023-06-01")
    }

    private actor RequestLog {
        var requests: [URLRequest] = []
        func append(_ request: URLRequest) {
            requests.append(request)
        }
    }
}
