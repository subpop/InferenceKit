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

@Suite("OpenAI Responses codec")
struct OpenAIResponsesCodecTests {
    private let format = OpenAIResponsesFormat()

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
                    parametersJSON: #"{"type": "object", "properties": {"city": {"type": "string"}}}"#
                ),
            ],
            options: GenerationOptions(
                maxOutputTokens: 1024,
                thinking: .effort(.low)
            )
        )
    }

    @Test("Encodes a full request to the Responses wire format")
    func requestGolden() throws {
        let urlRequest = try format.makeChatRequest(
            makeRequest(),
            provider: .openAI(apiKey: "key"),
            model: .custom("test-model")
        )

        #expect(urlRequest.url?.absoluteString == "https://api.openai.com/v1/responses")
        #expect(urlRequest.httpMethod == "POST")

        let body = try JSONValue(parsing: #require(urlRequest.httpBody))
        let expected = try JSONValue(parsing: #"""
        {
          "model": "test-model",
          "instructions": "You are helpful.",
          "input": [
            {
              "type": "message",
              "role": "user",
              "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,iQ=="},
                {"type": "input_text", "text": "Weather?"}
              ]
            },
            {
              "type": "reasoning",
              "summary": [{"type": "summary_text", "text": "hmm"}]
            },
            {
              "type": "message",
              "role": "assistant",
              "content": [{"type": "output_text", "text": "Let me check."}]
            },
            {
              "type": "function_call",
              "call_id": "call_1",
              "name": "get_weather",
              "arguments": "{\"city\": \"Cupertino\"}"
            },
            {
              "type": "function_call_output",
              "call_id": "call_1",
              "output": "72F sunny"
            }
          ],
          "stream": true,
          "max_output_tokens": 1024,
          "tools": [
            {
              "type": "function",
              "name": "get_weather",
              "description": "Get the weather",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
          ],
          "reasoning": {"effort": "low", "summary": "auto"}
        }
        """#)
        #expect(body == expected)
    }

    @Test("Thinking configurations map to reasoning")
    func thinkingOptions() throws {
        var request = try makeRequest()
        request.options.thinking = .automatic
        let automatic = try JSONValue(parsing: #require(
            format.makeChatRequest(request, provider: .openAI(apiKey: "k"), model: .custom("m")).httpBody
        ))
        #expect(automatic["reasoning"] == .object(["summary": .string("auto")]))

        request.options.thinking = .effort(.none)
        let noEffort = try JSONValue(parsing: #require(
            format.makeChatRequest(request, provider: .openAI(apiKey: "k"), model: .custom("m")).httpBody
        ))
        #expect(noEffort["reasoning"] == .object(["effort": .string("none"), "summary": .string("auto")]))

        request.options.thinking = nil
        let none = try JSONValue(parsing: #require(
            format.makeChatRequest(request, provider: .openAI(apiKey: "k"), model: .custom("m")).httpBody
        ))
        #expect(none["reasoning"] == nil)
    }

    @Test("Budgets and stop sequences are unsupported")
    func unsupported() throws {
        var request = try makeRequest()
        request.options.thinking = .budgetedTokens(2048)
        #expect(throws: InferenceError.self) {
            _ = try format.makeChatRequest(request, provider: .openAI(apiKey: "k"), model: .custom("m"))
        }
        request.options.thinking = nil
        request.options.stopSequences = ["END"]
        #expect(throws: InferenceError.self) {
            _ = try format.makeChatRequest(request, provider: .openAI(apiKey: "k"), model: .custom("m"))
        }
    }

    @Test("Compaction maps to context management")
    func compaction() throws {
        var request = try makeRequest()
        request.options.thinking = nil
        request.options.compaction = Compaction(triggerThresholdTokens: 50_000)
        let body = try JSONValue(parsing: #require(
            format.makeChatRequest(request, provider: .openAI(apiKey: "k"), model: .custom("m")).httpBody
        ))
        #expect(
            body["context_management"]?[0]?["compact_threshold"]?.intValue == 50_000
        )
    }

    @Test("Reasoning signatures round-trip as item ids")
    func reasoningSignature() throws {
        var request = try makeRequest()
        request.options.thinking = nil
        request.messages[1].content[0] = .thinking(ThinkingBlock(text: "thought", signature: "rs_1"))
        let body = try JSONValue(parsing: #require(
            format.makeChatRequest(request, provider: .openAI(apiKey: "k"), model: .custom("m")).httpBody
        ))
        #expect(body["input"]?[1]?["type"]?.stringValue == "reasoning")
        #expect(body["input"]?[1]?["id"]?.stringValue == "rs_1")
    }

    // MARK: - Stream decoding

    private let streamPayloads: [String] = [
        #"{"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5.1"}}"#,
        #"{"type": "response.output_item.added", "item": {"id": "rs_1", "type": "reasoning"}}"#,
        #"{"type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "delta": "Thinking"}"#,
        #"{"type": "response.output_item.added", "item": {"id": "msg_1", "type": "message"}}"#,
        #"{"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hello"}"#,
        #"{"type": "response.output_text.delta", "item_id": "msg_1", "delta": " there"}"#,
        #"{"type": "response.output_item.added", "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "get_weather"}}"#,
        #"{"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": "{\"city\":"}"#,
        #"{"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": "\"Cupertino\"}"}"#,
        #"{"type": "response.function_call_arguments.done", "item_id": "fc_1", "arguments": "{\"city\": \"Cupertino\"}"}"#,
        #"{"type": "response.completed", "response": {"id": "resp_1", "model": "gpt-5.1", "status": "completed", "usage": {"input_tokens": 12, "output_tokens": 34, "output_tokens_details": {"reasoning_tokens": 5}}, "output": [{"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "Thinking"}]}, {"id": "msg_1", "type": "message", "content": [{"type": "output_text", "text": "Hello there"}]}, {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\": \"Cupertino\"}"}]}}"#,
    ]

    @Test("Decodes a full Responses stream")
    func streamGolden() throws {
        let decoder = format.makeStreamDecoder()
        var events: [StreamEvent] = []
        for payload in streamPayloads {
            events += try decoder.feed(payload)
        }
        events += try decoder.finish()

        let toolCall = ToolCall(
            id: "call_1",
            name: "get_weather",
            arguments: #"{"city": "Cupertino"}"#
        )
        let assistant = Message(role: .assistant, content: [
            .thinking(ThinkingBlock(text: "Thinking", signature: "rs_1")),
            .text("Hello there"),
            .toolCall(ToolCallBlock(id: "call_1", name: "get_weather", arguments: #"{"city": "Cupertino"}"#)),
        ])

        #expect(events.map(Self.strippingMessageID) == [
            .thinkingDelta("Thinking"),
            .textDelta("Hello"),
            .textDelta(" there"),
            .toolCall(toolCall),
            .usage(Usage(inputTokens: 12, outputTokens: 34, reasoningTokens: 5)),
            .completed(Completion(
                finishReason: .toolCalls,
                usage: Usage(inputTokens: 12, outputTokens: 34, reasoningTokens: 5),
                responseID: "resp_1",
                model: "gpt-5.1",
                assistantMessage: assistant
            )),
        ].map(Self.strippingMessageID))
    }

    /// The `response.completed` path emits no separate toolCall event, but
    /// truncated streams must flush buffered calls at `finish()`.
    @Test("Truncated streams flush buffered tool calls")
    func truncatedStream() throws {
        let decoder = format.makeStreamDecoder()
        var events: [StreamEvent] = []
        for payload in streamPayloads.prefix(9) {
            events += try decoder.feed(payload)
        }
        events += try decoder.finish()

        #expect(events.contains(.toolCall(ToolCall(
            id: "call_1",
            name: "get_weather",
            arguments: #"{"city":"Cupertino"}"#
        ))))
        guard case .completed(let completion) = events.last else {
            Issue.record("Expected a completion event")
            return
        }
        #expect(completion.finishReason == .toolCalls)
        #expect(completion.assistantMessage?.content.last == .toolCall(ToolCallBlock(
            id: "call_1",
            name: "get_weather",
            arguments: #"{"city":"Cupertino"}"#
        )))
    }

    @Test("Stream failures throw")
    func streamError() {
        let decoder = format.makeStreamDecoder()
        #expect(throws: InferenceError.self) {
            _ = try decoder.feed(#"{"type": "response.failed", "response": {"error": {"code": "x", "message": "Failed"}}}"#)
        }
        let decoder2 = format.makeStreamDecoder()
        #expect(throws: InferenceError.self) {
            _ = try decoder2.feed(#"{"type": "error", "error": {"code": "x", "message": "Bad"}}"#)
        }
    }

    @Test("Incomplete responses report length")
    func incompleteReportsLength() throws {
        let decoder = format.makeStreamDecoder()
        let events = try decoder.feed(
            #"{"type": "response.completed", "response": {"id": "r", "status": "incomplete", "usage": null, "output": [{"type": "message", "content": [{"type": "output_text", "text": "partial"}]}]}}"#
        )
        guard case .completed(let completion) = events.last else {
            Issue.record("Expected a completion event")
            return
        }
        #expect(completion.finishReason == .length)
    }

    @Test("Model list hits /models")
    func modelList() async throws {
        let provider = ProviderConfiguration.openAI(apiKey: "key")
        let models = try await format.listModels(provider: provider) { request in
            #expect(request.url?.absoluteString == "https://api.openai.com/v1/models")
            #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer key")
            let page = #"{"data": [{"id": "gpt-5.1"}, {"id": "gpt-4o"}]}"#
            let response = HTTPURLResponse(
                url: request.url ?? URL(staticString: "https://example.com"),
                statusCode: 200,
                httpVersion: nil,
                headerFields: nil
            )!
            return (Data(page.utf8), response)
        }
        #expect(models.map(\.id) == ["gpt-5.1", "gpt-4o"])
    }

    private static func strippingMessageID(_ event: StreamEvent) -> StreamEvent {
        guard case .completed(var completion) = event else { return event }
        completion.assistantMessage?.id = UUID(uuidString: "00000000-0000-0000-0000-000000000000")!
        return .completed(completion)
    }
}
