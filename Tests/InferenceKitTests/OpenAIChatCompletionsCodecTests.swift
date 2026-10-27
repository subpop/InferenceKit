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

@Suite("OpenAI Chat Completions codec")
struct OpenAIChatCompletionsCodecTests {
    private let format = OpenAIChatCompletionsFormat()

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
                temperature: 0.5,
                maxOutputTokens: 1024,
                stopSequences: ["END"],
                thinking: .effort(.low)
            )
        )
    }

    @Test("Encodes a full request to the Chat Completions wire format")
    func requestGolden() throws {
        let urlRequest = try format.makeChatRequest(
            makeRequest(),
            provider: .openAIChatCompletions(apiKey: "key"),
            model: .custom("test-model")
        )

        #expect(urlRequest.url?.absoluteString == "https://api.openai.com/v1/chat/completions")
        #expect(urlRequest.httpMethod == "POST")

        let body = try JSONValue(parsing: #require(urlRequest.httpBody))
        let expected = try JSONValue(parsing: #"""
        {
          "model": "test-model",
          "messages": [
            {"role": "system", "content": "You are helpful."},
            {
              "role": "user",
              "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iQ=="}},
                {"type": "text", "text": "Weather?"}
              ]
            },
            {
              "role": "assistant",
              "content": "Let me check.",
              "tool_calls": [
                {
                  "id": "call_1",
                  "type": "function",
                  "function": {"name": "get_weather", "arguments": "{\"city\": \"Cupertino\"}"}
                }
              ]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "72F sunny"}
          ],
          "stream": true,
          "stream_options": {"include_usage": true},
          "temperature": 0.5,
          "max_completion_tokens": 1024,
          "stop": ["END"],
          "tools": [
            {
              "type": "function",
              "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
              }
            }
          ],
          "reasoning_effort": "low"
        }
        """#)
        #expect(body == expected)
    }

    @Test("Effort none sends an explicit reasoning_effort")
    func effortNone() throws {
        var request = try makeRequest()
        request.options.thinking = .effort(.none)
        let body = try JSONValue(parsing: #require(
            format.makeChatRequest(request, provider: .openAIChatCompletions(apiKey: "k"), model: .custom("m")).httpBody
        ))
        #expect(body["reasoning_effort"] == .string("none"))
    }

    @Test("Text-only user messages encode as plain strings")
    func plainUserMessage() throws {
        var request = try makeRequest()
        request.messages[0] = .user("Just text")
        request.options.thinking = nil
        request.options.stopSequences = nil
        let body = try JSONValue(parsing: #require(
            format.makeChatRequest(request, provider: .openAIChatCompletions(apiKey: "k"), model: .custom("m")).httpBody
        ))
        #expect(body["messages"]?[1]?["content"]?.stringValue == "Just text")
    }

    @Test("Thinking blocks are dropped from history")
    func thinkingDropped() throws {
        let body = try JSONValue(parsing: #require(
            format.makeChatRequest(makeRequest(), provider: .openAIChatCompletions(apiKey: "k"), model: .custom("m")).httpBody
        ))
        // The assistant message encoded "hmm" thinking but only its text and
        // tool calls reach the wire.
        #expect(body["messages"]?[2]?["content"]?.stringValue == "Let me check.")
        let toolCalls = body["messages"]?[2]?["tool_calls"]?.arrayValue
        #expect(toolCalls?.count == 1)
    }

    @Test("Reasoning budgets are unsupported")
    func unsupported() throws {
        var request = try makeRequest()
        request.options.thinking = .budgetedTokens(2048)
        #expect(throws: InferenceError.self) {
            _ = try format.makeChatRequest(request, provider: .openAIChatCompletions(apiKey: "k"), model: .custom("m"))
        }
    }

    // MARK: - Stream decoding

    private let streamPayloads: [String] = [
        #"{"id": "chatcmpl_1", "model": "gpt-5.1", "choices": [{"index": 0, "delta": {"role": "assistant"}}]}"#,
        #"{"choices": [{"index": 0, "delta": {"reasoning_content": "Thinking"}}]}"#,
        #"{"choices": [{"index": 0, "delta": {"content": "Hello"}}]}"#,
        #"{"choices": [{"index": 0, "delta": {"content": " there"}}]}"#,
        #"{"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]}}]}"#,
        #"{"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\":"}}]}}]}"#,
        #"{"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"Cupertino\"}"}}]}}]}"#,
        #"{"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}"#,
        #"{"usage": {"prompt_tokens": 12, "completion_tokens": 34, "completion_tokens_details": {"reasoning_tokens": 5}}}"#,
        #"[DONE]"#,
    ]

    @Test("Decodes a full Chat Completions stream")
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
            arguments: #"{"city":"Cupertino"}"#
        )
        let assistant = Message(role: .assistant, content: [
            .thinking(ThinkingBlock(text: "Thinking")),
            .text("Hello there"),
            .toolCall(ToolCallBlock(id: "call_1", name: "get_weather", arguments: #"{"city":"Cupertino"}"#)),
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
                responseID: "chatcmpl_1",
                model: "gpt-5.1",
                assistantMessage: assistant
            )),
        ].map(Self.strippingMessageID))
    }

    @Test("Multiple tool calls flush in index order")
    func multipleToolCalls() throws {
        let decoder = format.makeStreamDecoder()
        var events: [StreamEvent] = []
        let payloads: [String] = [
            #"{"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "id": "call_b", "function": {"name": "b", "arguments": "{}"}}]}}]}"#,
            #"{"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_a", "function": {"name": "a", "arguments": "{}"}}]}}]}"#,
            #"{"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}"#,
        ]
        for payload in payloads {
            events += try decoder.feed(payload)
        }
        let calls = events.compactMap { event -> ToolCall? in
            if case .toolCall(let call) = event { return call }
            return nil
        }
        #expect(calls.map(\.id) == ["call_a", "call_b"])
    }

    @Test("Finish reasons map")
    func finishReasons() throws {
        for (raw, expected) in [
            ("stop", FinishReason.stop),
            ("length", FinishReason.length),
            ("tool_calls", FinishReason.toolCalls),
            ("content_filter", FinishReason.contentFilter),
            ("weird_future_value", FinishReason.other),
        ] {
            let decoder = format.makeStreamDecoder()
            let events = try decoder.feed(
                #"{"choices": [{"index": 0, "delta": {}, "finish_reason": "\#(raw)"}]}"#
            )
            events.forEach { _ in }
            let finished = try decoder.finish()
            guard case .completed(let completion) = finished.last else {
                Issue.record("Expected a completion event for \(raw)")
                continue
            }
            #expect(completion.finishReason == expected)
        }
    }

    private static func strippingMessageID(_ event: StreamEvent) -> StreamEvent {
        guard case .completed(var completion) = event else { return event }
        completion.assistantMessage?.id = UUID(uuidString: "00000000-0000-0000-0000-000000000000")!
        return .completed(completion)
    }
}
