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

@Suite("Ollama codec")
struct OllamaCodecTests {
    private let format = OllamaFormat()

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
                thinking: .budgetedTokens(2048),
                contextWindowTokens: 8192
            )
        )
    }

    @Test("Encodes a full request to the /api/chat wire format")
    func requestGolden() throws {
        let urlRequest = try format.makeChatRequest(
            makeRequest(),
            provider: .ollama(),
            model: .custom("llama3.2")
        )

        #expect(urlRequest.url?.absoluteString == "http://localhost:11434/api/chat")
        #expect(urlRequest.httpMethod == "POST")

        let body = try JSONValue(parsing: #require(urlRequest.httpBody))
        let expected = try JSONValue(parsing: #"""
        {
          "model": "llama3.2",
          "messages": [
            {"role": "system", "content": "You are helpful."},
            {
              "role": "user",
              "content": "Weather?",
              "images": ["iQ=="]
            },
            {
              "role": "assistant",
              "content": "Let me check.",
              "thinking": "hmm",
              "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Cupertino"}}}
              ]
            },
            {"role": "tool", "content": "72F sunny", "tool_name": "get_weather"}
          ],
          "stream": true,
          "think": {"type": "enable", "budget": 2048},
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
          "options": {
            "num_ctx": 8192,
            "temperature": 0.5
          }
        }
        """#)
        #expect(body == expected)
    }

    @Test("Thinking configurations map to the think parameter")
    func thinkingOptions() throws {
        var request = try makeRequest()
        request.options.thinking = .automatic
        let automatic = try JSONValue(parsing: #require(
            format.makeChatRequest(request, provider: .ollama(), model: .custom("m")).httpBody
        ))
        #expect(automatic["think"] == .bool(true))

        request.options.thinking = .disabled
        let disabled = try JSONValue(parsing: #require(
            format.makeChatRequest(request, provider: .ollama(), model: .custom("m")).httpBody
        ))
        #expect(disabled["think"] == .bool(false))

        request.options.thinking = .effort(.high)
        #expect(throws: InferenceError.self) {
            _ = try format.makeChatRequest(request, provider: .ollama(), model: .custom("m"))
        }
    }

    @Test("Reasoning effort is unsupported")
    func effortUnsupported() throws {
        var request = try makeRequest()
        request.options.thinking = .effort(.low)
        #expect(throws: InferenceError.self) {
            _ = try format.makeChatRequest(request, provider: .ollama(), model: .custom("m"))
        }
    }

    // MARK: - Stream decoding

    private let streamPayloads: [String] = [
        #"{"message": {"role": "assistant", "thinking": "Thinking", "content": ""}}"#,
        #"{"message": {"role": "assistant", "content": "Hello"}}"#,
        #"{"message": {"role": "assistant", "content": " there"}}"#,
        #"{"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Cupertino"}}}]}}"#,
        #"{"done": true, "done_reason": "stop", "prompt_eval_count": 12, "eval_count": 34}"#,
    ]

    @Test("Decodes a full /api/chat stream")
    func streamGolden() throws {
        let decoder = format.makeStreamDecoder()
        var events: [StreamEvent] = []
        for payload in streamPayloads {
            events += try decoder.feed(payload)
        }
        events += try decoder.finish()

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
            #expect(call.name == "get_weather")
            let parsed = try JSONValue(parsing: call.arguments)
            let expectedArgs = try JSONValue(parsing: #"{"city": "Cupertino"}"#)
            #expect(parsed == expectedArgs)
        } else {
            Issue.record("Expected tool call")
        }
        if case .usage(let usage) = events[4] {
            #expect(usage == Usage(inputTokens: 12, outputTokens: 34))
        } else {
            Issue.record("Expected usage")
        }
        if case .completed(let completion) = events[5] {
            #expect(completion.finishReason == .stop)
            #expect(completion.usage == Usage(inputTokens: 12, outputTokens: 34))
            #expect(completion.assistantMessage?.content.count == 3)
            #expect(completion.assistantMessage?.content[0] == .thinking(ThinkingBlock(text: "Thinking")))
            #expect(completion.assistantMessage?.content[1] == .text("Hello there"))
            if case .toolCall(let block)? = completion.assistantMessage?.content[2] {
                #expect(block.name == "get_weather")
            } else {
                Issue.record("Expected tool call block")
            }
        } else {
            Issue.record("Expected completion")
        }
    }

    @Test("Streams without done still complete")
    func truncatedStream() throws {
        let decoder = format.makeStreamDecoder()
        var events: [StreamEvent] = []
        for payload in streamPayloads.prefix(3) {
            events += try decoder.feed(payload)
        }
        events += try decoder.finish()
        guard case .completed(let completion) = events.last else {
            Issue.record("Expected a completion event")
            return
        }
        #expect(completion.assistantMessage?.content == [
            .thinking(ThinkingBlock(text: "Thinking")),
            .text("Hello there"),
        ])
    }

    @Test("Model list hits /api/tags")
    func modelList() async throws {
        let provider = ProviderConfiguration.ollama()
        let models = try await format.listModels(provider: provider) { request in
            #expect(request.url?.absoluteString == "http://localhost:11434/api/tags")
            let tags = #"{"models": [{"name": "llama3.2:latest"}, {"model": "qwen3:8b"}]}"#
            let response = HTTPURLResponse(
                url: request.url ?? URL(staticString: "https://example.com"),
                statusCode: 200,
                httpVersion: nil,
                headerFields: nil
            )!
            return (Data(tags.utf8), response)
        }
        #expect(models.map(\.id) == ["llama3.2:latest", "qwen3:8b"])
    }

    @Test("Error bodies decode")
    func errorBodies() {
        #expect(
            format.decodeErrorMessage(Data(#"{"error": "model not found"}"#.utf8)) == "model not found"
        )
    }
}
