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

@Suite("Message model")
struct MessageTests {
    @Test("Codable round-trip preserves all block kinds")
    func codableRoundTrip() throws {
        let message = Message(
            role: .assistant,
            content: [
                .thinking(ThinkingBlock(text: "hmm", signature: "sig")),
                .text("Hello"),
                .toolCall(ToolCallBlock(id: "call_1", name: "tool", arguments: "{}")),
                .toolResult(ToolResultBlock(callID: "call_1", name: "tool", output: "ok", isError: true)),
                .image(ImageBlock(data: Data([1]), mediaType: "image/png")),
                .file(FileBlock(data: Data([2]), filename: "a.pdf", mediaType: "application/pdf")),
            ]
        )

        let data = try JSONEncoder().encode(message)
        let decoded = try JSONDecoder().decode(Message.self, from: data)
        #expect(decoded == message)
    }

    @Test("Convenience constructors build expected content")
    func conveniences() {
        let user = Message.user("hi", attachments: [
            .image(ImageBlock(data: Data([3]), mediaType: "image/jpeg"))
        ])
        #expect(user.role == .user)
        #expect(user.content.count == 2)
        #expect(user.content.first?.text == nil)

        let assistant = Message.assistant(
            thinking: "thought",
            text: "answer",
            toolCalls: [ToolCall(id: "1", name: "n", arguments: "{}")]
        )
        #expect(assistant.content.count == 3)
        #expect(assistant.content[0] == .thinking(ThinkingBlock(text: "thought")))
        #expect(assistant.content[1] == .text("answer"))
        #expect(assistant.content[2] == .toolCall(ToolCallBlock(id: "1", name: "n", arguments: "{}")))

        let tool = Message.toolResult(callID: "1", name: "n", output: "result")
        #expect(tool.role == .tool)
    }

    @Test("Tool calls decode typed arguments")
    func toolCallDecoding() throws {
        struct Args: Codable, Equatable {
            var city: String
        }
        let call = ToolCall(id: "1", name: "weather", arguments: #"{"city": "Cupertino"}"#)
        let args = try call.decodeArguments(as: Args.self)
        #expect(args == Args(city: "Cupertino"))
    }

    @Test("Options validation rejects out-of-range budgets")
    func optionValidation() throws {
        let model = ModelConfiguration.custom(
            "m",
            thinking: ThinkingCapability(budgetRange: 1024...32000)
        )
        let options = GenerationOptions(thinking: .budgetedTokens(100))
        #expect(throws: InferenceError.self) {
            try options.validate(against: model)
        }
        var valid = GenerationOptions(thinking: .budgetedTokens(2048))
        try valid.validate(against: model)

        // Unknown capability: no validation performed.
        valid.thinking = .budgetedTokens(1)
        try valid.validate(against: ModelConfiguration.custom("unknown"))
    }
}
