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

/// Decodes the Ollama NDJSON stream. Tool calls arrive as complete objects;
/// Ollama does not assign call ids, so one is generated per call.
final class OllamaStreamDecoder: WireStreamDecoder {
    private var usage = Usage()
    private var finishReason: String?
    private var text = ""
    private var thinking = ""
    private var blocks: [ContentBlock] = []
    private var sawCompletion = false

    init() {}

    func feed(_ payload: String) throws -> [StreamEvent] {
        guard let data = payload.data(using: .utf8),
              let chunk = try? JSONDecoder().decode(OllamaStreamChunk.self, from: data)
        else {
            return []
        }

        var events: [StreamEvent] = []

        if let message = chunk.message {
            if let reasoning = message.thinking, !reasoning.isEmpty {
                thinking += reasoning
                events.append(.thinkingDelta(reasoning))
            }
            if let content = message.content, !content.isEmpty {
                text += content
                events.append(.textDelta(content))
            }
            for call in message.tool_calls ?? [] {
                guard let function = call.function, let name = function.name else { continue }
                let arguments = function.arguments?.jsonString ?? "{}"
                let id = UUID().uuidString
                events.append(.toolCall(ToolCall(id: id, name: name, arguments: arguments)))
                blocks.append(.toolCall(ToolCallBlock(id: id, name: name, arguments: arguments)))
            }
        }

        if chunk.done == true {
            usage.inputTokens = chunk.prompt_eval_count
            usage.outputTokens = chunk.eval_count
            finishReason = chunk.done_reason ?? finishReason
            sawCompletion = true
            events.append(.usage(usage))
            events.append(.completed(Completion(
                finishReason: Self.finishReason(finishReason),
                usage: usage,
                assistantMessage: assistantMessage()
            )))
        }
        return events
    }

    func finish() throws -> [StreamEvent] {
        guard !sawCompletion else { return [] }
        return [.completed(Completion(
            finishReason: Self.finishReason(finishReason),
            usage: usage,
            assistantMessage: assistantMessage()
        ))]
    }

    private func assistantMessage() -> Message? {
        var content: [ContentBlock] = []
        if !thinking.isEmpty {
            content.append(.thinking(ThinkingBlock(text: thinking)))
        }
        if !text.isEmpty {
            content.append(.text(text))
        }
        content.append(contentsOf: blocks)
        guard !content.isEmpty else { return nil }
        return Message(role: .assistant, content: content)
    }

    private static func finishReason(_ raw: String?) -> FinishReason? {
        guard let raw else { return nil }
        switch raw {
        case "stop": return .stop
        case "length": return .length
        default: return .other
        }
    }
}
