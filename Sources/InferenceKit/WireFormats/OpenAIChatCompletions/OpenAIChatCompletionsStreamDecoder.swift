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

/// Decodes the Chat Completions SSE stream. Tool call arguments arrive as
/// indexed deltas; calls are flushed when the provider reports a finish
/// reason, or at stream end. The terminal completion event is emitted from
/// `finish()`, since Chat Completions has no end-of-response event.
final class OpenAIChatCompletionsStreamDecoder: WireStreamDecoder {
    private struct ToolBuffer {
        var id: String
        var name: String
        var arguments: String
    }

    private var responseID: String?
    private var modelName: String?
    private var usage = Usage()
    private var finishReason: String?
    private var text = ""
    private var thinking = ""
    private var toolBuffers: [Int: ToolBuffer] = [:]
    private var toolCallBlocks: [ContentBlock] = []

    init() {}

    func feed(_ payload: String) throws -> [StreamEvent] {
        guard payload != "[DONE]",
              let data = payload.data(using: .utf8),
              let chunk = try? JSONDecoder().decode(OpenAIChatChunk.self, from: data)
        else {
            return []
        }

        responseID = chunk.id ?? responseID
        modelName = chunk.model ?? modelName

        var events: [StreamEvent] = []

        if let summary = chunk.usage {
            usage.inputTokens = summary.prompt_tokens
            usage.outputTokens = summary.completion_tokens
            usage.reasoningTokens = summary.completion_tokens_details?.reasoning_tokens
            events.append(.usage(usage))
        }

        for choice in chunk.choices ?? [] {
            if let reason = choice.finish_reason {
                finishReason = reason
            }
            guard let delta = choice.delta else { continue }

            if let content = delta.content {
                text += content
                events.append(.textDelta(content))
            }
            // DeepSeek-style providers use `reasoning_content`; others use
            // `reasoning`.
            if let reasoning = delta.reasoning_content ?? delta.reasoning {
                thinking += reasoning
                events.append(.thinkingDelta(reasoning))
            }
            for call in delta.tool_calls ?? [] {
                let index = call.index ?? 0
                var buffer = toolBuffers[index]
                    ?? ToolBuffer(id: call.id ?? "call-\(index)", name: "", arguments: "")
                if let id = call.id {
                    buffer.id = id
                }
                if let name = call.function?.name {
                    buffer.name = name
                }
                if let arguments = call.function?.arguments {
                    buffer.arguments += arguments
                }
                toolBuffers[index] = buffer
            }
        }

        if finishReason != nil {
            events.append(contentsOf: flushToolCalls())
        }
        return events
    }

    func finish() throws -> [StreamEvent] {
        var events = flushToolCalls()
        events.append(.completed(Completion(
            finishReason: Self.finishReason(finishReason),
            usage: usage,
            responseID: responseID,
            model: modelName,
            assistantMessage: assistantMessage()
        )))
        return events
    }

    /// Emits buffered tool calls, recording them for the assistant message.
    private func flushToolCalls() -> [StreamEvent] {
        guard !toolBuffers.isEmpty else { return [] }
        var events: [StreamEvent] = []
        for index in toolBuffers.keys.sorted() {
            let buffer = toolBuffers.removeValue(forKey: index)!
            events.append(.toolCall(ToolCall(id: buffer.id, name: buffer.name, arguments: buffer.arguments)))
            toolCallBlocks.append(.toolCall(ToolCallBlock(
                id: buffer.id,
                name: buffer.name,
                arguments: buffer.arguments
            )))
        }
        return events
    }

    private func assistantMessage() -> Message? {
        var blocks: [ContentBlock] = []
        if !thinking.isEmpty {
            blocks.append(.thinking(ThinkingBlock(text: thinking)))
        }
        if !text.isEmpty {
            blocks.append(.text(text))
        }
        blocks.append(contentsOf: toolCallBlocks)
        guard !blocks.isEmpty else { return nil }
        return Message(role: .assistant, content: blocks)
    }

    private static func finishReason(_ raw: String?) -> FinishReason? {
        guard let raw else { return nil }
        switch raw {
        case "stop": return .stop
        case "length": return .length
        case "tool_calls": return .toolCalls
        case "content_filter": return .contentFilter
        default: return .other
        }
    }
}
