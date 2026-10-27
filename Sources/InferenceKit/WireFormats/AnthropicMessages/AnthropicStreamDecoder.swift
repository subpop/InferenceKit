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

/// Decodes the Anthropic Messages SSE stream. Reconstructs the assistant
/// turn — including thinking signatures and tool use blocks — for the
/// completion event.
///
/// Redacted thinking blocks are acknowledged but not decoded or round-tripped.
final class AnthropicStreamDecoder: WireStreamDecoder {
    private struct ToolBuffer {
        let id: String
        let name: String
        var arguments: String
    }

    private var messageID: String?
    private var modelName: String?
    private var usage = Usage()
    private var stopReason: String?
    private var toolBuffers: [Int: ToolBuffer] = [:]
    private var thinkingText: [Int: String] = [:]
    private var thinkingSignatures: [Int: String] = [:]
    private var textBuffers: [Int: String] = [:]
    private var completedBlocks: [(index: Int, block: ContentBlock)] = []
    private var sawCompletion = false

    init() {}

    func feed(_ payload: String) throws -> [StreamEvent] {
        guard payload != "[DONE]",
              let data = payload.data(using: .utf8),
              let event = try? JSONDecoder().decode(AnthropicStreamEvent.self, from: data)
        else {
            return []
        }

        switch event.type {
        case "message_start":
            messageID = event.message?.id
            modelName = event.message?.model
            if let summary = event.message?.usage {
                usage.inputTokens = summary.input_tokens
                usage.cacheReadTokens = summary.cache_read_input_tokens
                usage.cacheWriteTokens = summary.cache_creation_input_tokens
                return [.usage(usage)]
            }
            return []

        case "content_block_start":
            if let index = event.index, let block = event.content_block {
                switch block.type {
                case "tool_use":
                    toolBuffers[index] = ToolBuffer(
                        id: block.id ?? UUID().uuidString,
                        name: block.name ?? "",
                        arguments: ""
                    )
                case "thinking":
                    thinkingText[index] = ""
                default:
                    break
                }
            }
            return []

        case "content_block_delta":
            guard let index = event.index, let delta = event.delta else { return [] }
            switch delta.type {
            case "text_delta":
                guard let text = delta.text else { return [] }
                textBuffers[index, default: ""] += text
                return [.textDelta(text)]
            case "thinking_delta":
                guard let text = delta.thinking else { return [] }
                thinkingText[index, default: ""] += text
                return [.thinkingDelta(text)]
            case "signature_delta":
                if let signature = delta.signature {
                    thinkingSignatures[index, default: ""] += signature
                }
                return []
            case "input_json_delta":
                if let partial = delta.partial_json {
                    toolBuffers[index]?.arguments += partial
                }
                return []
            default:
                return []
            }

        case "content_block_stop":
            guard let index = event.index else { return [] }
            var events: [StreamEvent] = []
            if let text = thinkingText.removeValue(forKey: index) {
                completedBlocks.append(
                    (index, .thinking(ThinkingBlock(
                        text: text,
                        signature: thinkingSignatures.removeValue(forKey: index)
                    )))
                )
            }
            if let text = textBuffers.removeValue(forKey: index) {
                completedBlocks.append((index, .text(text)))
            }
            if let tool = toolBuffers.removeValue(forKey: index) {
                events.append(.toolCall(ToolCall(id: tool.id, name: tool.name, arguments: tool.arguments)))
                completedBlocks.append(
                    (index, .toolCall(ToolCallBlock(id: tool.id, name: tool.name, arguments: tool.arguments)))
                )
            }
            return events

        case "message_delta":
            if let reason = event.delta?.stop_reason {
                stopReason = reason
            }
            if let summary = event.usage {
                usage.inputTokens = summary.input_tokens ?? usage.inputTokens
                usage.outputTokens = summary.output_tokens ?? usage.outputTokens
                usage.cacheReadTokens = summary.cache_read_input_tokens ?? usage.cacheReadTokens
                usage.cacheWriteTokens = summary.cache_creation_input_tokens ?? usage.cacheWriteTokens
                return [.usage(usage)]
            }
            return []

        case "message_stop":
            sawCompletion = true
            return [.completed(completion())]

        case "error":
            throw InferenceError.server(
                event.error?.message ?? event.error?.type ?? "Unknown Anthropic stream error."
            )

        default:
            // ping and future event types.
            return []
        }
    }

    func finish() throws -> [StreamEvent] {
        guard !sawCompletion else { return [] }
        var events = flushUnterminatedBlocks()
        events.append(.completed(completion()))
        return events
    }

    /// Records blocks that never received `content_block_stop`, so truncated
    /// streams still reconstruct their partial content. Tool buffers also
    /// emit their calls.
    private func flushUnterminatedBlocks() -> [StreamEvent] {
        var events: [StreamEvent] = []
        let indexes = Set(thinkingText.keys)
            .union(textBuffers.keys)
            .union(toolBuffers.keys)
            .sorted()
        for index in indexes {
            if let text = thinkingText.removeValue(forKey: index) {
                completedBlocks.append(
                    (index, .thinking(ThinkingBlock(
                        text: text,
                        signature: thinkingSignatures.removeValue(forKey: index)
                    )))
                )
            }
            if let text = textBuffers.removeValue(forKey: index) {
                completedBlocks.append((index, .text(text)))
            }
            if let tool = toolBuffers.removeValue(forKey: index) {
                events.append(.toolCall(ToolCall(id: tool.id, name: tool.name, arguments: tool.arguments)))
                completedBlocks.append(
                    (index, .toolCall(ToolCallBlock(id: tool.id, name: tool.name, arguments: tool.arguments)))
                )
            }
        }
        return events
    }

    private func completion() -> Completion {
        Completion(
            finishReason: Self.finishReason(stopReason),
            usage: usage,
            responseID: messageID,
            model: modelName,
            assistantMessage: assistantMessage()
        )
    }

    private func assistantMessage() -> Message? {
        let blocks = completedBlocks.sorted { $0.index < $1.index }.map(\.block)
        guard !blocks.isEmpty else { return nil }
        return Message(role: .assistant, content: blocks)
    }

    private static func finishReason(_ raw: String?) -> FinishReason? {
        guard let raw else { return nil }
        switch raw {
        case "end_turn", "stop_sequence": return .stop
        case "max_tokens": return .length
        case "tool_use": return .toolCalls
        case "refusal": return .contentFilter
        default: return .other
        }
    }
}
