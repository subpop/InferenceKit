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

/// Decodes the Gemini SSE stream. Each data payload is a complete
/// GenerateContentResponse whose parts carry incremental content.
final class GeminiStreamDecoder: WireStreamDecoder {
    private var modelName: String?
    private var usage = Usage()
    private var finishReason: String?
    private var blocks: [ContentBlock] = []
    private var sawCompletion = false

    init() {}

    func feed(_ payload: String) throws -> [StreamEvent] {
        guard payload != "[DONE]",
              let data = payload.data(using: .utf8),
              let chunk = try? JSONDecoder().decode(GeminiStreamChunk.self, from: data)
        else {
            return []
        }

        if let error = chunk.error {
            throw InferenceError.server(
                error.message ?? error.status ?? "Unknown Gemini stream error."
            )
        }

        modelName = chunk.modelVersion ?? modelName
        if let metadata = chunk.usageMetadata {
            usage.inputTokens = metadata.promptTokenCount
            usage.outputTokens = metadata.candidatesTokenCount
            usage.reasoningTokens = metadata.thoughtsTokenCount
        }

        var events: [StreamEvent] = []

        for candidate in chunk.candidates ?? [] {
            if let reason = candidate.finishReason {
                finishReason = reason
            }
            for part in candidate.content?.parts ?? [] {
                if let functionCall = part.functionCall, let name = functionCall.name {
                    let id = functionCall.id ?? UUID().uuidString
                    let arguments = functionCall.args?.jsonString ?? "{}"
                    events.append(.toolCall(ToolCall(id: id, name: name, arguments: arguments)))
                    blocks.append(.toolCall(ToolCallBlock(id: id, name: name, arguments: arguments)))
                } else if let text = part.text {
                    let isThought = part.thought ?? false
                    if isThought {
                        events.append(.thinkingDelta(text))
                        appendThinking(text: text, signature: part.thoughtSignature)
                    } else {
                        events.append(.textDelta(text))
                        appendText(text)
                    }
                }
            }
        }

        if finishReason != nil, !sawCompletion {
            sawCompletion = true
            events.append(.usage(usage))
            events.append(.completed(Completion(
                finishReason: Self.finishReason(finishReason),
                usage: usage,
                model: modelName,
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
            model: modelName,
            assistantMessage: assistantMessage()
        ))]
    }

    private func assistantMessage() -> Message? {
        guard !blocks.isEmpty else { return nil }
        return Message(role: .assistant, content: blocks)
    }

    /// Consecutive text parts across chunks are one logical block.
    private func appendText(_ text: String) {
        if case .text(let existing)? = blocks.last {
            blocks[blocks.count - 1] = .text(existing + text)
        } else {
            blocks.append(.text(text))
        }
    }

    /// Consecutive thought parts with the same signature are one block.
    private func appendThinking(text: String, signature: String?) {
        if case .thinking(let existing)? = blocks.last, existing.signature == signature {
            blocks[blocks.count - 1] = .thinking(ThinkingBlock(text: existing.text + text, signature: signature))
        } else {
            blocks.append(.thinking(ThinkingBlock(text: text, signature: signature)))
        }
    }

    private static func finishReason(_ raw: String?) -> FinishReason? {
        guard let raw else { return nil }
        switch raw {
        case "STOP": return .stop
        case "MAX_TOKENS": return .length
        case "SAFETY", "RECITATION", "PROHIBITED_CONTENT": return .contentFilter
        default: return .other
        }
    }
}
