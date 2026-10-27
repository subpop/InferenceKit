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

/// Decodes the OpenAI Responses SSE stream. The assistant turn is rebuilt from
/// the `response.completed` output array when present, and from accumulated
/// buffers otherwise.
final class OpenAIResponsesStreamDecoder: WireStreamDecoder {
    private struct ToolBuffer {
        var callID: String
        var name: String
        var arguments: String
    }

    private var toolBuffers: [String: ToolBuffer] = [:]
    private var reasoningText: [String: String] = [:]
    private var textBuffers: [String: String] = [:]
    private var sawCompletion = false

    init() {}

    func feed(_ payload: String) throws -> [StreamEvent] {
        guard payload != "[DONE]",
              let data = payload.data(using: .utf8),
              let event = try? JSONDecoder().decode(OpenAIResponsesEvent.self, from: data)
        else {
            return []
        }

        switch event.type {
        case "response.output_item.added":
            if let item = event.item {
                switch item.type {
                case "function_call":
                    // Argument deltas are keyed by item id; the call id rides
                    // in the buffer until the call is emitted.
                    if let itemID = item.id {
                        toolBuffers[itemID] = ToolBuffer(
                            callID: item.call_id ?? itemID,
                            name: item.name ?? "",
                            arguments: ""
                        )
                    }
                case "reasoning":
                    if let itemID = item.id {
                        reasoningText[itemID] = ""
                    }
                default:
                    break
                }
            }
            return []

        case "response.output_text.delta":
            guard let text = event.delta else { return [] }
            if let itemID = event.item_id {
                textBuffers[itemID, default: ""] += text
            } else {
                textBuffers["", default: ""] += text
            }
            return [.textDelta(text)]

        case "response.reasoning_summary_text.delta":
            guard let text = event.delta else { return [] }
            if let itemID = event.item_id {
                reasoningText[itemID, default: ""] += text
            } else {
                reasoningText["", default: ""] += text
            }
            return [.thinkingDelta(text)]

        case "response.function_call_arguments.delta":
            if let itemID = event.item_id, let text = event.delta {
                toolBuffers[itemID]?.arguments += text
            }
            return []

        case "response.function_call_arguments.done":
            guard let itemID = event.item_id, let buffer = toolBuffers.removeValue(forKey: itemID) else {
                return []
            }
            var final = buffer
            if let arguments = event.arguments {
                final.arguments = arguments
            }
            return [.toolCall(ToolCall(id: final.callID, name: final.name, arguments: final.arguments))]

        case "response.completed":
            sawCompletion = true
            if let response = event.response {
                var events: [StreamEvent] = []
                var usage = Usage()
                if let summary = response.usage {
                    usage.inputTokens = summary.input_tokens
                    usage.outputTokens = summary.output_tokens
                    usage.reasoningTokens = summary.output_tokens_details?.reasoning_tokens
                    events.append(.usage(usage))
                }
                let outputBlocks = Self.contentBlocks(from: response.output)
                events.append(.completed(Completion(
                    finishReason: Self.finishReason(response: response, outputBlocks: outputBlocks),
                    usage: usage,
                    responseID: response.id,
                    model: response.model,
                    assistantMessage: outputBlocks.isEmpty
                        ? nil
                        : Message(role: .assistant, content: outputBlocks)
                )))
                return events
            }
            return [.completed(Completion(assistantMessage: accumulatedMessage()))]

        case "response.failed":
            if let message = event.response?.error?.message {
                throw InferenceError.server(message)
            }
            throw InferenceError.server("The response failed.")

        case "error":
            throw InferenceError.server(
                event.error?.message ?? event.error?.code ?? "Unknown OpenAI stream error."
            )

        default:
            return []
        }
    }

    func finish() throws -> [StreamEvent] {
        guard !sawCompletion else { return [] }
        var events: [StreamEvent] = []
        var sawToolCalls = false
        for (_, buffer) in toolBuffers {
            events.append(.toolCall(ToolCall(id: buffer.callID, name: buffer.name, arguments: buffer.arguments)))
            sawToolCalls = true
        }
        events.append(.completed(Completion(
            finishReason: sawToolCalls ? .toolCalls : .stop,
            assistantMessage: accumulatedMessage()
        )))
        return events
    }

    /// Reconstructs the assistant turn from the deltas seen so far, for
    /// streams that ended without `response.completed`.
    private func accumulatedMessage() -> Message? {
        var blocks: [ContentBlock] = []
        for (_, text) in reasoningText {
            blocks.append(.thinking(ThinkingBlock(text: text)))
        }
        for (_, text) in textBuffers {
            blocks.append(.text(text))
        }
        for (_, buffer) in toolBuffers {
            blocks.append(.toolCall(ToolCallBlock(id: buffer.callID, name: buffer.name, arguments: buffer.arguments)))
        }
        guard !blocks.isEmpty else { return nil }
        return Message(role: .assistant, content: blocks)
    }

    // MARK: - Output item decoding

    private static func contentBlocks(
        from output: [OpenAIResponsesEvent.ResponseSummary.OutputItem]?
    ) -> [ContentBlock] {
        guard let output else { return [] }
        var blocks: [ContentBlock] = []
        for item in output {
            switch item.type {
            case "reasoning":
                let text = (item.summary ?? [])
                    .compactMap(\.text)
                    .joined()
                blocks.append(.thinking(ThinkingBlock(text: text, signature: item.id)))
            case "message":
                let text = (item.content ?? [])
                    .filter { $0.type == "output_text" }
                    .compactMap(\.text)
                    .joined()
                if !text.isEmpty {
                    blocks.append(.text(text))
                }
            case "function_call":
                if let callID = item.call_id, let name = item.name {
                    blocks.append(.toolCall(ToolCallBlock(
                        id: callID,
                        name: name,
                        arguments: item.arguments ?? "{}"
                    )))
                }
            default:
                break
            }
        }
        return blocks
    }

    private static func finishReason(
        response: OpenAIResponsesEvent.ResponseSummary,
        outputBlocks: [ContentBlock]
    ) -> FinishReason {
        if response.status == "incomplete" {
            return .length
        }
        if outputBlocks.contains(where: { if case .toolCall = $0 { return true } else { return false } }) {
            return .toolCalls
        }
        return .stop
    }
}
