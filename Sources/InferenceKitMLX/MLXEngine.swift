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
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

import InferenceKit

/// Local inference with MLX on Apple Silicon, via `mlx-swift-lm`.
///
/// The engine owns a loaded ``MLXLMCommon/ModelContainer``; each call to
/// ``stream(_:)`` renders the caller's full history into a fresh
/// `ChatSession` (the engine itself stays stateless, matching remote
/// engines — at the cost of re-prefilling the history per request).
///
/// Loading downloads from Hugging Face by default:
///
///     let engine = try await MLXEngine.load("mlx-community/Llama-3.2-1B-Instruct-4bit")
///     let client = InferenceClient(engine: engine)
///
/// Text and tool calling are supported. Vision models and reasoning budgets
/// are out of scope for v1; images and files are rejected, and thinking
/// blocks in history are dropped (MLX chat templates do not resend them).
public struct MLXEngine: InferenceEngine {
    public let model: InferenceKit.ModelConfiguration

    /// The loaded model. Exposed for callers that want to drive
    /// `ChatSession` directly.
    public let container: ModelContainer

    /// Wraps an already-loaded container (for example one produced by
    /// `MLXVLM` or a custom factory).
    public init(container: ModelContainer, model: InferenceKit.ModelConfiguration) {
        self.container = container
        self.model = model
    }

    /// Loads a model from a Hugging Face repository (e.g.
    /// `"mlx-community/Llama-3.2-1B-Instruct-4bit"`), downloading weights
    /// via the default Hub integration.
    ///
    /// - Parameters:
    ///   - id: repository id, in `organization/model` form.
    ///   - model: the InferenceKit model configuration describing the loaded
    ///     model. Defaults to a custom configuration named after the repo id.
    ///   - revision: the revision to download.
    ///   - progressHandler: download progress callback.
    public static func load(
        _ id: String,
        model: InferenceKit.ModelConfiguration? = nil,
        revision: String = "main",
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> MLXEngine {
        let container = try await loadModelContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            id: id,
            revision: revision,
            progressHandler: progressHandler
        )
        return MLXEngine(
            container: container,
            model: model ?? .custom(id, displayName: id)
        )
    }

    /// Loads a model from a local directory of configuration and weights.
    public static func load(
        directory: URL,
        model: InferenceKit.ModelConfiguration? = nil
    ) async throws -> MLXEngine {
        let container = try await loadModelContainer(
            from: directory,
            using: #huggingFaceTokenizerLoader()
        )
        return MLXEngine(
            container: container,
            model: model ?? .custom(directory.lastPathComponent, displayName: directory.lastPathComponent)
        )
    }

    public func stream(_ request: InferenceRequest) -> AsyncThrowingStream<StreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let container = self.container
            let model = self.model
            let task = Task {
                do {
                    let chatMessages = try Self.chatMessages(from: request.messages)
                    let parameters = try Self.generateParameters(
                        from: request.options,
                        model: model
                    )
                    let tools = try Self.toolSpecs(from: request.tools)

                    // A fresh session per request: engines are stateless and
                    // the caller owns the history.
                    let session = ChatSession(
                        container,
                        instructions: request.system.isEmpty ? nil : request.system.joinedPrompt,
                        generateParameters: parameters
                    )
                    session.tools = tools.isEmpty ? nil : tools

                    var text = ""
                    var blocks: [ContentBlock] = []

                    for try await generation in session.streamDetails(to: chatMessages) {
                        switch generation {
                        case .chunk(let chunk):
                            if !chunk.isEmpty {
                                text += chunk
                                blocks.append(.text(chunk))
                                continuation.yield(.textDelta(chunk))
                            }

                        case .toolCall(let call):
                            let event = Self.toolCall(from: call)
                            continuation.yield(.toolCall(event))
                            blocks.append(.toolCall(ToolCallBlock(
                                id: event.id,
                                name: event.name,
                                arguments: event.arguments
                            )))

                        case .rejectedToolCall(let rejection):
                            throw InferenceError.server(
                                "The model emitted an unusable tool call (\(rejection.reason.rawValue))"
                                    + (rejection.detail.map { ": \($0)" } ?? "")
                                    + "."
                            )

                        case .info(let info):
                            let usage = Usage(
                                inputTokens: info.promptTokenCount + info.cachedPromptTokenCount,
                                outputTokens: info.generationTokenCount
                            )
                            continuation.yield(.usage(usage))

                            // Merge adjacent text blocks into one.
                            var content: [ContentBlock] = []
                            for block in blocks {
                                if case .text(let existing) = content.last,
                                    case .text(let newText) = block
                                {
                                    content[content.count - 1] = .text(existing + newText)
                                } else {
                                    content.append(block)
                                }
                            }

                            continuation.yield(.completed(Completion(
                                finishReason: Self.finishReason(info.stopReason),
                                usage: usage,
                                model: model.id,
                                assistantMessage: content.isEmpty
                                    ? nil
                                    : InferenceKit.Message(role: .assistant, content: content)
                            )))
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    // MARK: - Mapping

    private static func finishReason(_ raw: GenerateStopReason) -> FinishReason {
        switch raw {
        case .stop: .stop
        case .length: .length
        case .cancelled: .other
        }
    }

    /// Canonical messages to MLX chat messages. System prompts are handled by
    /// the session's `instructions`; thinking blocks are dropped; each tool
    /// result becomes its own MLX tool message.
    private static func chatMessages(from messages: [InferenceKit.Message]) throws -> [Chat.Message] {
        var result: [Chat.Message] = []

        for message in messages {
            switch message.role {
            case .user:
                var text = ""
                for block in message.content {
                    switch block {
                    case .text(let value):
                        text += value
                    case .image, .file:
                        throw InferenceError.unsupportedFeature(
                            "Image and file inputs on MLX (vision models are out of scope for v1)",
                            context: "the MLX engine"
                        )
                    case .thinking, .toolCall, .toolResult:
                        throw InferenceError.invalidConfiguration(
                            "Block is not valid in a user message."
                        )
                    }
                }
                result.append(.user(text))

            case .assistant:
                var text = ""
                var toolCalls: [MLXLMCommon.ToolCall] = []
                for block in message.content {
                    switch block {
                    case .text(let value):
                        text += value
                    case .thinking:
                        continue
                    case .toolCall(let call):
                        toolCalls.append(try mlxToolCall(from: call))
                    case .image, .file:
                        throw InferenceError.unsupportedFeature(
                            "Attachments in assistant messages",
                            context: "the MLX engine"
                        )
                    case .toolResult:
                        throw InferenceError.invalidConfiguration(
                            "Tool results belong in tool messages."
                        )
                    }
                }
                result.append(.assistant(text, toolCalls: toolCalls.isEmpty ? nil : toolCalls))

            case .tool:
                for block in message.content {
                    guard case .toolResult(let toolResult) = block else {
                        throw InferenceError.invalidConfiguration(
                            "Tool messages may only contain tool results."
                        )
                    }
                    result.append(.tool(
                        toolResult.output,
                        id: toolResult.callID,
                        name: toolResult.name
                    ))
                }
            }
        }

        return result
    }

    private static func mlxToolCall(from call: ToolCallBlock) throws -> MLXLMCommon.ToolCall {
        let arguments: [String: MLXLMCommon.JSONValue]
        if let data = call.arguments.data(using: .utf8) {
            arguments = (try? JSONDecoder().decode([String: MLXLMCommon.JSONValue].self, from: data)) ?? [:]
        } else {
            arguments = [:]
        }
        return MLXLMCommon.ToolCall(
            function: .init(name: call.name, arguments: arguments),
            id: call.id.isEmpty ? nil : call.id
        )
    }

    private static func toolCall(from call: MLXLMCommon.ToolCall) -> InferenceKit.ToolCall {
        let arguments: String
        if let data = try? JSONEncoder().encode(call.function.arguments),
            let string = String(data: data, encoding: .utf8)
        {
            arguments = string
        } else {
            arguments = "{}"
        }
        return ToolCall(
            id: call.id ?? UUID().uuidString,
            name: call.function.name,
            arguments: arguments
        )
    }

    private static func toolSpecs(from tools: [ToolDefinition]) throws -> [ToolSpec] {
        try tools.map { tool in
            [
                "type": "function",
                "function": [
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": sendableValue(tool.parametersSchema),
                ] as [String: any Sendable],
            ] as ToolSpec
        }
    }

    /// Converts a JSON value to the plain Sendable tree MLX's tool layer
    /// consumes.
    private static func sendableValue(_ json: InferenceKit.JSONValue) -> any Sendable {
        switch json {
        case .null: return false as Bool?
        case .bool(let value): return value
        case .int(let value): return Int(value)
        case .double(let value): return value
        case .string(let value): return value
        case .array(let values): return values.map { sendableValue($0) }
        case .object(let values): return values.mapValues { sendableValue($0) }
        }
    }

    private static func generateParameters(
        from options: GenerationOptions,
        model: InferenceKit.ModelConfiguration
    ) throws -> GenerateParameters {
        if let stopSequences = options.stopSequences, !stopSequences.isEmpty {
            throw InferenceError.unsupportedFeature("Stop sequences on MLX", context: "the MLX engine")
        }
        if case .budgetedTokens = options.thinking {
            throw InferenceError.unsupportedFeature("Reasoning token budgets on MLX", context: "the MLX engine")
        }
        if case .effort = options.thinking {
            throw InferenceError.unsupportedFeature("Reasoning effort on MLX", context: "the MLX engine")
        }

        return GenerateParameters(
            maxTokens: options.maxOutputTokens,
            maxKVSize: options.contextWindowTokens,
            temperature: options.temperature.map { Float($0) } ?? 0.6,
            topP: options.topP.map { Float($0) } ?? 1.0
        )
    }
}
