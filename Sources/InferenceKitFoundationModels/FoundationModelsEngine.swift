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
import FoundationModels

import InferenceKit

/// Runs Apple's on-device Foundation Models behind the ``InferenceEngine``
/// protocol.
///
/// Foundation Models sessions are stateful and prompt-oriented, so this engine
/// creates a session per request and renders the conversation history into
/// the prompt. Tool calling is not supported: Foundation Models tools are
/// Swift-typed (`@Generable`) and cannot be bridged from JSON schemas. Images
/// and files are likewise ignored; only text content is sent.
@available(macOS 26.0, iOS 26.0, *)
public final class FoundationModelsEngine: InferenceEngine {
    public let model: ModelConfiguration

    public init(model: ModelConfiguration = .foundationModels) {
        self.model = model
    }

    public func stream(_ request: InferenceRequest) -> AsyncThrowingStream<StreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let availability = await MainActor.run {
                        SystemLanguageModel.default.availability
                    }
                    guard case .available = availability else {
                        if case .unavailable(let reason) = availability {
                            throw InferenceError.modelUnavailable("\(reason)")
                        }
                        throw InferenceError.modelUnavailable("Apple Intelligence is not available.")
                    }

                    let instructions = request.system.joinedPrompt
                    let prompt = Self.transcript(for: request.messages)

                    let session = await MainActor.run {
                        LanguageModelSession(instructions: instructions)
                    }

                    let stream = session.streamResponse(to: prompt)
                    var lastContent = ""
                    for try await snapshot in stream {
                        let newText = String(snapshot.content.dropFirst(lastContent.count))
                        if !newText.isEmpty {
                            lastContent = snapshot.content
                            continuation.yield(.textDelta(newText))
                        }
                    }

                    continuation.yield(.completed(Completion(
                        finishReason: .stop,
                        assistantMessage: lastContent.isEmpty
                            ? nil
                            : Message.assistant(text: lastContent)
                    )))
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

    /// Renders conversation history into a transcript prompt.
    private static func transcript(for messages: [Message]) -> String {
        messages.map { message in
            let text = message.content.map(Self.blockText).joined(separator: "\n")
            switch message.role {
            case .user:
                return "User: \(text)"
            case .assistant:
                return "Assistant: \(text)"
            case .tool:
                return "Tool result: \(text)"
            }
        }
        .joined(separator: "\n\n")
    }

    private static func blockText(_ block: ContentBlock) -> String {
        switch block {
        case .text(let text):
            return text
        case .toolResult(let result):
            return result.output
        case .thinking(let thinking):
            return thinking.text
        case .toolCall(let call):
            return "\(call.name)(\(call.arguments))"
        case .image, .file:
            return ""
        }
    }
}
