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

/// An event streamed from an inference engine while generating a response.
public enum StreamEvent: Sendable, Hashable {
    /// Incremental response text.
    case textDelta(String)
    /// Incremental reasoning ("thinking") text.
    case thinkingDelta(String)
    /// A completed tool call. Argument deltas are not streamed; calls are
    /// emitted whole.
    case toolCall(ToolCall)
    /// Token usage, reported when the provider reports it (usually once the
    /// counts are final).
    case usage(Usage)
    /// The terminal event of a response.
    case completed(Completion)
}

/// Token usage reported by a provider.
public struct Usage: Sendable, Hashable, Codable {
    public var inputTokens: Int?
    public var outputTokens: Int?
    public var cacheReadTokens: Int?
    public var cacheWriteTokens: Int?
    public var reasoningTokens: Int?

    public init(
        inputTokens: Int? = nil,
        outputTokens: Int? = nil,
        cacheReadTokens: Int? = nil,
        cacheWriteTokens: Int? = nil,
        reasoningTokens: Int? = nil
    ) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.cacheReadTokens = cacheReadTokens
        self.cacheWriteTokens = cacheWriteTokens
        self.reasoningTokens = reasoningTokens
    }
}

/// Why generation finished.
public enum FinishReason: String, Sendable, Hashable, Codable {
    case stop
    case length
    case toolCalls
    case contentFilter
    case other
}

/// The terminal payload of a streamed response.
public struct Completion: Sendable, Hashable, Codable {
    public var finishReason: FinishReason?
    public var usage: Usage?
    /// Provider-assigned response identifier, when available.
    public var responseID: String?
    /// The model the server reports it used, when available.
    public var model: String?
    /// The complete assistant turn reconstructed from the stream, suitable for
    /// appending to conversation history. Includes provider round-trip data
    /// (thinking signatures, tool call ids) that delta events alone cannot
    /// reconstruct.
    public var assistantMessage: Message?

    public init(
        finishReason: FinishReason? = nil,
        usage: Usage? = nil,
        responseID: String? = nil,
        model: String? = nil,
        assistantMessage: Message? = nil
    ) {
        self.finishReason = finishReason
        self.usage = usage
        self.responseID = responseID
        self.model = model
        self.assistantMessage = assistantMessage
    }
}
