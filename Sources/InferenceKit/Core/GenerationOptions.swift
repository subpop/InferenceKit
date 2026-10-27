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

/// How the model should reason ("think") during generation. Each wire format
/// maps what it can and throws ``InferenceError/unsupportedFeature(_:context:)`` for
/// what it cannot.
public enum ThinkingConfiguration: Sendable, Hashable, Codable {
    /// Disable reasoning where the provider supports doing so explicitly.
    case disabled
    /// Let the provider decide (Anthropic adaptive thinking, Ollama `think`,
    /// OpenAI reasoning summaries on, Gemini default behavior).
    case automatic
    /// Request a reasoning token budget (Anthropic `budget_tokens`, Gemini
    /// `thinkingBudget`, Ollama `think.budget`).
    case budgetedTokens(Int)
    /// Request a reasoning effort level (OpenAI `reasoning_effort`,
    /// Anthropic `output_config.effort`).
    case effort(Effort)

    /// Reasoning effort levels, ascending. `none` sits below the ladder: it
    /// asks the model not to reason at all, where the model supports it.
    public enum Effort: String, Sendable, Hashable, Codable, CaseIterable {
        /// No reasoning at all (OpenAI effort `none`; GPT-6 Sol and Luna).
        /// Unlike ``ThinkingConfiguration/disabled``, this is an explicit
        /// wire value rather than an omitted parameter. In contexts typed
        /// `Effort?`, plain `.none` means `Optional.none` — write
        /// `Effort.none` there.
        case none
        case minimal
        case low
        case medium
        case high
        case xhigh
        case max
    }
}

/// What a model supports on the reasoning axis. Used to validate requests
/// early and to supply defaults. A `nil` capability means the model is unknown
/// and no validation is performed.
public struct ThinkingCapability: Sendable, Hashable, Codable {
    /// Supported reasoning token budget range, if the model accepts budgets.
    public var budgetRange: ClosedRange<Int>?
    /// The budget the model uses when thinking is on and no budget is given.
    public var defaultBudgetTokens: Int?
    /// Supported effort levels, for models that use effort rather than budgets.
    public var supportedEfforts: [ThinkingConfiguration.Effort]?
    /// True when the model's thinking cannot be turned off (Claude Opus 5.5,
    /// Claude Sonnet 5.5). Requests to disable thinking are rejected.
    public var alwaysOn: Bool?

    public init(
        budgetRange: ClosedRange<Int>? = nil,
        defaultBudgetTokens: Int? = nil,
        supportedEfforts: [ThinkingConfiguration.Effort]? = nil,
        alwaysOn: Bool? = nil
    ) {
        self.budgetRange = budgetRange
        self.defaultBudgetTokens = defaultBudgetTokens
        self.supportedEfforts = supportedEfforts
        self.alwaysOn = alwaysOn
    }
}

/// Automatic context management. OpenAI Responses maps this to compaction
/// with a trigger threshold; Anthropic maps it to context editing; other
/// formats treat it as unsupported.
public struct Compaction: Sendable, Hashable, Codable {
    /// The input-token count at which the provider compacts the conversation.
    public var triggerThresholdTokens: Int

    public init(triggerThresholdTokens: Int) {
        self.triggerThresholdTokens = triggerThresholdTokens
    }
}

/// Normalized generation options. Fields a wire format cannot express either
/// throw ``InferenceError/unsupportedFeature(_:context:)`` (semantics-changing, e.g.
/// stop sequences) or are silently ignored where they are only advisory
/// (e.g. context window hints).
public struct GenerationOptions: Sendable, Hashable, Codable {
    public var temperature: Double?
    public var topP: Double?
    /// Maximum tokens the model may generate. Anthropic requires this; when
    /// unset, the model configuration's `maxOutputTokens` is used.
    public var maxOutputTokens: Int?
    public var stopSequences: [String]?
    public var thinking: ThinkingConfiguration?
    /// Advisory context window size for engines that size local state from it
    /// (Ollama `num_ctx`, MLX). Ignored by hosted APIs.
    public var contextWindowTokens: Int?
    public var compaction: Compaction?

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        maxOutputTokens: Int? = nil,
        stopSequences: [String]? = nil,
        thinking: ThinkingConfiguration? = nil,
        contextWindowTokens: Int? = nil,
        compaction: Compaction? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.maxOutputTokens = maxOutputTokens
        self.stopSequences = stopSequences
        self.thinking = thinking
        self.contextWindowTokens = contextWindowTokens
        self.compaction = compaction
    }

    /// Validates the options against a model's known capabilities. Unknown
    /// capabilities (nil fields) are not validated.
    public func validate(against model: ModelConfiguration) throws {
        guard let thinking else { return }
        guard let capability = model.thinking else { return }

        if case .budgetedTokens(let budget) = thinking, let range = capability.budgetRange {
            guard range.contains(budget) else {
                throw InferenceError.invalidConfiguration(
                    "Thinking budget \(budget) is outside the supported range "
                        + "\(range.lowerBound)–\(range.upperBound) for model \"\(model.id)\"."
                )
            }
        }
        if case .effort(let effort) = thinking, let supported = capability.supportedEfforts {
            guard supported.contains(effort) else {
                throw InferenceError.invalidConfiguration(
                    "Reasoning effort \"\(effort.rawValue)\" is not supported by model "
                        + "\"\(model.id)\". Supported efforts: "
                        + "\(supported.map(\.rawValue).joined(separator: ", "))."
                )
            }
        }
        // Effort-era models (Claude 4.7 onward, OpenAI reasoning models)
        // reject token budgets.
        if case .budgetedTokens = thinking,
           capability.budgetRange == nil, capability.supportedEfforts != nil {
            throw InferenceError.invalidConfiguration(
                "Model \"\(model.id)\" does not accept thinking budgets. "
                    + "Use a reasoning effort instead."
            )
        }
        // Budget-era models (Claude 4.5, Gemini) reject efforts.
        if case .effort = thinking,
           capability.supportedEfforts == nil, capability.budgetRange != nil {
            throw InferenceError.invalidConfiguration(
                "Model \"\(model.id)\" does not accept reasoning efforts. "
                    + "Use a thinking budget instead."
            )
        }
        // Always-on thinking cannot be disabled.
        if case .disabled = thinking, capability.alwaysOn == true {
            throw InferenceError.invalidConfiguration(
                "Model \"\(model.id)\" cannot disable thinking."
            )
        }
    }
}
