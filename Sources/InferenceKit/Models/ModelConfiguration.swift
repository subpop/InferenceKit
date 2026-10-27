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

/// Input content kinds a model accepts.
public enum InputModality: String, Sendable, Hashable, Codable, CaseIterable {
    case text
    case image
    case pdf
}

/// A model's identity and capabilities. Instances come from three sources:
/// the bundled catalog (``ModelCatalog``), provider discovery, or your own
/// definitions for custom and fine-tuned models.
public struct ModelConfiguration: Sendable, Hashable, Codable, Identifiable {
    /// The canonical model identifier, as used in conversation history.
    public var id: String
    public var displayName: String?
    /// The identifier to send on the wire. Falls back to `id` when nil
    /// (e.g. `"claude-sonnet-4-5"` on the wire, `"claude-sonnet-4-5-20250929"`
    /// as `providerModelID`).
    public var providerModelID: String?
    /// The model's context window, when known.
    public var contextWindowTokens: Int?
    /// The maximum tokens the model can generate, when known. Used as the
    /// default for `maxOutputTokens` where the provider requires one.
    public var maxOutputTokens: Int?
    /// Reasoning capabilities, when known.
    public var thinking: ThinkingCapability?
    /// Supported input content kinds. Nil means unknown.
    public var inputModalities: Set<InputModality>?

    public init(
        id: String,
        displayName: String? = nil,
        providerModelID: String? = nil,
        contextWindowTokens: Int? = nil,
        maxOutputTokens: Int? = nil,
        thinking: ThinkingCapability? = nil,
        inputModalities: Set<InputModality>? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.providerModelID = providerModelID
        self.contextWindowTokens = contextWindowTokens
        self.maxOutputTokens = maxOutputTokens
        self.thinking = thinking
        self.inputModalities = inputModalities
    }

    /// The identifier sent on the wire.
    public var wireID: String { providerModelID ?? id }

    /// A user-defined model configuration with unknown capabilities.
    public static func custom(
        _ id: String,
        displayName: String? = nil,
        providerModelID: String? = nil,
        contextWindowTokens: Int? = nil,
        maxOutputTokens: Int? = nil,
        thinking: ThinkingCapability? = nil,
        inputModalities: Set<InputModality>? = nil
    ) -> ModelConfiguration {
        ModelConfiguration(
            id: id,
            displayName: displayName,
            providerModelID: providerModelID,
            contextWindowTokens: contextWindowTokens,
            maxOutputTokens: maxOutputTokens,
            thinking: thinking,
            inputModalities: inputModalities
        )
    }
}
