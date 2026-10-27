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

/// Known-model metadata, keyed by provider and model id. The bundled catalog
/// (``bundled``) is compiled into the package and supplies the capabilities —
/// context windows, thinking support, modalities — that model-list APIs omit.
public struct ModelCatalog: Sendable {
    /// Provider id → model id → configuration. Provider ids match the preset
    /// ids in ``ProviderConfiguration`` (`"anthropic"`, `"openai"`,
    /// `"openai-chat"`, `"google"`, `"ollama"`).
    public let providers: [String: [String: ModelConfiguration]]

    public init(providers: [String: [String: ModelConfiguration]] = [:]) {
        self.providers = providers
    }

    /// Looks up a model by canonical id or wire id (`providerModelID`).
    public func model(_ id: String, providerID: String) -> ModelConfiguration? {
        guard let models = providers[providerID] else { return nil }
        if let exact = models[id] {
            return exact
        }
        return models.values.first { $0.providerModelID == id }
    }

    /// Enriches a discovered model with catalog capabilities. A discovered id
    /// that matches the catalog keeps the discovered id as its wire id; an
    /// unknown id yields a configuration with unknown capabilities.
    public func resolve(_ discovered: DiscoveredModel, providerID: String) -> ModelConfiguration {
        if let entry = model(discovered.id, providerID: providerID) {
            var config = entry
            config.providerModelID = discovered.id
            return config
        }
        return ModelConfiguration(
            id: discovered.id,
            displayName: discovered.displayName
        )
    }
}

extension ModelCatalog: Decodable {}
extension ModelCatalog: Encodable {}

// MARK: - Flagship model constants

extension ModelConfiguration {
    /// The on-device Apple Foundation Models model. For use with
    /// `InferenceKitFoundationModels`.
    public static let foundationModels = ModelConfiguration(
        id: "apple-foundation-model",
        displayName: "Foundation Models (Apple Intelligence)"
    )

    private static func known(_ id: String, _ providerID: String) -> ModelConfiguration {
        ModelCatalog.bundled.model(id, providerID: providerID) ?? ModelConfiguration(id: id)
    }

    /// Claude Sonnet 5.5.
    public static var claudeSonnet5_5: ModelConfiguration { known("claude-sonnet-5-5", "anthropic") }
    /// Claude Haiku 5.5.
    public static var claudeHaiku5_5: ModelConfiguration { known("claude-haiku-5-5", "anthropic") }
    /// Claude Opus 5.5.
    public static var claudeOpus5_5: ModelConfiguration { known("claude-opus-5-5", "anthropic") }
    /// Claude Sonnet 4.5.
    public static var claudeSonnet4_5: ModelConfiguration { known("claude-sonnet-4-5", "anthropic") }
    /// Claude Haiku 4.5.
    public static var claudeHaiku4_5: ModelConfiguration { known("claude-haiku-4-5", "anthropic") }
    /// Claude Opus 4.5.
    public static var claudeOpus4_5: ModelConfiguration { known("claude-opus-4-5", "anthropic") }
    /// GPT-6 Astra.
    public static var gpt6Astra: ModelConfiguration { known("gpt-6-astra", "openai") }
    /// GPT-6.1 Sol.
    public static var gpt6_1Sol: ModelConfiguration { known("gpt-6.1-sol", "openai") }
    /// GPT-6 Sol.
    public static var gpt6Sol: ModelConfiguration { known("gpt-6-sol", "openai") }
    /// GPT-6 Luna.
    public static var gpt6Luna: ModelConfiguration { known("gpt-6-luna", "openai") }
    /// GPT-5.1.
    public static var gpt5_1: ModelConfiguration { known("gpt-5.1", "openai") }
    /// GPT-5 mini.
    public static var gpt5Mini: ModelConfiguration { known("gpt-5-mini", "openai") }
    /// Gemini 2.5 Flash.
    public static var gemini2_5Flash: ModelConfiguration { known("gemini-2.5-flash", "google") }
    /// Gemini 2.5 Pro.
    public static var gemini2_5Pro: ModelConfiguration { known("gemini-2.5-pro", "google") }
    /// Gemini 3 Pro Preview.
    public static var gemini3ProPreview: ModelConfiguration { known("gemini-3-pro-preview", "google") }
}
