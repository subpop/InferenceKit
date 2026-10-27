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

import Testing
import Foundation

@testable import InferenceKit

@Suite("Model catalog")
struct ModelCatalogTests {
    @Test("Bundled catalog resolves known models")
    func bundledCatalog() {
        let catalog = ModelCatalog.bundled
        let sonnet = catalog.model("claude-sonnet-4-5", providerID: "anthropic")
        #expect(sonnet?.displayName == "Claude Sonnet 4.5")
        #expect(sonnet?.contextWindowTokens == 200_000)
        #expect(sonnet?.thinking?.budgetRange == 1024...64_000)
        #expect(sonnet?.inputModalities?.contains(.image) == true)

        // Wire-id lookup: the dated alias resolves to the same entry.
        let dated = catalog.model("claude-sonnet-4-5-20250929", providerID: "anthropic")
        #expect(dated?.id == "claude-sonnet-4-5")

        // Flagship constants.
        #expect(ModelConfiguration.gpt5_1.contextWindowTokens == 400_000)
        #expect(ModelConfiguration.gemini2_5Flash.thinking?.budgetRange?.upperBound == 24_576)

        // Cross-provider lookups do not leak.
        #expect(catalog.model("claude-sonnet-4-5", providerID: "openai") == nil)
    }

    @Test("Claude 5.5 and GPT-6 families carry their capabilities")
    func newGenerations() throws {
        let catalog = ModelCatalog.bundled

        let opus = try #require(catalog.model("claude-opus-5-5", providerID: "anthropic"))
        #expect(opus.contextWindowTokens == 1_000_000)
        #expect(opus.maxOutputTokens == 128_000)
        #expect(opus.thinking?.supportedEfforts == [.low, .medium, .high, .xhigh, .max])
        #expect(opus.thinking?.alwaysOn == true)
        // 5.5 ids are dateless; there is no dated snapshot to pin.
        #expect(opus.providerModelID == nil)
        #expect(opus.inputModalities == [.text, .image, .pdf])

        // Haiku 5.5 can disable thinking; only Opus and Sonnet cannot.
        let haiku = try #require(catalog.model("claude-haiku-5-5", providerID: "anthropic"))
        #expect(haiku.thinking?.alwaysOn == nil)

        let astra = try #require(catalog.model("gpt-6-astra", providerID: "openai"))
        #expect(astra.contextWindowTokens == 1_050_000)
        #expect(astra.maxOutputTokens == 128_000)
        #expect(astra.inputModalities == [.text, .image])

        // Sol and Luna accept effort none (reasoning fully off); Astra and
        // 6.1 Sol do not.
        let luna = try #require(catalog.model("gpt-6-luna", providerID: "openai"))
        #expect(luna.thinking?.supportedEfforts == [.none, .low, .medium, .high, .xhigh, .max])
        #expect(astra.thinking?.supportedEfforts?.contains(.none) == false)

        // Flagship constants resolve.
        #expect(ModelConfiguration.claudeOpus5_5.thinking?.alwaysOn == true)
        #expect(ModelConfiguration.claudeSonnet5_5.contextWindowTokens == 1_000_000)
        #expect(ModelConfiguration.gpt6Astra.thinking?.supportedEfforts?.contains(.max) == true)
        #expect(ModelConfiguration.gpt6Luna.id == "gpt-6-luna")
    }

    @Test("Thinking validation separates effort-era and budget-era models")
    func eraValidation() throws {
        let catalog = ModelCatalog.bundled
        let opus = try #require(catalog.model("claude-opus-5-5", providerID: "anthropic"))
        let sonnet45 = try #require(catalog.model("claude-sonnet-4-5", providerID: "anthropic"))
        let astra = try #require(catalog.model("gpt-6-astra", providerID: "openai"))

        // Effort-era models (Claude 4.7 onward) reject token budgets, and
        // always-on models reject disabling.
        #expect(throws: InferenceError.self) {
            try GenerationOptions(thinking: .budgetedTokens(2048)).validate(against: opus)
        }
        #expect(throws: InferenceError.self) {
            try GenerationOptions(thinking: .disabled).validate(against: opus)
        }
        try GenerationOptions(thinking: .effort(.xhigh)).validate(against: opus)

        // Budget-era models (Claude 4.5) reject efforts.
        #expect(throws: InferenceError.self) {
            try GenerationOptions(thinking: .effort(.high)).validate(against: sonnet45)
        }
        try GenerationOptions(thinking: .budgetedTokens(2048)).validate(against: sonnet45)

        // GPT-6 drops the minimal effort GPT-5.1 had.
        #expect(throws: InferenceError.self) {
            try GenerationOptions(thinking: .effort(.minimal)).validate(against: astra)
        }

        // Effort none is model-specific even within GPT-6.
        let luna = try #require(catalog.model("gpt-6-luna", providerID: "openai"))
        try GenerationOptions(thinking: .effort(.none)).validate(against: luna)
        #expect(throws: InferenceError.self) {
            try GenerationOptions(thinking: .effort(.none)).validate(against: astra)
        }
    }

    @Test("Discovered models merge with catalog capabilities")
    func discoveredResolution() {
        let catalog = ModelCatalog.bundled

        // A known id keeps catalog metadata and uses the discovered id on the wire.
        let known = catalog.resolve(
            DiscoveredModel(id: "claude-sonnet-4-5-20250929", displayName: "Ignored"),
            providerID: "anthropic"
        )
        #expect(known.id == "claude-sonnet-4-5")
        #expect(known.wireID == "claude-sonnet-4-5-20250929")
        #expect(known.contextWindowTokens == 200_000)

        // An unknown id resolves to a permissive configuration.
        let unknown = catalog.resolve(
            DiscoveredModel(id: "claude-future-9", displayName: "Claude Future 9"),
            providerID: "anthropic"
        )
        #expect(unknown.id == "claude-future-9")
        #expect(unknown.displayName == "Claude Future 9")
        #expect(unknown.wireID == "claude-future-9")
        #expect(unknown.thinking == nil)
        #expect(unknown.contextWindowTokens == nil)
    }

    @Test("Custom model configurations stand alone")
    func customModels() {
        let custom = ModelConfiguration.custom(
            "my-finetune",
            displayName: "My Fine-tune",
            providerModelID: "ft:abc123",
            contextWindowTokens: 16_384,
            thinking: ThinkingCapability(supportedEfforts: [.low, .high])
        )
        #expect(custom.wireID == "ft:abc123")
        #expect(custom.thinking?.supportedEfforts == [.low, .high])

        // Effort validation consults the custom capability.
        let options = GenerationOptions(thinking: .effort(.medium))
        #expect(throws: InferenceError.self) {
            try options.validate(against: custom)
        }
    }

    @Test("Catalog is Codable for user-defined catalogs")
    func codable() throws {
        let entry = ModelConfiguration.custom("m1", contextWindowTokens: 1000)
        let catalog = ModelCatalog(providers: ["mine": ["m1": entry]])
        let data = try JSONEncoder().encode(catalog)
        let decoded = try JSONDecoder().decode(ModelCatalog.self, from: data)
        #expect(decoded.model("m1", providerID: "mine")?.contextWindowTokens == 1000)
    }
}
