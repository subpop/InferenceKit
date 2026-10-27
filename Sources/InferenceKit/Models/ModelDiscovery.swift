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

/// A model as reported by a provider's model-list API. Model-list APIs return
/// identifiers and (sometimes) display names, but rarely capabilities;
/// ``ModelCatalog`` merges these with known-model metadata.
public struct DiscoveredModel: Sendable, Hashable, Codable {
    /// The model identifier as the provider reports it (usable on the wire).
    public var id: String
    public var displayName: String?

    public init(id: String, displayName: String? = nil) {
        self.id = id
        self.displayName = displayName
    }
}

/// Merges provider model discovery with the bundled known-model catalog.
public enum ModelDiscovery {
    /// Lists the chat-capable models a provider offers, resolved against the
    /// bundled catalog: discovered identifiers gain catalog capabilities when
    /// they match, and remain usable (with unknown capabilities) when they
    /// don't.
    public static func listModels(
        provider: ProviderConfiguration,
        urlSession: URLSession = .shared
    ) async throws -> [ModelConfiguration] {
        let format = WireFormats.format(for: provider.wireFormatID)
        let discovered = try await format.listModels(
            provider: provider,
            perform: Transport.performer(urlSession: urlSession)
        )
        return discovered.map { ModelCatalog.bundled.resolve($0, providerID: provider.id) }
    }
}
