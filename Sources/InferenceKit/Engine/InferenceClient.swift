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

/// The main entry point for inference. Wraps an ``InferenceEngine`` — remote
/// (any provider configuration) or local (MLX, Foundation Models) — behind a
/// uniform, stateless API.
///
///     let client = InferenceClient(
///         provider: .anthropic(apiKey: key),
///         model: ModelCatalog.bundled.model("claude-sonnet-4-5", providerID: "anthropic")
///             ?? .custom("claude-sonnet-4-5")
///     )
///
///     for try await event in client.stream(request) { ... }
public struct InferenceClient: Sendable {
    private let engine: any InferenceEngine

    /// The model this client runs.
    public var model: ModelConfiguration { engine.model }

    /// Creates a client over any engine, including local engines.
    public init(engine: any InferenceEngine) {
        self.engine = engine
    }

    /// Creates a client for a remote provider.
    public init(
        provider: ProviderConfiguration,
        model: ModelConfiguration,
        urlSession: URLSession = .shared
    ) {
        self.engine = RemoteEngine(provider: provider, model: model, urlSession: urlSession)
    }

    /// Streams a response. Events end with `.completed`; errors surface by
    /// throwing the stream.
    public func stream(_ request: InferenceRequest) -> AsyncThrowingStream<StreamEvent, Error> {
        engine.stream(request)
    }

    /// Runs a request to completion and returns its terminal payload.
    @discardableResult
    public func complete(_ request: InferenceRequest) async throws -> Completion {
        var completion: Completion?
        for try await event in stream(request) {
            if case .completed(let value) = event {
                completion = value
            }
        }
        guard let completion else {
            throw InferenceError.decodingFailed("Stream finished without a completion event.")
        }
        return completion
    }

    /// Lists the models a provider offers, merging discovered ids with the
    /// bundled catalog's capabilities.
    public static func discoverModels(
        provider: ProviderConfiguration,
        urlSession: URLSession = .shared
    ) async throws -> [ModelConfiguration] {
        try await ModelDiscovery.listModels(provider: provider, urlSession: urlSession)
    }
}
