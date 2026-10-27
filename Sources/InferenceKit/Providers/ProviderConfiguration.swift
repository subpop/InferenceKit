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

/// The wire protocol spoken by a provider. Determines how requests are
/// encoded and how streaming responses are decoded.
public enum WireFormatID: String, Sendable, Hashable, Codable, CaseIterable {
    /// Anthropic Messages API (`POST /messages`, SSE).
    case anthropicMessages
    /// OpenAI Responses API (`POST /responses`, SSE).
    case openAIResponses
    /// OpenAI Chat Completions API (`POST /chat/completions`, SSE).
    case openAIChatCompletions
    /// Google Gemini `generateContent` API (SSE with `alt=sse`).
    case geminiGenerateContent
    /// Ollama's first-party chat API (`POST /api/chat`, newline-delimited JSON).
    case ollama
}

/// How a provider authenticates requests.
public enum Authentication: Sendable, Hashable, Codable {
    case none
    /// `Authorization: Bearer <token>` (OpenAI-style).
    case bearer(String)
    /// A custom header carrying a static key (Anthropic `x-api-key`,
    /// Google `x-goog-api-key`).
    case apiKey(header: String, value: String)
}

/// Describes an inference endpoint: where to talk, what protocol it speaks,
/// and how to authenticate. Known providers ship as static presets; any
/// OpenAI- or Anthropic-compatible service is a custom `ProviderConfiguration`.
public struct ProviderConfiguration: Sendable, Hashable, Codable, Identifiable {
    public var id: String
    public var displayName: String
    public var baseURL: URL
    public var wireFormatID: WireFormatID
    public var authentication: Authentication
    /// Extra headers applied to every request, after authentication. Useful
    /// for gateways and proxies.
    public var defaultHeaders: [String: String]

    public init(
        id: String,
        displayName: String? = nil,
        baseURL: URL,
        wireFormatID: WireFormatID,
        authentication: Authentication = .none,
        defaultHeaders: [String: String] = [:]
    ) {
        self.id = id
        self.displayName = displayName ?? id
        self.baseURL = baseURL
        self.wireFormatID = wireFormatID
        self.authentication = authentication
        self.defaultHeaders = defaultHeaders
    }
}

// MARK: - Known provider presets

extension ProviderConfiguration {
    /// Anthropic's API.
    public static func anthropic(
        apiKey: String,
        baseURL: URL = URL(staticString: "https://api.anthropic.com/v1")
    ) -> ProviderConfiguration {
        ProviderConfiguration(
            id: "anthropic",
            displayName: "Anthropic",
            baseURL: baseURL,
            wireFormatID: .anthropicMessages,
            authentication: .apiKey(header: "x-api-key", value: apiKey)
        )
    }

    /// OpenAI's API, speaking the Responses API.
    public static func openAI(
        apiKey: String,
        baseURL: URL = URL(staticString: "https://api.openai.com/v1")
    ) -> ProviderConfiguration {
        ProviderConfiguration(
            id: "openai",
            displayName: "OpenAI",
            baseURL: baseURL,
            wireFormatID: .openAIResponses,
            authentication: .bearer(apiKey)
        )
    }

    /// OpenAI's API, speaking Chat Completions. Also the starting point for
    /// most OpenAI-compatible services.
    public static func openAIChatCompletions(
        apiKey: String,
        baseURL: URL = URL(staticString: "https://api.openai.com/v1")
    ) -> ProviderConfiguration {
        ProviderConfiguration(
            id: "openai-chat",
            displayName: "OpenAI (Chat Completions)",
            baseURL: baseURL,
            wireFormatID: .openAIChatCompletions,
            authentication: .bearer(apiKey)
        )
    }

    /// Google's Gemini API.
    public static func gemini(
        apiKey: String,
        baseURL: URL = URL(staticString: "https://generativelanguage.googleapis.com/v1beta")
    ) -> ProviderConfiguration {
        ProviderConfiguration(
            id: "google",
            displayName: "Google Gemini",
            baseURL: baseURL,
            wireFormatID: .geminiGenerateContent,
            authentication: .apiKey(header: "x-goog-api-key", value: apiKey)
        )
    }

    /// A local Ollama server, using its first-party API.
    public static func ollama(
        baseURL: URL = URL(staticString: "http://localhost:11434")
    ) -> ProviderConfiguration {
        ProviderConfiguration(
            id: "ollama",
            displayName: "Ollama",
            baseURL: baseURL,
            wireFormatID: .ollama,
            authentication: .none
        )
    }
}
