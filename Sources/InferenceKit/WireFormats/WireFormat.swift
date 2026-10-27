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

/// The framing a wire format uses for streaming responses.
public enum WireTransport: Sendable, Hashable {
    /// Server-sent events; each `data:` line is one JSON payload.
    case sse
    /// Newline-delimited JSON.
    case ndjson
}

/// Executes a `URLRequest` and returns the body with the HTTP response.
/// Abstracted so wire formats can drive paginated model listing against any
/// transport, including test stubs.
public typealias RequestPerformer = @Sendable (URLRequest) async throws -> (Data, HTTPURLResponse)

/// Decodes streaming transport payloads into events. One instance is created
/// per request and consumed by a single task; implementations are not
/// required to be thread-safe.
public protocol WireStreamDecoder {
    /// Feeds one transport payload: an SSE `data:` payload or an NDJSON line.
    /// Returns the events to emit, or throws when the provider reports an
    /// error inside the stream.
    func feed(_ payload: String) throws -> [StreamEvent]

    /// Called when the transport stream ends. Returns any final events, such
    /// as a synthesized completion when the provider's terminal event never
    /// arrived.
    func finish() throws -> [StreamEvent]
}

/// Encodes and decodes one wire protocol. Implementations are stateless;
/// per-request state lives in the stream decoder they produce.
public protocol WireFormat: Sendable {
    var id: WireFormatID { get }
    var transport: WireTransport { get }

    /// Builds the streaming chat request for `request`. The returned request
    /// is finalized by the transport layer (authentication, provider headers).
    func makeChatRequest(
        _ request: InferenceRequest,
        provider: ProviderConfiguration,
        model: ModelConfiguration
    ) throws -> URLRequest

    /// Creates the per-request stream decoder.
    func makeStreamDecoder() -> any WireStreamDecoder

    /// Lists chat-capable models from the provider, driving pagination via
    /// `perform`.
    func listModels(
        provider: ProviderConfiguration,
        perform: RequestPerformer
    ) async throws -> [DiscoveredModel]
}

extension WireFormat {
    /// Decodes a DTO, wrapping failures as ``InferenceError/decodingFailed(_:)``.
    func decodeDTO<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw InferenceError.decodingFailed("\(String(describing: type)): \(error)")
        }
    }

    /// Extracts a human-readable message from a provider error body. All
    /// supported providers shape errors as `{"error": {"message": ...}}` or,
    /// in Ollama's case, `{"error": "..."}`.
    public func decodeErrorMessage(_ data: Data) -> String? {
        guard let json = try? JSONValue(parsing: data) else { return nil }
        if case .string(let message)? = json["error"] {
            return message
        }
        if case .object(let error)? = json["error"], case .string(let message)? = error["message"] {
            return message
        }
        return nil
    }
}

/// Registry of the built-in wire formats.
public enum WireFormats {
    /// The wire format implementation for an identifier.
    public static func format(for id: WireFormatID) -> any WireFormat {
        switch id {
        case .anthropicMessages: AnthropicMessagesFormat()
        case .openAIResponses: OpenAIResponsesFormat()
        case .openAIChatCompletions: OpenAIChatCompletionsFormat()
        case .geminiGenerateContent: GeminiGenerateContentFormat()
        case .ollama: OllamaFormat()
        }
    }
}

/// Shared transport helpers.
enum Transport {
    /// Applies authentication and provider default headers to a request.
    static func finalized(_ request: URLRequest, provider: ProviderConfiguration) -> URLRequest {
        var request = request
        switch provider.authentication {
        case .none:
            break
        case .bearer(let token):
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        case .apiKey(let header, let value):
            request.setValue(value, forHTTPHeaderField: header)
        }
        for (name, value) in provider.defaultHeaders {
            request.setValue(value, forHTTPHeaderField: name)
        }
        return request
    }

    /// Throws ``InferenceError/http(status:providerID:message:)`` for
    /// non-success responses.
    static func check(
        _ response: HTTPURLResponse,
        data: Data,
        provider: ProviderConfiguration,
        format: any WireFormat
    ) throws {
        guard !(200..<300).contains(response.statusCode) else { return }
        throw InferenceError.http(
            status: response.statusCode,
            providerID: provider.id,
            message: format.decodeErrorMessage(data)
        )
    }

    /// A `RequestPerformer` backed by a `URLSession`.
    static func performer(urlSession: URLSession) -> RequestPerformer {
        { request in
            let (data, response) = try await urlSession.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                throw InferenceError.decodingFailed("Non-HTTP response.")
            }
            return (data, http)
        }
    }
}
