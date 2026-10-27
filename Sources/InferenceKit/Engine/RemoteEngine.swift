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

/// An engine that talks to a remote provider over HTTP. Generic over the
/// wire format; transport, authentication, and error handling are shared.
public struct RemoteEngine: InferenceEngine {
    public let provider: ProviderConfiguration
    public let model: ModelConfiguration
    public let urlSession: URLSession
    public let format: any WireFormat

    public init(
        provider: ProviderConfiguration,
        model: ModelConfiguration,
        urlSession: URLSession = .shared
    ) {
        self.provider = provider
        self.model = model
        self.urlSession = urlSession
        self.format = WireFormats.format(for: provider.wireFormatID)
    }

    public func stream(_ request: InferenceRequest) -> AsyncThrowingStream<StreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try request.options.validate(against: model)

                    let chatRequest = Transport.finalized(
                        try format.makeChatRequest(request, provider: provider, model: model),
                        provider: provider
                    )
                    let (bytes, response) = try await urlSession.bytes(for: chatRequest)

                    guard let http = response as? HTTPURLResponse else {
                        throw InferenceError.decodingFailed("Non-HTTP response.")
                    }
                    guard (200..<300).contains(http.statusCode) else {
                        throw InferenceError.http(
                            status: http.statusCode,
                            providerID: provider.id,
                            message: format.decodeErrorMessage(try await Self.errorBody(of: bytes))
                        )
                    }

                    let decoder = format.makeStreamDecoder()
                    for try await line in bytes.lines {
                        guard let payload = Self.payload(inLine: line, transport: format.transport) else {
                            continue
                        }
                        for event in try decoder.feed(payload) {
                            continuation.yield(event)
                        }
                    }
                    for event in try decoder.finish() {
                        continuation.yield(event)
                    }
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

    /// Collects the (bounded) error body from a failed response stream.
    private static func errorBody(of bytes: URLSession.AsyncBytes) async throws -> Data {
        var body = Data()
        for try await byte in bytes {
            body.append(byte)
            if body.count >= 65_536 {
                break
            }
        }
        return body
    }

    /// Maps a transport line to a decoder payload based on framing: SSE data
    /// field payloads for event streams, non-empty lines for NDJSON.
    static func payload(inLine line: String, transport: WireTransport) -> String? {
        switch transport {
        case .sse:
            return SSE.dataPayload(in: line)
        case .ndjson:
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            return trimmed.isEmpty ? nil : trimmed
        }
    }
}
