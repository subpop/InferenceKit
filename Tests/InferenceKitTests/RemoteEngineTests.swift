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

/// A URLProtocol stub that replays a fixed HTTP response and records requests.
final class StubURLProtocol: URLProtocol {
    nonisolated(unsafe) static var handler: ((URLRequest) throws -> (Int, String))?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let handler = Self.handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.unsupportedURL))
            return
        }
        do {
            let (status, body) = try handler(request)
            let response = HTTPURLResponse(
                url: request.url ?? URL(staticString: "https://example.com"),
                statusCode: status,
                httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": "text/event-stream"]
            )!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data(body.utf8))
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

extension URLRequest {
    /// The request body, reading `httpBodyStream` when URLSession has moved
    /// the body there (as it does before the protocol layer sees it).
    var stubBody: Data {
        if let body = httpBody {
            return body
        }
        guard let stream = httpBodyStream else { return Data() }
        stream.open()
        defer { stream.close() }
        var data = Data()
        let bufferSize = 4096
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { buffer.deallocate() }
        while stream.hasBytesAvailable {
            let read = stream.read(buffer, maxLength: bufferSize)
            if read <= 0 { break }
            data.append(buffer, count: read)
        }
        return data
    }
}

@Suite("RemoteEngine", .serialized)
struct RemoteEngineTests {
    private func makeSession() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubURLProtocol.self]
        return URLSession(configuration: config)
    }

    @Test("Streams a full SSE response through the engine")
    func endToEndStream() async throws {
        StubURLProtocol.handler = { request in
            // Transport-level checks: URL, method, auth, body.
            #expect(request.url?.absoluteString == "https://api.anthropic.com/v1/messages")
            #expect(request.httpMethod == "POST")
            #expect(request.value(forHTTPHeaderField: "x-api-key") == "test-key")
            #expect(request.value(forHTTPHeaderField: "anthropic-version") == "2023-06-01")
            let body = String(data: request.stubBody, encoding: .utf8) ?? ""
            #expect(body.contains("\"stream\":true"))
            #expect(body.contains("iQ=="))

            let stream = """
            event: message_start
            data: {"type": "message_start", "message": {"id": "msg_1", "model": "claude-sonnet-4-5", "usage": {"input_tokens": 10}}}

            event: content_block_start
            data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}

            event: content_block_delta
            data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}

            event: content_block_stop
            data: {"type": "content_block_stop", "index": 0}

            event: message_delta
            data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}

            event: message_stop
            data: {"type": "message_stop"}

            """
            return (200, stream)
        }

        let engine = RemoteEngine(
            provider: .anthropic(apiKey: "test-key"),
            model: .custom("claude-sonnet-4-5", maxOutputTokens: 512),
            urlSession: makeSession()
        )
        let request = InferenceRequest(
            messages: [
                .user("Hi", attachments: [.image(ImageBlock(data: Data([0x89]), mediaType: "image/png"))])
            ]
        )

        var text = ""
        var completion: Completion?
        for try await event in engine.stream(request) {
            switch event {
            case .textDelta(let delta):
                text += delta
            case .completed(let value):
                completion = value
            default:
                break
            }
        }

        #expect(text == "Hi")
        #expect(completion?.finishReason == .stop)
        #expect(completion?.usage?.outputTokens == 5)
        #expect(completion?.assistantMessage?.content == [.text("Hi")])
    }

    @Test("HTTP errors surface with provider messages")
    func httpError() async throws {
        StubURLProtocol.handler = { _ in
            (429, #"{"error": {"type": "rate_limit_error", "message": "Rate limited"}}"#)
        }

        let engine = RemoteEngine(
            provider: .anthropic(apiKey: "test-key"),
            model: .custom("claude-sonnet-4-5", maxOutputTokens: 512),
            urlSession: makeSession()
        )

        await #expect(throws: InferenceError.self) {
            for try await _ in engine.stream(InferenceRequest(messages: [.user("Hi")])) {}
        }
    }

    @Test("Complete collects a full response")
    func complete() async throws {
        StubURLProtocol.handler = { _ in
            let stream = """
            data: {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 3}}}

            data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}

            data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

            data: {"type": "content_block_stop", "index": 0}

            data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}

            data: {"type": "message_stop"}

            """
            return (200, stream)
        }

        let client = InferenceClient(
            provider: .anthropic(apiKey: "test-key"),
            model: .custom("claude-sonnet-4-5", maxOutputTokens: 512),
            urlSession: makeSession()
        )
        let completion = try await client.complete(InferenceRequest(messages: [.user("Hi")]))
        #expect(completion.finishReason == .stop)
        #expect(completion.assistantMessage?.content == [.text("Hello")])
    }

    @Test("Ollama NDJSON streams through the engine")
    func ndjsonEndToEnd() async throws {
        StubURLProtocol.handler = { request in
            #expect(request.url?.absoluteString == "http://localhost:11434/api/chat")
            let stream = """
            {"message": {"role": "assistant", "content": "Hello"}}
            {"message": {"role": "assistant", "content": "!"}}
            {"done": true, "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 2}

            """
            return (200, stream)
        }

        let client = InferenceClient(
            provider: .ollama(),
            model: .custom("llama3.2"),
            urlSession: makeSession()
        )
        let completion = try await client.complete(InferenceRequest(messages: [.user("Hi")]))
        #expect(completion.assistantMessage?.content == [.text("Hello!")])
        #expect(completion.usage == Usage(inputTokens: 5, outputTokens: 2))
    }

    @Test("Discovery merges catalog metadata")
    func discovery() async throws {
        StubURLProtocol.handler = { request in
            #expect(request.url?.absoluteString.contains("models") == true)
            let page = #"{"data": [{"id": "claude-sonnet-4-5-20250929", "display_name": "Ignored"}, {"id": "claude-future-9"}], "has_more": false}"#
            return (200, page)
        }

        let models = try await InferenceClient.discoverModels(
            provider: .anthropic(apiKey: "test-key"),
            urlSession: makeSession()
        )
        #expect(models.count == 2)
        #expect(models[0].id == "claude-sonnet-4-5")
        #expect(models[0].contextWindowTokens == 200_000)
        #expect(models[0].wireID == "claude-sonnet-4-5-20250929")
        #expect(models[1].id == "claude-future-9")
        #expect(models[1].thinking == nil)
    }
}
