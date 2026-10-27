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

@Suite("SSE line filtering")
struct SSEParserTests {
    @Test("Extracts data payloads with and without spaces")
    func dataPayloads() {
        #expect(SSE.dataPayload(in: "data: hello") == "hello")
        #expect(SSE.dataPayload(in: "data:hello") == "hello")
        #expect(SSE.dataPayload(in: "data: ") == "")
        #expect(SSE.dataPayload(in: "data: [DONE]") == "[DONE]")
    }

    @Test("Non-data lines are ignored")
    func nonDataLines() {
        #expect(SSE.dataPayload(in: "event: message") == nil)
        #expect(SSE.dataPayload(in: "id: 42") == nil)
        #expect(SSE.dataPayload(in: "retry: 1000") == nil)
        #expect(SSE.dataPayload(in: ": comment") == nil)
        #expect(SSE.dataPayload(in: "") == nil)
    }

    @Test("RemoteEngine maps lines by transport")
    func transportMapping() {
        // SSE: only data fields pass through, prefix stripped.
        #expect(RemoteEngine.payload(inLine: "data: {\"a\":1}", transport: .sse) == #"{"a":1}"#)
        #expect(RemoteEngine.payload(inLine: "data:{\"a\":1}", transport: .sse) == #"{"a":1}"#)
        #expect(RemoteEngine.payload(inLine: "event: x", transport: .sse) == nil)
        #expect(RemoteEngine.payload(inLine: ": keep-alive", transport: .sse) == nil)
        #expect(RemoteEngine.payload(inLine: "", transport: .sse) == nil)

        // NDJSON: non-empty lines pass through, whitespace-trimmed.
        #expect(RemoteEngine.payload(inLine: #"{"done":true}"#, transport: .ndjson) == #"{"done":true}"#)
        #expect(RemoteEngine.payload(inLine: "  ", transport: .ndjson) == nil)
        #expect(RemoteEngine.payload(inLine: "", transport: .ndjson) == nil)
    }
}
