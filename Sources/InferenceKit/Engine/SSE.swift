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

/// Minimal server-sent events line filtering. Every provider InferenceKit
/// supports emits one complete JSON object per `data:` line, so each data
/// payload is surfaced independently; multi-line data events are not merged.
enum SSE {
    /// Returns the payload of an SSE `data:` field line, or nil for event,
    /// id, retry, comment, and blank lines.
    static func dataPayload(in line: String) -> String? {
        guard line.hasPrefix("data:") else { return nil }
        var payload = String(line.dropFirst("data:".count))
        if payload.hasPrefix(" ") {
            payload.removeFirst()
        }
        return payload
    }
}
