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

extension URL {
    /// Creates a URL from a string known to be valid at compile time.
    public init(staticString string: String) {
        guard let url = URL(string: string) else {
            fatalError("Invalid static URL: \(string)")
        }
        self = url
    }
}

extension [String] {
    /// Joins prompt parts with a blank line, for wire formats that take a
    /// single system string.
    public var joinedPrompt: String { joined(separator: "\n\n") }
}
