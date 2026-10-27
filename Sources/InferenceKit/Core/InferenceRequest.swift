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

/// A complete inference request. The caller supplies the full conversation
/// history; engines are stateless.
public struct InferenceRequest: Sendable, Hashable, Codable {
    /// System prompt parts. Wire formats that require a single string join
    /// these with a blank line.
    public var system: [String]
    /// The full conversation history, oldest first.
    public var messages: [Message]
    /// Tools the model may call.
    public var tools: [ToolDefinition]
    public var options: GenerationOptions

    public init(
        system: [String] = [],
        messages: [Message],
        tools: [ToolDefinition] = [],
        options: GenerationOptions = GenerationOptions()
    ) {
        self.system = system
        self.messages = messages
        self.tools = tools
        self.options = options
    }
}
