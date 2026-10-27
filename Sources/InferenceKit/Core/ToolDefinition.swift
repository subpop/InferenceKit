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

/// A tool the model may call.
public struct ToolDefinition: Sendable, Hashable, Codable {
    public var name: String
    public var description: String
    /// The parameters JSON Schema as a JSON object.
    public var parametersSchema: JSONValue

    public init(name: String, description: String, parametersSchema: JSONValue) {
        self.name = name
        self.description = description
        self.parametersSchema = parametersSchema
    }

    /// Creates a tool definition from a JSON Schema string.
    public init(name: String, description: String, parametersJSON: String) throws {
        let schema = try JSONValue(parsing: parametersJSON)
        guard schema.objectValue != nil else {
            throw InferenceError.invalidConfiguration(
                "Tool schema for \"\(name)\" must be a JSON object."
            )
        }
        self.init(name: name, description: description, parametersSchema: schema)
    }
}

/// A tool call emitted by the model during streaming.
public struct ToolCall: Sendable, Hashable, Codable {
    public var id: String
    public var name: String
    /// The call arguments as raw JSON text.
    public var arguments: String

    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }

    /// Decodes the arguments into a concrete type.
    public func decodeArguments<T: Decodable>(as type: T.Type) throws -> T {
        guard let data = arguments.data(using: .utf8) else {
            throw InferenceError.decodingFailed("Tool arguments are not valid UTF-8.")
        }
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw InferenceError.decodingFailed(
                "Failed to decode arguments for tool \"\(name)\": \(error.localizedDescription)"
            )
        }
    }

    /// The arguments as a JSON value.
    public var argumentsJSON: JSONValue? {
        try? JSONValue(parsing: arguments)
    }
}
