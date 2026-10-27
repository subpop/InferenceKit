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

/// Errors thrown by InferenceKit.
public enum InferenceError: Error, LocalizedError, Sendable, Equatable {
    /// The provider returned a non-success HTTP status.
    case http(status: Int, providerID: String, message: String?)
    /// The provider reported an error inside an otherwise successful stream.
    case server(String)
    /// A requested feature has no equivalent on the chosen wire format or engine.
    case unsupportedFeature(String, context: String)
    /// The request or configuration is invalid for the chosen provider.
    case invalidConfiguration(String)
    /// A response could not be decoded.
    case decodingFailed(String)
    /// A local model is unavailable or failed to load.
    case modelUnavailable(String)

    public var errorDescription: String? {
        switch self {
        case .http(let status, let providerID, let message):
            if let message {
                "\(providerID) returned HTTP \(status): \(message)"
            } else {
                "\(providerID) returned HTTP \(status)."
            }
        case .server(let message):
            "The provider reported an error: \(message)"
        case .unsupportedFeature(let feature, let context):
            "\(feature) is not supported by \(context)."
        case .invalidConfiguration(let detail):
            "Invalid configuration: \(detail)"
        case .decodingFailed(let detail):
            "Failed to decode response: \(detail)"
        case .modelUnavailable(let detail):
            "Model unavailable: \(detail)"
        }
    }
}
