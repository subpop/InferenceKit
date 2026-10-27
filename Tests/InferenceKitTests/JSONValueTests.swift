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

@Suite("JSONValue")
struct JSONValueTests {
    @Test("Parses scalars, arrays, and objects")
    func parsing() throws {
        let value = try JSONValue(parsing: #"""
        {
            "null": null,
            "bool": true,
            "int": 42,
            "double": 1.5,
            "string": "hello",
            "array": [1, "two", false],
            "object": {"nested": {"deep": true}}
        }
        """#)

        #expect(value["null"]?.isNull == true)
        #expect(value["bool"]?.boolValue == true)
        #expect(value["int"]?.intValue == 42)
        #expect(value["double"]?.doubleValue == 1.5)
        #expect(value["string"]?.stringValue == "hello")
        #expect(value["array"]?[2]?.boolValue == false)
        #expect(value["object"]?["nested"]?["deep"]?.boolValue == true)
    }

    @Test("Round-trips through data")
    func roundTrip() throws {
        let original = JSONValue.object([
            "type": .string("text"),
            "count": .int(7),
            "items": .array([.string("a"), .bool(false)]),
        ])
        let data = try #require(original.jsonData)
        let parsed = try JSONValue(parsing: data)
        #expect(parsed == original)
    }

    @Test("Accessors are nil for mismatched types")
    func accessors() {
        let value = JSONValue.string("text")
        #expect(value.intValue == nil)
        #expect(value.boolValue == nil)
        #expect(value.arrayValue == nil)
        #expect(value["key"] == nil)
        #expect(value[0] == nil)
    }
}
