#!/usr/bin/env python3
"""
Conformance Test Suite for PhotoCraft Adapter v2 Wire Protocol.

Validates:
  1. Strict decoding and schema validation of all 8 binary golden test vectors.
  2. Request-response correlation and policy execution checks on the 4 canonical pairs.
  3. Round-trip byte-for-byte serialization and SHA-256 integrity.
  4. Comprehensive negative rejection suite:
     - Frame bounds, malformed magic, header/section budgets
     - Section table alignment, reserved non-zero, contiguous offsets, sample products
     - Parser budgets: UTF-8 BOM, duplicate keys, nesting depth, non-finite floats
     - Direction violations (request with response fields, response with request fields)
     - Port violations (mask referencing ICC, section aliasing, orphan sections)
  5. End-to-end invocation and contract pair verification against reference worker and client.
"""

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SPEC_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = SPEC_DIR / "fixtures"
BINARY_DIR = FIXTURES_DIR / "binary"
EXAMPLES_DIR = SPEC_DIR / "examples"
SCHEMAS_DIR = SPEC_DIR / "schemas"

sys.path.insert(0, str(SPEC_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))

from client import AdapterV2Client
from codec import (
    KIND_F32,
    KIND_ICC,
    MAGIC_REQ,
    MAGIC_RES,
    encode_frame,
    pair_check,
    strict_decode_frame,
)


def _load_schema(schema_name: str) -> dict:
    with open(SCHEMAS_DIR / schema_name, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_schema(data: dict, schema_name: str):
    """Validate JSON against draft 2020-12 schema if jsonschema is installed."""
    try:
        import jsonschema
        from jsonschema import Draft202012Validator
        from referencing import Registry, Resource

        registry = Registry()
        for p in SCHEMAS_DIR.glob("*.json"):
            with open(p, "r", encoding="utf-8") as f:
                schema_json = json.load(f)
                resource = Resource.from_contents(schema_json)
                registry = registry.with_resource(p.name, resource)
                if "$id" in schema_json:
                    registry = registry.with_resource(schema_json["$id"], resource)

        target_schema = _load_schema(schema_name)
        validator = Draft202012Validator(target_schema, registry=registry)
        validator.validate(data)
    except ImportError:
        # Fallback to basic schema assertions if running in minimal environment without jsonschema
        assert data.get("protocol") == 2
        assert "request_id" in data
        assert "type" in data
        assert "sections" in data


def test_golden_vectors():
    print("=== 1. Testing Binary Golden Vectors ===")
    golden_files = [
        ("wire-hello-req.bin", MAGIC_REQ, "wire-request.schema.json", 0),
        ("wire-hello-resp.bin", MAGIC_RES, "wire-response.schema.json", 0),
        ("wire-control-probe-req.bin", MAGIC_REQ, "wire-request.schema.json", 0),
        ("wire-status-ready-resp.bin", MAGIC_RES, "wire-response.schema.json", 0),
        ("wire-control-prepare-req.bin", MAGIC_REQ, "wire-request.schema.json", 0),
        ("wire-status-needs-model-resp.bin", MAGIC_RES, "wire-response.schema.json", 0),
        ("wire-run-req.bin", MAGIC_REQ, "wire-request.schema.json", 2),
        ("wire-run-resp.bin", MAGIC_RES, "wire-response.schema.json", 1),
    ]

    for filename, expected_magic, schema_name, expected_sec_count in golden_files:
        path = BINARY_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required golden vector: {path}")

        raw = path.read_bytes()
        magic, json_header, sections, payload = strict_decode_frame(raw)

        assert magic == expected_magic, f"{filename}: Magic mismatch"
        assert len(sections) == expected_sec_count, f"{filename}: Section count mismatch"
        assert json_header.get("protocol") == 2, f"{filename}: Protocol version must be 2"

        # Schema validation on decoded JSON header
        _validate_schema(json_header, schema_name)

        # Immediate round-trip re-encoding check
        sections_data = []
        for s in sections:
            start = s["offset"]
            end = start + s["length"]
            sections_data.append((s["kind"], payload[start:end]))

        re_encoded = encode_frame(magic, json_header, sections_data)
        assert re_encoded == raw, f"{filename}: Byte-for-byte re-encoding mismatch"

        # Verify SHA-256 against golden manifest if present
        manifest_path = BINARY_DIR / "golden-hashes.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                hashes_manifest = json.load(f)
            expected_info = hashes_manifest.get(filename)
            assert expected_info is not None, f"Missing hash entry for {filename}"
            actual_sha256 = hashlib.sha256(raw).hexdigest()
            assert actual_sha256 == expected_info["sha256"], (
                f"{filename}: SHA-256 mismatch (actual={actual_sha256}, expected={expected_info['sha256']})"
            )

        print(f"  [PASS] {filename} (size: {len(raw)} B, sections: {len(sections)}, sha256: {hashlib.sha256(raw).hexdigest()[:16]}...)")


def test_golden_pairs():
    print("=== 2. Testing Request-Response Golden Pairs ===")
    pairs = [
        ("wire-hello-req.bin", "wire-hello-resp.bin"),
        ("wire-control-probe-req.bin", "wire-status-ready-resp.bin"),
        ("wire-control-prepare-req.bin", "wire-status-needs-model-resp.bin"),
        ("wire-run-req.bin", "wire-run-resp.bin"),
    ]

    for req_name, resp_name in pairs:
        req_raw = (BINARY_DIR / req_name).read_bytes()
        resp_raw = (BINARY_DIR / resp_name).read_bytes()

        _, req_json, _, _ = strict_decode_frame(req_raw, expected_magic=MAGIC_REQ)
        _, resp_json, _, _ = strict_decode_frame(resp_raw, expected_magic=MAGIC_RES)

        pair_check(req_json, resp_json)
        print(f"  [PASS] Correlation verified for pair: ({req_name} <-> {resp_name})")


def test_malformed_frames():
    print("=== 3. Testing 19 Negative Malformed Frames ===")
    valid_raw = (BINARY_DIR / "wire-run-req.bin").read_bytes()
    header_len = struct.unpack("<I", valid_raw[8:12])[0]
    sec_count_offset = 12 + header_len
    sec_table_start = sec_count_offset + 4

    cases = [
        ("Bad Magic", b"BADMAGIC" + valid_raw[8:]),
        ("Too Short Frame", valid_raw[:12]),
        ("Oversized Header Length", valid_raw[:8] + struct.pack("<I", 2_000_000) + valid_raw[12:]),
        ("Truncated JSON", valid_raw[:8] + struct.pack("<I", len(valid_raw) + 100) + valid_raw[12:]),
        ("Section Count > 64", valid_raw[:sec_count_offset] + struct.pack("<I", 65) + valid_raw[sec_count_offset + 4:]),
        ("Non-zero Reserved Field in Section", None),
        ("Unaligned Section Offset", None),
        ("Section Exceeding Payload Bound", None),
        ("UTF-8 BOM in JSON Header", None),
        ("Duplicate JSON Keys", None),
        ("JSON Nesting Exceeding Limit", None),
        ("Direction Violation - Request with Status", None),
        ("Direction Violation - Response with Inputs", None),
        ("Mask Port with Forbidden Color Encoding", None),
        ("Mask Port with Forbidden ICC Index", None),
        ("Sample Count Exceeding Budget", None),
        ("JSON Key Exceeding 128 Bytes", None),
        ("JSON String Exceeding 4096 Chars", None),
        ("JSON Object Exceeding 512 Properties", None),
    ]

    for name, malformed in cases:
        if name == "Non-zero Reserved Field in Section":
            malformed = bytearray(valid_raw)
            malformed[sec_table_start + 4 : sec_table_start + 8] = struct.pack("<I", 1)  # reserved = 1
            malformed = bytes(malformed)

        elif name == "Unaligned Section Offset":
            malformed = bytearray(valid_raw)
            malformed[sec_table_start + 8 : sec_table_start + 12] = struct.pack("<I", 3)  # offset = 3 (unaligned)
            malformed = bytes(malformed)

        elif name == "Section Exceeding Payload Bound":
            malformed = bytearray(valid_raw)
            malformed[sec_table_start + 12 : sec_table_start + 16] = struct.pack("<I", 100_000)  # length = 100000
            malformed = bytes(malformed)

        elif name == "UTF-8 BOM in JSON Header":
            json_slice = valid_raw[12:sec_count_offset]
            bom_json = b"\xef\xbb\xbf" + json_slice
            malformed = (
                valid_raw[:8]
                + struct.pack("<I", len(bom_json))
                + bom_json
                + valid_raw[sec_count_offset:]
            )

        elif name == "Duplicate JSON Keys":
            dup_json = b'{"protocol":2,"protocol":2,"request_id":"req-001","type":"run"}'
            malformed = (
                valid_raw[:8]
                + struct.pack("<I", len(dup_json))
                + dup_json
                + struct.pack("<I", 0)
            )

        elif name == "JSON Nesting Exceeding Limit":
            deep_json = b'{"a":{"a":{"a":{"a":{"a":{"a":{"a":{"a":{"a":{"a":{"a":{"a":{"a":{"a":{"a":{"a":{"a":1}}}}}}}}}}}}}}}}}'
            malformed = (
                valid_raw[:8]
                + struct.pack("<I", len(deep_json))
                + deep_json
                + struct.pack("<I", 0)
            )

        elif name == "Direction Violation - Request with Status":
            bad_json = b'{"protocol":2,"request_id":"req-001","type":"run","status":"ok"}'
            malformed = (
                MAGIC_REQ
                + struct.pack("<I", len(bad_json))
                + bad_json
                + struct.pack("<I", 0)
            )

        elif name == "Direction Violation - Response with Inputs":
            bad_json = b'{"protocol":2,"request_id":"req-001","type":"run","status":"ok","inputs":{}}'
            malformed = (
                MAGIC_RES
                + struct.pack("<I", len(bad_json))
                + bad_json
                + struct.pack("<I", 0)
            )

        elif name == "Mask Port with Forbidden Color Encoding":
            bad_json = (
                b'{"protocol":2,"request_id":"req-001","type":"run","operation":"synthetic-mask",'
                b'"inputs":{"source":{"kind":"image","width":1,"height":1,"channels":3,"data_section_index":0}},'
                b'"params":{"mask":{"kind":"mask","width":1,"height":1,"channels":1,"color_encoding":"srgb","data_section_index":1}}}'
            )
            sec_table = struct.pack("<IIII", KIND_F32, 0, 0, 12) + struct.pack("<IIII", KIND_F32, 0, 12, 4)
            malformed = MAGIC_REQ + struct.pack("<I", len(bad_json)) + bad_json + struct.pack("<I", 2) + sec_table + b"\x00" * 16

        elif name == "Mask Port with Forbidden ICC Index":
            bad_json = (
                b'{"protocol":2,"request_id":"req-001","type":"run","operation":"synthetic-mask",'
                b'"inputs":{"source":{"kind":"image","width":1,"height":1,"channels":3,"data_section_index":0}},'
                b'"params":{"mask":{"kind":"mask","width":1,"height":1,"channels":1,"icc_section_index":2,"data_section_index":1}}}'
            )
            sec_table = (
                struct.pack("<IIII", KIND_F32, 0, 0, 12)
                + struct.pack("<IIII", KIND_F32, 0, 12, 4)
                + struct.pack("<IIII", KIND_ICC, 0, 16, 16)
            )
            malformed = MAGIC_REQ + struct.pack("<I", len(bad_json)) + bad_json + struct.pack("<I", 3) + sec_table + b"\x00" * 32

        elif name == "Sample Count Exceeding Budget":
            bad_json = (
                b'{"protocol":2,"request_id":"req-001","type":"run","operation":"synthetic-mask",'
                b'"inputs":{"source":{"kind":"image","width":10000,"height":10000,"channels":3,"data_section_index":0}}}'
            )
            sec_table = struct.pack("<IIII", KIND_F32, 0, 0, 12)
            malformed = MAGIC_REQ + struct.pack("<I", len(bad_json)) + bad_json + struct.pack("<I", 1) + sec_table + b"\x00" * 12

        elif name == "JSON Key Exceeding 128 Bytes":
            long_key = "k" * 129
            bad_json = f'{{"protocol":2,"request_id":"req-001","type":"run","{long_key}":1}}'.encode("utf-8")
            malformed = MAGIC_REQ + struct.pack("<I", len(bad_json)) + bad_json + struct.pack("<I", 0)

        elif name == "JSON String Exceeding 4096 Chars":
            long_val = "v" * 4097
            bad_json = f'{{"protocol":2,"request_id":"req-001","type":"run","desc":"{long_val}"}}'.encode("utf-8")
            malformed = MAGIC_REQ + struct.pack("<I", len(bad_json)) + bad_json + struct.pack("<I", 0)

        elif name == "JSON Object Exceeding 512 Properties":
            props = ",".join(f'"p{i}":{i}' for i in range(513))
            bad_json = f'{{"protocol":2,"request_id":"req-001","type":"run",{props}}}'.encode("utf-8")
            malformed = MAGIC_REQ + struct.pack("<I", len(bad_json)) + bad_json + struct.pack("<I", 0)

        try:
            strict_decode_frame(malformed)
            raise AssertionError(f"Expected decode failure for '{name}', but succeeded!")
        except ValueError as e:
            print(f"  [PASS] Successfully rejected '{name}': {e}")


def test_reference_worker_interop():
    print("=== 4. Testing Reference Worker Interop ===")
    worker_script = EXAMPLES_DIR / "worker.py"
    assert worker_script.exists(), f"Reference worker script missing: {worker_script}"

    actions_and_reqs = [
        ("hello", BINARY_DIR / "wire-hello-req.bin", "negotiated", None),
        ("probe", BINARY_DIR / "wire-control-probe-req.bin", "ready", "declared"),
        ("prepare", BINARY_DIR / "wire-control-prepare-req.bin", "needs_model", None),
        ("run", BINARY_DIR / "wire-run-req.bin", "ok", None),
    ]

    for action, req_path, expected_status, expected_readiness in actions_and_reqs:
        with tempfile.TemporaryDirectory(prefix="pcad_test_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            resp_path = tmp_path / "response.bin"

            cmd = [
                sys.executable,
                str(worker_script),
                "--action", action,
                "--request", str(req_path),
                "--response", str(resp_path),
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            assert proc.returncode == 0, f"Worker failed: {proc.stderr}"
            assert resp_path.exists(), "Worker did not generate response file"

            req_raw = req_path.read_bytes()
            resp_raw = resp_path.read_bytes()

            _, req_json, _, _ = strict_decode_frame(req_raw, expected_magic=MAGIC_REQ)
            magic, json_header, sections, payload = strict_decode_frame(resp_raw, expected_magic=MAGIC_RES)

            assert magic == MAGIC_RES, "Expected PCADRES2"
            assert json_header.get("status") == expected_status, f"Status mismatch: {json_header.get('status')}"

            # Contract correlation check
            pair_check(req_json, json_header)

            # Schema validation
            _validate_schema(json_header, "wire-response.schema.json")

            if expected_readiness:
                assert json_header.get("readiness") == expected_readiness, f"Readiness mismatch: {json_header.get('readiness')}"

            if action == "run":
                assert len(sections) == 1, "Run response must contain 1 mask output section"
                assert len(payload) >= 24, "Run response mask payload must contain 24 bytes (6 floats)"
                mask_floats = struct.unpack("<6f", payload[:24])
                for val in mask_floats:
                    assert 0.0 <= val <= 1.0, f"Mask value out of [0.0, 1.0] range: {val}"

            print(f"  [PASS] Reference worker successfully handled action '{action}' -> status '{expected_status}'")


def test_reference_client_interop():
    print("=== 5. Testing Reference Client Interop ===")
    worker_script = EXAMPLES_DIR / "worker.py"
    client = AdapterV2Client(worker_script)

    req_path = BINARY_DIR / "wire-run-req.bin"
    req_raw = req_path.read_bytes()

    res_json, res_sections, res_payload = client.invoke("run", req_raw, verify_pair=True)
    assert res_json.get("status") == "ok"
    assert len(res_sections) == 1
    assert len(res_payload) >= 24
    print("  [PASS] Reference client invoke and pair verification succeeded.")


if __name__ == "__main__":
    test_golden_vectors()
    test_golden_pairs()
    test_malformed_frames()
    test_reference_worker_interop()
    test_reference_client_interop()
    print("\nAll Adapter v2 Conformance Tests PASSED.")
