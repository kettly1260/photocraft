//! Strict tests for protocol v2 wire framing, bounded JSON parsing, section table validation,
//! and request/response serialization.
#![cfg(not(target_arch = "wasm32"))]

use std::collections::BTreeMap;
use photocraft_plugins::worker_v2::{
    decode_v2_frame, encode_v2_frame, validate_json_bounds,
    SectionEntry, V2ClientNegotiation, V2DevicePolicy, V2HelloRequest, V2HelloResponse,
    V2Port, V2ResourceIdentity, V2RunRequest, V2RunResponse, V2SectionMeta, V2ServerNegotiation,
    V2StatusResponse, KIND_F32, MAX_HEADER_BYTES, MAX_JSON_DEPTH, MAX_PAYLOAD_BYTES, MAX_SECTIONS,
    REQUEST_MAGIC, RESPONSE_MAGIC,
};

#[test]
fn v2_wire_magic_and_framing() {
    let header = br#"{"protocol":2,"request_id":"req-1","type":"hello","sections":[],"negotiation":{"wire_versions":[2],"supported_features":[],"required_features":[]}}"#;
    let frame = encode_v2_frame(REQUEST_MAGIC, header, &[], &[]).expect("valid frame encode");
    assert_eq!(&frame[0..8], REQUEST_MAGIC);

    let decoded = decode_v2_frame(&frame, REQUEST_MAGIC).expect("valid frame decode");
    assert_eq!(decoded.magic, *REQUEST_MAGIC);
    assert_eq!(decoded.header_json, header);
    assert!(decoded.sections.is_empty());
    assert!(decoded.payload.is_empty());

    // Wrong expected magic
    assert!(decode_v2_frame(&frame, RESPONSE_MAGIC).is_err());
}

#[test]
fn v2_section_table_alignment_and_bounds() {
    let header = br#"{"protocol":2,"request_id":"req-2","type":"hello","sections":[],"negotiation":{"wire_versions":[2],"supported_features":[],"required_features":[]}}"#;
    let payload = vec![0u8; 64];

    // Valid section
    let valid_sec = SectionEntry {
        kind: KIND_F32,
        reserved: 0,
        offset: 0,
        length: 64,
    };
    let frame = encode_v2_frame(REQUEST_MAGIC, header, &[valid_sec], &payload).expect("encode valid section");
    let decoded = decode_v2_frame(&frame, REQUEST_MAGIC).expect("decode valid section");
    assert_eq!(decoded.sections.len(), 1);
    assert_eq!(decoded.sections[0], valid_sec);

    // Reserved non-zero must fail encode
    let bad_reserved = SectionEntry {
        kind: KIND_F32,
        reserved: 1,
        offset: 0,
        length: 64,
    };
    assert!(encode_v2_frame(REQUEST_MAGIC, header, &[bad_reserved], &payload).is_err());

    // Offset not 4-byte aligned must fail encode
    let unaligned_offset = SectionEntry {
        kind: KIND_F32,
        reserved: 0,
        offset: 3,
        length: 60,
    };
    assert!(encode_v2_frame(REQUEST_MAGIC, header, &[unaligned_offset], &payload).is_err());

    // Section beyond payload bounds
    let out_of_bounds = SectionEntry {
        kind: KIND_F32,
        reserved: 0,
        offset: 32,
        length: 64, // 32 + 64 = 96 > 64
    };
    let bad_frame = encode_v2_frame(REQUEST_MAGIC, header, &[out_of_bounds], &payload).expect("encode succeeds");
    assert!(decode_v2_frame(&bad_frame, REQUEST_MAGIC).is_err());
}

#[test]
fn v2_json_bounds_depth_and_bom() {
    // UTF-8 BOM must be rejected
    let bom_json = b"\xef\xbb\xbf{\"protocol\":2}";
    assert!(validate_json_bounds(bom_json).is_err());

    // Depth <= 16 must pass
    let shallow = br#"{"a":{"b":{"c":1}}}"#;
    assert!(validate_json_bounds(shallow).is_ok());

    // Depth > 16 must fail
    let mut deep = String::new();
    for _ in 0..17 {
        deep.push_str("{\"k\":");
    }
    deep.push('1');
    for _ in 0..17 {
        deep.push('}');
    }
    assert!(validate_json_bounds(deep.as_bytes()).is_err());
}

#[test]
fn v2_hello_and_status_roundtrip() {
    let hello_req = V2HelloRequest {
        protocol: 2,
        request_id: "req-hello".into(),
        msg_type: "hello".into(),
        sections: Vec::new(),
        negotiation: V2ClientNegotiation {
            wire_versions: vec![2],
            supported_features: vec!["image-mask-v2".into()],
            required_features: vec!["image-mask-v2".into()],
        },
    };
    let json_bytes = serde_json::to_vec(&hello_req).expect("serialize hello");
    let parsed: V2HelloRequest = serde_json::from_slice(&json_bytes).expect("deserialize hello");
    assert_eq!(parsed.protocol, 2);
    assert_eq!(parsed.msg_type, "hello");

    let status_resp = V2StatusResponse {
        protocol: 2,
        request_id: "req-status".into(),
        msg_type: "control".into(),
        action: "probe".into(),
        status: "ready".into(),
        message: "NPU ready".into(),
        readiness: Some("ready".into()),
        error: None,
        sections: Vec::new(),
    };
    let status_bytes = serde_json::to_vec(&status_resp).expect("serialize status");
    let parsed_status: V2StatusResponse = serde_json::from_slice(&status_bytes).expect("deserialize status");
    assert_eq!(parsed_status.status, "ready");
    assert_eq!(parsed_status.readiness, Some("ready".into()));
}
