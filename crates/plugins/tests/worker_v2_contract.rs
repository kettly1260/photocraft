//! Strict tests for protocol v2 wire framing, bounded JSON parsing, section table validation,
//! and request/response serialization.
#![cfg(not(target_arch = "wasm32"))]

use std::io::Cursor;
use std::path::PathBuf;
use std::time::Duration;
use photocraft_plugins::adapter::{
    Adapter, Context, Descriptor, Error, Image, Operation, OutputKind, Request, API_VERSION,
};
use photocraft_plugins::worker_v2::{
    decode_v2_frame, encode_v2_frame, read_v2_frame_bounded, validate_json_bounds,
    SectionEntry, V2ClientNegotiation, V2HelloRequest, V2ResourceIdentity,
    V2StatusResponse, WorkerAdapterV2, KIND_F32, MAX_HEADER_BYTES, MAX_PAYLOAD_BYTES, MAX_SECTIONS,
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

fn make_test_frame(header_json: &[u8], section_count: u32, section_entries: &[u8]) -> Vec<u8> {
    let mut frame = Vec::new();
    frame.extend_from_slice(RESPONSE_MAGIC);
    frame.extend_from_slice(&(header_json.len() as u32).to_le_bytes());
    frame.extend_from_slice(header_json);
    frame.extend_from_slice(&section_count.to_le_bytes());
    frame.extend_from_slice(section_entries);
    frame
}

#[test]
fn v2_bounded_read_overflow_assertions() {
    // 1. Header length exceeds MAX_HEADER_BYTES (1 MiB)
    let mut bad_header = Vec::new();
    bad_header.extend_from_slice(RESPONSE_MAGIC);
    bad_header.extend_from_slice(&((MAX_HEADER_BYTES + 1) as u32).to_le_bytes());
    bad_header.extend_from_slice(&(0u32).to_le_bytes());
    let err = read_v2_frame_bounded(&mut Cursor::new(bad_header), RESPONSE_MAGIC).unwrap_err();
    assert!(matches!(err, Error::Limit(_)), "Expected Limit error for oversized header, got {:?}", err);

    // 2. Section count exceeds MAX_SECTIONS (64)
    let bad_sections = make_test_frame(b"{}", (MAX_SECTIONS + 1) as u32, &[]);
    let err = read_v2_frame_bounded(&mut Cursor::new(bad_sections), RESPONSE_MAGIC).unwrap_err();
    assert!(matches!(err, Error::Limit(_)), "Expected Limit error for section count > 64, got {:?}", err);

    // 3. Section table unaligned offset (not multiple of 4)
    let mut unaligned_sec_entry = Vec::new();
    unaligned_sec_entry.extend_from_slice(&(2u32).to_le_bytes());
    unaligned_sec_entry.extend_from_slice(&(0u32).to_le_bytes());
    unaligned_sec_entry.extend_from_slice(&(3u32).to_le_bytes()); // offset = 3 (unaligned!)
    unaligned_sec_entry.extend_from_slice(&(4u32).to_le_bytes());
    let unaligned_sec = make_test_frame(b"{}", 1, &unaligned_sec_entry);
    let err = read_v2_frame_bounded(&mut Cursor::new(unaligned_sec), RESPONSE_MAGIC).unwrap_err();
    assert!(matches!(err, Error::Failed(_)), "Expected Failed error for unaligned section offset, got {:?}", err);

    // 4. Section table unaligned length (not multiple of 4)
    let mut unaligned_len_entry = Vec::new();
    unaligned_len_entry.extend_from_slice(&(2u32).to_le_bytes());
    unaligned_len_entry.extend_from_slice(&(0u32).to_le_bytes());
    unaligned_len_entry.extend_from_slice(&(0u32).to_le_bytes());
    unaligned_len_entry.extend_from_slice(&(5u32).to_le_bytes()); // length = 5 (unaligned!)
    let unaligned_len = make_test_frame(b"{}", 1, &unaligned_len_entry);
    let err = read_v2_frame_bounded(&mut Cursor::new(unaligned_len), RESPONSE_MAGIC).unwrap_err();
    assert!(matches!(err, Error::Failed(_)), "Expected Failed error for unaligned section length, got {:?}", err);

    // 5. Section table non-zero reserved field
    let mut non_zero_res_entry = Vec::new();
    non_zero_res_entry.extend_from_slice(&(2u32).to_le_bytes());
    non_zero_res_entry.extend_from_slice(&(1u32).to_le_bytes()); // reserved = 1 (invalid!)
    non_zero_res_entry.extend_from_slice(&(0u32).to_le_bytes());
    non_zero_res_entry.extend_from_slice(&(4u32).to_le_bytes());
    let non_zero_res = make_test_frame(b"{}", 1, &non_zero_res_entry);
    let err = read_v2_frame_bounded(&mut Cursor::new(non_zero_res), RESPONSE_MAGIC).unwrap_err();
    assert!(matches!(err, Error::Failed(_)), "Expected Failed error for non-zero reserved field, got {:?}", err);

    // 6. Payload exceeds MAX_PAYLOAD_BYTES (256 MiB)
    let mut bad_payload_entry = Vec::new();
    bad_payload_entry.extend_from_slice(&(2u32).to_le_bytes());
    bad_payload_entry.extend_from_slice(&(0u32).to_le_bytes());
    bad_payload_entry.extend_from_slice(&(0u32).to_le_bytes());
    bad_payload_entry.extend_from_slice(&((MAX_PAYLOAD_BYTES + 4) as u32).to_le_bytes());
    let bad_payload = make_test_frame(b"{}", 1, &bad_payload_entry);
    let err = read_v2_frame_bounded(&mut Cursor::new(bad_payload), RESPONSE_MAGIC).unwrap_err();
    assert!(matches!(err, Error::Limit(_)), "Expected Limit error for payload > 256 MiB, got {:?}", err);
}

struct TestNoopContext;
impl Context for TestNoopContext {
    fn cancelled(&self) -> bool {
        false
    }
    fn progress(&self, _fraction: f32) {}
}

#[test]
fn rust_csharp_v2_process_interop_regression() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("resolve repo root");
    let csharp_proj = repo_root.join("examples/onnx_qnn_worker/Tests/PhotoCraft.OnnxQnn.Tests.csproj");
    assert!(csharp_proj.exists(), "Required C# test project not found at {:?}", csharp_proj);

    let worker_exe = if let Ok(val) = std::env::var("PHOTOCRAFT_QNN_TEST_BIN") {
        let p = PathBuf::from(val);
        assert!(p.exists(), "Specified PHOTOCRAFT_QNN_TEST_BIN executable does not exist: {:?}", p);
        p
    } else {
        let dotnet_version = std::process::Command::new("dotnet")
            .arg("--version")
            .output()
            .expect(".NET SDK 8.0+ is required on PATH for cross-process interop regression");
        assert!(dotnet_version.status.success(), ".NET SDK 8.0+ must be available on PATH");

        let temp_build = tempfile::Builder::new()
            .prefix("photocraft-qnn-interop-")
            .tempdir()
            .expect("create tempdir");
        let build_out = temp_build.keep().join("bin");

        let build_status = std::process::Command::new("dotnet")
            .arg("build")
            .arg(&csharp_proj)
            .arg("-c")
            .arg("Release")
            .arg("-o")
            .arg(&build_out)
            .status()
            .expect("execute dotnet build");
        assert!(build_status.success(), "dotnet build failed for PhotoCraft.OnnxQnn.Tests.csproj");

        #[cfg(windows)]
        let exe = build_out.join("PhotoCraft.OnnxQnn.Tests.exe");
        #[cfg(not(windows))]
        let exe = build_out.join("PhotoCraft.OnnxQnn.Tests");

        assert!(exe.exists(), "Compiled executable not found at {:?}", exe);
        exe
    };

    let exe_dir = worker_exe.parent().expect("resolve worker exe directory").to_path_buf();

    // 1. Setup synthetic model fixture beside the executable
    let setup_status = std::process::Command::new(&worker_exe)
        .arg("--setup-synthetic")
        .arg(&exe_dir)
        .status()
        .expect("execute --setup-synthetic");
    assert!(setup_status.success(), "--setup-synthetic failed");

    let descriptor = Descriptor {
        api_version: API_VERSION,
        id: "test-qnn-adapter".into(),
        name: "test-qnn-adapter".into(),
        version: "2.0.0".into(),
        operations: vec![Operation {
            id: "select-subject".into(),
            output_kind: OutputKind::Mask,
            color_modes: vec!["rgb".into()],
        }],
        devices: vec![],
        resource_formats: vec![],
        params_schema: serde_json::json!({
            "type": "object",
            "properties": {}
        }),
    };

    let mut adapter = WorkerAdapterV2::new(worker_exe, descriptor, Duration::from_secs(30))
        .expect("WorkerAdapterV2 creation");

    let ctx = TestNoopContext;

    // 2. Hello handshake (reads enabled_features)
    let hello = adapter.hello(&ctx).expect("adapter hello");
    assert_eq!(hello.protocol, 2);
    assert_eq!(hello.msg_type, "hello");
    assert!(hello.negotiation.enabled_features.contains(&"image-mask-v2".to_string()));

    // 3. Needs model check: probe a missing model
    let missing_status = adapter.probe("nonexistent-model", Some("qnn.htp"), &ctx)
        .expect("probe missing model");
    assert_eq!(missing_status.status, "needs_model");
    assert_eq!(missing_status.error.as_ref().map(|e| e.code.as_str()), Some("resource_missing"));

    // 4. Probe and prepare with installed synthetic model
    let reg_path = exe_dir.join("qnn-models.json");
    assert!(reg_path.exists(), "qnn-models.json missing beside worker exe");
    let registry_content = std::fs::read_to_string(&reg_path).expect("read registry");
    let registry: serde_json::Value = serde_json::from_str(&registry_content).expect("parse registry");
    let profile = &registry["profiles"][0];
    let profile_id = profile["id"].as_str().expect("profile id");
    let model_sha256 = profile["sha256"].as_str().unwrap_or("").to_string();

    adapter.set_resource_identity(V2ResourceIdentity {
        id: profile_id.into(),
        version: "1.0.0".into(),
        manifest_digest: model_sha256,
    });

    let probe_status = adapter.probe(profile_id, Some("qnn.htp"), &ctx)
        .expect("probe ready model");
    assert_eq!(probe_status.status, "ready");

    let prepare_status = adapter.prepare(profile_id, Some("qnn.htp"), &ctx)
        .expect("prepare ready model");
    assert_eq!(prepare_status.status, "ready");

    // 5. Run 1x1 image inference
    // Note: Test-only FakeBackend is used in PhotoCraft.OnnxQnn.Tests for cross-language contract verification.
    // This does not constitute real Qualcomm NPU / QNN hardware evidence.
    let req = Request {
        operation: "select-subject".into(),
        params: serde_json::json!({}),
        image: Image {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
            channels: 3,
            color_mode: "rgb".into(),
            source_depth: 32,
            alpha: false,
            icc_profile: Vec::new(),
            data: vec![0.5, 0.5, 0.5],
        },
        resource: Some(profile_id.into()),
        device: Some("qnn.htp".into()),
        allow_fallback: false,
        selection: None,
    };

    let resp = adapter.run(&req, &ctx).expect("adapter run 1x1 image");
    assert_eq!(resp.output.kind, OutputKind::Mask);
    assert_eq!(resp.output.image.width, 1);
    assert_eq!(resp.output.image.height, 1);
    assert_eq!(resp.output.image.channels, 1);
    assert_eq!(resp.output.image.color_mode, "gray");
    assert_eq!(resp.output.image.data.len(), 1);
    assert!((0.0..=1.0).contains(&resp.output.image.data[0]));
    assert_eq!(resp.execution.devices, vec!["qnn.htp".to_string()]);
    assert_eq!(resp.execution.backend, "qnn");
}
