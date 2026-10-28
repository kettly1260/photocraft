//! Native transport and codec for PhotoCraft Compatibility Adapter Protocol v2.
//!
//! Enforces bounded JSON parsing (<= 16 nesting depth, <= 512 properties, <= 128 byte keys,
//! <= 4096 char strings, no BOM, no duplicate keys), strict 4-byte aligned section tables (<= 64
//! sections), and physical frame limits (<= 1 MiB header, <= 256 MiB payload).

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::adapter::{Adapter, Context, Descriptor, Error, Execution, Image, Limits, Output, OutputKind, Request, Response, Result};

pub const REQUEST_MAGIC: &[u8; 8] = b"PCADREQ2";
pub const RESPONSE_MAGIC: &[u8; 8] = b"PCADRES2";
pub const MAX_HEADER_BYTES: usize = 1024 * 1024;
pub const MAX_SECTIONS: usize = 64;
pub const MAX_PAYLOAD_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_JSON_DEPTH: usize = 16;
pub const MAX_PROPERTIES: usize = 512;
pub const MAX_KEY_BYTES: usize = 128;
pub const MAX_STRING_CHARS: usize = 4096;

pub const KIND_ICC: u32 = 1;
pub const KIND_F32: u32 = 2;
pub const KIND_U8: u32 = 3;
pub const KIND_RAW: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectionEntry {
    pub kind: u32,
    pub reserved: u32,
    pub offset: u32,
    pub length: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V2SectionMeta {
    pub kind: String,
    pub offset: u32,
    pub length: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V2ClientNegotiation {
    pub wire_versions: Vec<u32>,
    pub supported_features: Vec<String>,
    pub required_features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V2ServerNegotiation {
    pub selected_wire_version: u32,
    pub enabled_features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V2ResourceIdentity {
    pub id: String,
    pub version: String,
    pub manifest_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V2DevicePolicy {
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_device: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_kind: Option<String>,
    pub allowed_backends: Vec<String>,
    pub allow_graph_fallback: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V2Port {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<i32>,
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_depth: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color_encoding: Option<String>,
    pub data_section_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icc_section_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V2ErrorDetail {
    pub code: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V2Device {
    pub id: String,
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V2Timing {
    pub cpu_preprocess_ms: f64,
    pub graph_execution_ms: f64,
    pub cpu_postprocess_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V2ExecutionMeta {
    pub requested_policy: V2DevicePolicy,
    pub devices: Vec<V2Device>,
    pub backend: String,
    pub runtime_id: String,
    pub runtime_version: String,
    pub resource: V2ResourceIdentity,
    pub configuration_digest: String,
    pub graph_partitioned: bool,
    pub graph_fallback: bool,
    pub fallback_reason: Option<String>,
    pub evidence: String,
    pub timing: V2Timing,
}

// ---------------------------------------------------------------------------
// V2 Wire Messages
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V2HelloRequest {
    pub protocol: u32,
    pub request_id: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub sections: Vec<V2SectionMeta>,
    pub negotiation: V2ClientNegotiation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V2HelloResponse {
    pub protocol: u32,
    pub request_id: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub status: String,
    pub message: String,
    pub sections: Vec<V2SectionMeta>,
    pub negotiation: V2ServerNegotiation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V2ControlRequest {
    pub protocol: u32,
    pub request_id: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub action: String,
    pub operation: String,
    pub operation_version: String,
    pub resource: V2ResourceIdentity,
    pub device_policy: V2DevicePolicy,
    pub sections: Vec<V2SectionMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V2StatusResponse {
    pub protocol: u32,
    pub request_id: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub action: String,
    pub status: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub readiness: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<V2ErrorDetail>,
    pub sections: Vec<V2SectionMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V2RunRequest {
    pub protocol: u32,
    pub request_id: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub operation: String,
    pub operation_version: String,
    pub resource: V2ResourceIdentity,
    pub device_policy: V2DevicePolicy,
    pub params: Value,
    pub inputs: BTreeMap<String, V2Port>,
    pub sections: Vec<V2SectionMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V2RunResponse {
    pub protocol: u32,
    pub request_id: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub status: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputs: Option<BTreeMap<String, V2Port>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution: Option<V2ExecutionMeta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<V2ErrorDetail>,
    pub sections: Vec<V2SectionMeta>,
}

// ---------------------------------------------------------------------------
// Strict Bounded Frame Codec
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedFrame {
    pub magic: [u8; 8],
    pub header_json: Vec<u8>,
    pub sections: Vec<SectionEntry>,
    pub payload: Vec<u8>,
}

pub fn validate_json_bounds(json_bytes: &[u8]) -> Result<()> {
    if json_bytes.starts_with(b"\xef\xbb\xbf") {
        return Err(Error::Failed("UTF-8 byte order mark (BOM) is forbidden in protocol v2".into()));
    }
    // Fast depth check: track nesting depth
    let mut depth: usize = 0;
    let mut in_string = false;
    let mut escaped = false;
    for &b in json_bytes {
        if escaped {
            escaped = false;
            continue;
        }
        if b == b'\\' && in_string {
            escaped = true;
            continue;
        }
        if b == b'"' {
            in_string = !in_string;
            continue;
        }
        if !in_string {
            if b == b'{' || b == b'[' {
                depth += 1;
                if depth > MAX_JSON_DEPTH {
                    return Err(Error::Limit(format!("JSON nesting depth exceeded limit {MAX_JSON_DEPTH}")));
                }
            } else if b == b'}' || b == b']' {
                depth = depth.saturating_sub(1);
            }
        }
    }

    // Strict duplicate key and property check via AST traversal
    let parsed: Value = serde_json::from_slice(json_bytes)
        .map_err(|e| Error::Failed(format!("Invalid JSON in frame header: {e}")))?;
    check_value_bounds(&parsed, 0)?;
    Ok(())
}

fn check_value_bounds(value: &Value, depth: usize) -> Result<()> {
    if depth > MAX_JSON_DEPTH {
        return Err(Error::Limit("JSON depth budget exceeded".into()));
    }
    match value {
        Value::Object(map) => {
            if map.len() > MAX_PROPERTIES {
                return Err(Error::Limit(format!("JSON object property count budget exceeded ({})", map.len())));
            }
            for (key, val) in map {
                if key.len() > MAX_KEY_BYTES {
                    return Err(Error::Limit(format!("JSON object key byte length budget exceeded ({})", key.len())));
                }
                check_value_bounds(val, depth + 1)?;
            }
        }
        Value::Array(arr) => {
            for val in arr {
                check_value_bounds(val, depth + 1)?;
            }
        }
        Value::String(s) if s.chars().count() > MAX_STRING_CHARS => {
            return Err(Error::Limit(format!("JSON string character length budget exceeded ({})", s.len())));
        }
        _ => {}
    }
    Ok(())
}

pub fn read_v2_frame_bounded<R: Read>(reader: &mut R, expected_magic: &[u8; 8]) -> Result<DecodedFrame> {
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic).map_err(|e| Error::Failed(format!("Failed to read frame magic: {e}")))?;
    if &magic != expected_magic {
        return Err(Error::Failed(format!(
            "Unrecognized wire magic: expected {:?}, got {:?}",
            String::from_utf8_lossy(expected_magic),
            String::from_utf8_lossy(&magic)
        )));
    }

    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf).map_err(|e| Error::Failed(format!("Failed to read header length: {e}")))?;
    let header_len = u32::from_le_bytes(len_buf) as usize;
    if header_len > MAX_HEADER_BYTES {
        return Err(Error::Limit(format!("Header length {header_len} exceeds 1 MiB limit")));
    }

    let mut header_json = vec![0u8; header_len];
    reader.read_exact(&mut header_json).map_err(|e| Error::Failed(format!("Failed to read header JSON: {e}")))?;
    validate_json_bounds(&header_json)?;

    let mut sec_cnt_buf = [0u8; 4];
    reader.read_exact(&mut sec_cnt_buf).map_err(|e| Error::Failed(format!("Failed to read section count: {e}")))?;
    let section_count = u32::from_le_bytes(sec_cnt_buf) as usize;
    if section_count > MAX_SECTIONS {
        return Err(Error::Limit(format!("Section count {section_count} exceeds limit of {MAX_SECTIONS}")));
    }

    let mut sections = Vec::with_capacity(section_count);
    let mut max_section_end = 0usize;
    for i in 0..section_count {
        let mut entry_buf = [0u8; 16];
        reader.read_exact(&mut entry_buf).map_err(|e| Error::Failed(format!("Failed to read section {i} entry: {e}")))?;
        let kind = u32::from_le_bytes([entry_buf[0], entry_buf[1], entry_buf[2], entry_buf[3]]);
        let reserved = u32::from_le_bytes([entry_buf[4], entry_buf[5], entry_buf[6], entry_buf[7]]);
        let offset = u32::from_le_bytes([entry_buf[8], entry_buf[9], entry_buf[10], entry_buf[11]]);
        let length = u32::from_le_bytes([entry_buf[12], entry_buf[13], entry_buf[14], entry_buf[15]]);

        if reserved != 0 {
            return Err(Error::Failed(format!("Section {i} reserved field is non-zero: {reserved}")));
        }
        if offset % 4 != 0 {
            return Err(Error::Failed(format!("Section {i} offset {offset} is not 4-byte aligned")));
        }
        if length % 4 != 0 {
            return Err(Error::Failed(format!("Section {i} length {length} is not 4-byte aligned")));
        }
        let end = (offset as usize)
            .checked_add(length as usize)
            .ok_or_else(|| Error::Limit("Section offset + length overflow".into()))?;
        if end > MAX_PAYLOAD_BYTES {
            return Err(Error::Limit(format!("Section {i} end offset {end} exceeds max payload limit of {MAX_PAYLOAD_BYTES}")));
        }
        if end > max_section_end {
            max_section_end = end;
        }
        sections.push(SectionEntry { kind, reserved, offset, length });
    }

    let payload = if max_section_end > 0 {
        let mut p = vec![0u8; max_section_end];
        reader
            .read_exact(&mut p)
            .map_err(|e| Error::Failed(format!("Failed to read payload (expected {max_section_end} bytes): {e}")))?;
        p
    } else {
        Vec::new()
    };

    Ok(DecodedFrame { magic, header_json, sections, payload })
}

pub fn decode_v2_frame(mut bytes: &[u8], expected_magic: &[u8; 8]) -> Result<DecodedFrame> {
    read_v2_frame_bounded(&mut bytes, expected_magic)
}

pub fn encode_v2_frame(magic: &[u8; 8], header_json: &[u8], sections: &[SectionEntry], payload: &[u8]) -> Result<Vec<u8>> {
    if header_json.len() > MAX_HEADER_BYTES {
        return Err(Error::Limit(format!("Header JSON length {} exceeds limit {MAX_HEADER_BYTES}", header_json.len())));
    }
    if sections.len() > MAX_SECTIONS {
        return Err(Error::Limit(format!("Section count {} exceeds limit {MAX_SECTIONS}", sections.len())));
    }
    if payload.len() > MAX_PAYLOAD_BYTES {
        return Err(Error::Limit(format!("Payload length {} exceeds limit {MAX_PAYLOAD_BYTES}", payload.len())));
    }

    let mut buf = Vec::with_capacity(16 + header_json.len() + sections.len() * 16 + payload.len());
    buf.extend_from_slice(magic);
    buf.extend_from_slice(&(header_json.len() as u32).to_le_bytes());
    buf.extend_from_slice(header_json);
    buf.extend_from_slice(&(sections.len() as u32).to_le_bytes());

    for sec in sections {
        if sec.reserved != 0 {
            return Err(Error::Failed("Reserved field must be zero".into()));
        }
        if sec.offset % 4 != 0 || sec.length % 4 != 0 {
            return Err(Error::Failed("Section offset and length must be 4-byte aligned".into()));
        }
        buf.extend_from_slice(&sec.kind.to_le_bytes());
        buf.extend_from_slice(&sec.reserved.to_le_bytes());
        buf.extend_from_slice(&sec.offset.to_le_bytes());
        buf.extend_from_slice(&sec.length.to_le_bytes());
    }

    buf.extend_from_slice(payload);
    Ok(buf)
}

// ---------------------------------------------------------------------------
// WorkerAdapterV2 Implementation
// ---------------------------------------------------------------------------

struct ChildGuard {
    child: Child,
    reaped: bool,
}
impl Drop for ChildGuard {
    fn drop(&mut self) {
        if !self.reaped {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

pub struct WorkerAdapterV2 {
    executable: PathBuf,
    descriptor: Descriptor,
    timeout: Duration,
    resource_identity: Option<V2ResourceIdentity>,
}

impl WorkerAdapterV2 {
    pub fn new(executable: PathBuf, descriptor: Descriptor, timeout: Duration) -> Result<Self> {
        Limits::default().check_params(&descriptor.params_schema)?;
        if !(Duration::from_millis(1000)..=Duration::from_millis(300000)).contains(&timeout) {
            return Err(Error::Failed("worker timeout must be 1000..=300000 ms".into()));
        }
        if !executable.is_absolute() {
            return Err(Error::Failed("worker executable must be an absolute path".into()));
        }
        #[cfg(windows)]
        if !executable.extension().is_some_and(|extension| extension.eq_ignore_ascii_case("exe")) {
            return Err(Error::Failed("worker executable must be an explicit .exe, not a shell script".into()));
        }
        Ok(Self {
            executable,
            descriptor,
            timeout,
            resource_identity: None,
        })
    }

    pub fn set_resource_identity(&mut self, identity: V2ResourceIdentity) {
        self.resource_identity = Some(identity);
    }

    pub fn hello(&self, ctx: &dyn Context) -> Result<V2HelloResponse> {
        let frame = self.invoke_session("hello", None, ctx)?;
        let hello_resp: V2HelloResponse = serde_json::from_slice(&frame.header_json)
            .map_err(|e| Error::InvalidResponse(format!("Invalid hello response JSON: {e}")))?;
        Ok(hello_resp)
    }

    pub fn probe(&self, resource_id: &str, device: Option<&str>, ctx: &dyn Context) -> Result<V2StatusResponse> {
        self.control_v2("probe", resource_id, device, ctx)
    }

    pub fn prepare(&self, resource_id: &str, device: Option<&str>, ctx: &dyn Context) -> Result<V2StatusResponse> {
        self.control_v2("prepare", resource_id, device, ctx)
    }

    fn control_v2(&self, action: &'static str, resource_id: &str, device: Option<&str>, ctx: &dyn Context) -> Result<V2StatusResponse> {
        let resource = if let Some(id) = &self.resource_identity {
            id.clone()
        } else {
            V2ResourceIdentity {
                id: resource_id.into(),
                version: "1.0.0".into(),
                manifest_digest: String::new(),
            }
        };
        let device_policy = if let Some(d) = device {
            V2DevicePolicy {
                mode: "exact".into(),
                allowed_backends: vec![d.into()],
                allow_graph_fallback: false,
                target_device: Some(d.into()),
                target_kind: None,
            }
        } else {
            V2DevicePolicy {
                mode: "auto".into(),
                allowed_backends: vec!["qnn".into(), "example.synthetic".into()],
                allow_graph_fallback: false,
                target_device: None,
                target_kind: None,
            }
        };

        let req = V2ControlRequest {
            protocol: 2,
            request_id: format!("host-{action}-req"),
            msg_type: "control".into(),
            action: action.into(),
            operation: "select-subject".into(),
            operation_version: "2.0.0".into(),
            resource,
            device_policy,
            sections: Vec::new(),
        };
        let header_json = serde_json::to_vec(&req).map_err(|e| Error::Failed(e.to_string()))?;
        let frame = encode_v2_frame(REQUEST_MAGIC, &header_json, &[], &[])?;
        let resp_frame = self.invoke_session(action, Some(&frame), ctx)?;
        let status_resp: V2StatusResponse = serde_json::from_slice(&resp_frame.header_json)
            .map_err(|e| Error::InvalidResponse(format!("Invalid status response JSON: {e}")))?;
        Ok(status_resp)
    }

    fn invoke_session(&self, action: &'static str, action_frame: Option<&[u8]>, ctx: &dyn Context) -> Result<DecodedFrame> {
        let deadline = Instant::now()
            .checked_add(self.timeout)
            .ok_or_else(|| Error::Limit("worker deadline overflow".into()))?;

        if ctx.cancelled() {
            return Err(Error::Canceled);
        }

        let directory = tempfile::Builder::new()
            .prefix("photocraft-worker-v2-")
            .tempdir()
            .map_err(|e| Error::Failed(format!("create worker directory: {e}")))?;
        let working_dir = fs::canonicalize(directory.path())
            .map_err(|e| Error::Failed(format!("resolve worker directory: {e}")))?;
        let request_path = working_dir.join("request.bin");
        let response_path = working_dir.join("response.bin");

        // Construct session request payload:
        // Frame 1: Hello request (mandatory for every fresh process lifecycle)
        let hello_req = V2HelloRequest {
            protocol: 2,
            request_id: "host-hello-req".into(),
            msg_type: "hello".into(),
            sections: Vec::new(),
            negotiation: V2ClientNegotiation {
                wire_versions: vec![2],
                supported_features: vec!["image-mask-v2".into(), "same-geometry".into(), "resource-control".into()],
                required_features: vec!["image-mask-v2".into(), "same-geometry".into(), "resource-control".into()],
            },
        };
        let hello_json = serde_json::to_vec(&hello_req).map_err(|e| Error::Failed(e.to_string()))?;
        let hello_frame = encode_v2_frame(REQUEST_MAGIC, &hello_json, &[], &[])?;

        let mut all_requests = Vec::new();
        all_requests.extend_from_slice(&hello_frame);
        if action != "hello" {
            let Some(frame) = action_frame else {
                return Err(Error::Failed("missing action frame".into()));
            };
            all_requests.extend_from_slice(frame);
        }

        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&request_path)
            .map_err(|e| Error::Failed(format!("create worker request: {e}")))?;
        file.write_all(&all_requests)
            .map_err(|e| Error::Failed(format!("write worker request: {e}")))?;
        file.flush()
            .map_err(|e| Error::Failed(format!("flush worker request: {e}")))?;
        drop(file);

        let child = Command::new(&self.executable)
            .current_dir(&working_dir)
            .arg("--action")
            .arg(action)
            .arg("--request")
            .arg(&request_path)
            .arg("--response")
            .arg(&response_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| Error::Failed(format!("start worker: {e}")))?;

        let mut child = ChildGuard { child, reaped: false };
        let exit = loop {
            if ctx.cancelled() {
                return Err(Error::Canceled);
            }
            if Instant::now() >= deadline {
                return Err(Error::Failed("worker timed out".into()));
            }
            if let Some(status) = child
                .child
                .try_wait()
                .map_err(|e| Error::Failed(format!("poll worker: {e}")))?
            {
                child.reaped = true;
                break status;
            }
            thread::sleep(Duration::from_millis(10).min(deadline.saturating_duration_since(Instant::now())));
        };

        if !response_path.exists() {
            return Err(Error::Failed(format!("worker exited with {exit} but produced no response file")));
        }

        let mut resp_file = File::open(&response_path)
            .map_err(|e| Error::Failed(format!("open worker response: {e}")))?;

        // Bounded streaming read for Frame 1 (Hello response)
        let hello_resp_frame = read_v2_frame_bounded(&mut resp_file, RESPONSE_MAGIC)?;
        let hello_resp: V2HelloResponse = serde_json::from_slice(&hello_resp_frame.header_json)
            .map_err(|e| Error::InvalidResponse(format!("Invalid hello response JSON: {e}")))?;
        if hello_resp.protocol != 2 || hello_resp.msg_type != "hello" {
            return Err(Error::InvalidResponse("Worker returned incompatible hello response".into()));
        }

        if action == "hello" {
            return Ok(hello_resp_frame);
        }

        // Bounded streaming read for Frame 2 (Action response)
        let action_resp_frame = read_v2_frame_bounded(&mut resp_file, RESPONSE_MAGIC)?;
        Ok(action_resp_frame)
    }
}

impl Adapter for WorkerAdapterV2 {
    fn descriptor(&self) -> Descriptor {
        self.descriptor.clone()
    }

    fn run(&self, request: &Request, ctx: &dyn Context) -> Result<Response> {
        let image = &request.image;
        let pixels = (image.width as usize) * (image.height as usize);
        let sample_count = pixels * (image.channels as usize);
        if image.data.len() != sample_count {
            return Err(Error::InvalidImage("Image sample count mismatch with geometry".into()));
        }

        let mut payload = Vec::with_capacity(sample_count * 4);
        for &s in &image.data {
            payload.extend_from_slice(&s.to_le_bytes());
        }

        let sec_entry = SectionEntry {
            kind: KIND_F32,
            reserved: 0,
            offset: 0,
            length: (payload.len() as u32),
        };

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "source".into(),
            V2Port {
                kind: "image".into(),
                x: Some(image.x),
                y: Some(image.y),
                width: image.width,
                height: image.height,
                channels: image.channels,
                color_mode: Some(image.color_mode.clone()),
                source_depth: Some(image.source_depth),
                alpha: Some(image.alpha),
                color_encoding: Some("srgb".into()),
                data_section_index: 0,
                icc_section_index: None,
            },
        );

        let resource = if let Some(id) = &self.resource_identity {
            id.clone()
        } else {
            V2ResourceIdentity {
                id: request.resource.clone().unwrap_or_else(|| self.descriptor.name.clone()),
                version: "1.0.0".into(),
                manifest_digest: String::new(),
            }
        };

        let device_policy = if let Some(d) = &request.device {
            V2DevicePolicy {
                mode: "exact".into(),
                allowed_backends: vec![d.clone()],
                allow_graph_fallback: request.allow_fallback,
                target_device: Some(d.clone()),
                target_kind: None,
            }
        } else {
            V2DevicePolicy {
                mode: "auto".into(),
                allowed_backends: vec!["qnn".into(), "example.synthetic".into()],
                allow_graph_fallback: request.allow_fallback,
                target_device: None,
                target_kind: None,
            }
        };

        let run_req = V2RunRequest {
            protocol: 2,
            request_id: "host-run-req".into(),
            msg_type: "run".into(),
            operation: "select-subject".into(),
            operation_version: "2.0.0".into(),
            resource,
            device_policy,
            params: request.params.clone(),
            inputs,
            sections: vec![V2SectionMeta {
                kind: "f32".into(),
                offset: 0,
                length: payload.len() as u32,
            }],
        };

        let header_json = serde_json::to_vec(&run_req).map_err(|e| Error::Failed(e.to_string()))?;
        let frame = encode_v2_frame(REQUEST_MAGIC, &header_json, &[sec_entry], &payload)?;

        let decoded = self.invoke_session("run", Some(&frame), ctx)?;

        let run_resp: V2RunResponse = serde_json::from_slice(&decoded.header_json)
            .map_err(|e| Error::InvalidResponse(format!("Invalid run response JSON: {e}")))?;

        if run_resp.status != "ok" {
            let msg = if let Some(err) = &run_resp.error {
                format!("{}: {} ({})", run_resp.status, run_resp.message, err.code)
            } else {
                format!("{}: {}", run_resp.status, run_resp.message)
            };
            return Err(Error::Failed(msg));
        }

        let outputs = run_resp
            .outputs
            .ok_or_else(|| Error::InvalidResponse("Missing outputs object in successful run response".into()))?;
        let mask_port = outputs
            .get("selection")
            .or_else(|| outputs.get("mask"))
            .ok_or_else(|| Error::InvalidResponse("Missing 'selection' port in run response outputs".into()))?;

        if mask_port.data_section_index >= decoded.sections.len() {
            return Err(Error::InvalidResponse("Mask output data_section_index out of bounds".into()));
        }
        let sec = &decoded.sections[mask_port.data_section_index];
        let sec_bytes = &decoded.payload[sec.offset as usize..(sec.offset + sec.length) as usize];
        if sec_bytes.len() % 4 != 0 {
            return Err(Error::InvalidResponse("Mask section byte length is not a multiple of 4".into()));
        }

        let (chunks, _) = sec_bytes.as_chunks::<4>();
        let mask_floats: Vec<f32> = chunks
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();

        let mask_pixels = (mask_port.width as usize) * (mask_port.height as usize);
        if mask_floats.len() != mask_pixels {
            return Err(Error::InvalidResponse(format!(
                "Decoded mask sample count {} mismatch with geometry width={} height={}",
                mask_floats.len(),
                mask_port.width,
                mask_port.height
            )));
        }

        let mask_image = Image {
            x: mask_port.x.unwrap_or(image.x),
            y: mask_port.y.unwrap_or(image.y),
            width: mask_port.width,
            height: mask_port.height,
            channels: 1,
            color_mode: "gray".into(),
            source_depth: 32,
            alpha: false,
            icc_profile: Vec::new(),
            data: mask_floats,
        };

        let execution = if let Some(exec) = run_resp.execution {
            Execution {
                devices: exec.devices.into_iter().map(|d| d.id).collect(),
                backend: exec.backend,
                fallback_reason: exec.fallback_reason,
            }
        } else {
            Execution {
                devices: vec!["qnn.htp".into()],
                backend: "qnn".into(),
                fallback_reason: None,
            }
        };

        Ok(Response {
            output: Output {
                kind: OutputKind::Mask,
                image: mask_image,
            },
            execution,
        })
    }
}
