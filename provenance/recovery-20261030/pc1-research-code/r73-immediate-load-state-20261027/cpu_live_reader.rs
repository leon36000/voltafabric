use std::io::{BufRead,BufReader,Read};
use std::sync::atomic::{AtomicBool,Ordering};
const EXPERTS:[u32;3]=[17,93,241];
static GOOD_LOAD_SENT:AtomicBool=AtomicBool::new(false);
struct Binding {artifact_sha256:String}
struct Artifact {b:Binding}
impl Artifact {fn binding(&self)->&Binding{&self.b}}
struct Child;
impl Child {fn id(&self)->u32{123}}
fn observe_and_repair(bytes:Vec<u8>){
 let mut r73_stderr=BufReader::new(std::io::Cursor::new(bytes));let mut r73_prefix=String::new();
 let position=1;let epoch=61;let bad_expert=93;let child=Child;
 let bad_artifact=Artifact{b:Binding{artifact_sha256:"a".repeat(64)}};
        let r73_probe = r73_read_probe(&mut r73_stderr, &mut r73_prefix);
        let probe = &r73_probe;
        assert_eq!(probe["pid"], child.id(), "native probe PID must match owned raw child");
        assert_eq!(probe["artifact_sha256"], bad_artifact.binding().artifact_sha256);
        assert_eq!(probe["epoch"], epoch);
        assert_eq!(probe["expert"], bad_expert);
        assert_eq!(probe["before"]["staged"], serde_json::json!(EXPERTS[..position]));
        assert_eq!(probe["before"]["published"], serde_json::json!([]));
        assert!(probe["before"]["binding"].is_null());
        assert_eq!(probe["before"]["lifecycle"], serde_json::json!([position,0,0]));
        assert_eq!(probe["before"]["fatal"], false);
        assert_eq!(probe["after"]["staged"], serde_json::json!([]));
        assert_eq!(probe["after"]["published"], serde_json::json!([]));
        assert!(probe["after"]["binding"].is_null());
        assert_eq!(probe["after"]["lifecycle"], serde_json::json!([0,0,0]));
        assert_eq!(probe["after"]["fatal"], false);
        println!("{}", probe);
 GOOD_LOAD_SENT.store(true,Ordering::SeqCst);
}
fn main(){
 let base=serde_json::json!({"gpu_audit":"r73_load_refusal_state","pid":123,"artifact_sha256":"a".repeat(64),"epoch":61,"expert":93,"native_status":-5,"before":{"staged":[17],"published":[],"binding":null,"lifecycle":[1,0,0],"fatal":false},"after":{"staged":[],"published":[],"binding":null,"lifecycle":[0,0,0],"fatal":false}});
 let kind=std::env::args().nth(1).unwrap();let mut v=base;match kind.as_str(){"dirty"=>v["after"]["lifecycle"]=serde_json::json!([1,0,0]),"foreign"=>v["pid"]=serde_json::json!(999),_=>{}}
 let mut data=format!("{}\n",v).into_bytes();if kind=="missing"{data=b"{\"gpu_audit\":\"startup\"}\n".to_vec();}if kind=="partial"{data.pop();}if kind=="badjson"{data=b"garbage\n".to_vec();}if kind=="cap"{data=vec![b'x';65538];}
 let expected=kind=="healthy";let ok=std::panic::catch_unwind(||observe_and_repair(data)).is_ok();assert_eq!(ok,expected);assert_eq!(GOOD_LOAD_SENT.load(Ordering::SeqCst),expected,"repair emitted before immediate probe admission");
 println!("{}",serde_json::json!({"kind":kind,"gate_passed":ok,"repair_sent":GOOD_LOAD_SENT.load(Ordering::SeqCst),"CPU_cursor_only":true}));
}

// Private observer: a valid immediate state is required before the caller repairs.
fn r73_read_probe(reader: &mut impl BufRead, captured: &mut String) -> serde_json::Value {
    loop {
        let mut line = String::new();
        let n = (&mut *reader).take(65537).read_line(&mut line).expect("audit read");
        assert!(n > 0, "missing immediate state probe before repairing LOAD");
        assert!(n <= 65536 && captured.len() + n <= 65536, "audit prefix cap");
        assert!(line.ends_with('\n'), "truncated audit record");
        captured.push_str(&line);
        let value: serde_json::Value = serde_json::from_str(&line).expect("audit JSON record");
        if value["gpu_audit"] == "r73_load_refusal_state" {
            return value;
        }
        assert_eq!(value["gpu_audit"], "startup", "unexpected pre-refusal audit");
    }
}
