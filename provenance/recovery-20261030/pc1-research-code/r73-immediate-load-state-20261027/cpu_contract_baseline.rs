// CPU collaborators only; no CUDA/native library linked. Actual statements below.
use std::collections::BTreeMap;
use std::io::{self,Write};
use std::sync::atomic::{AtomicUsize,Ordering};
use voltafabric::{Error,Result};
mod resident {
 use super::*;
 pub static LIVE:AtomicUsize=AtomicUsize::new(0);
 pub static CLOSE:AtomicUsize=AtomicUsize::new(0);
 pub static DROP:AtomicUsize=AtomicUsize::new(0);
 #[derive(Debug,Clone,Copy)] pub struct Status(pub i32);
 pub struct Owner {pub live:bool}
 impl Owner {pub fn close(&mut self)->std::result::Result<(),Status>{
  CLOSE.fetch_add(1,Ordering::SeqCst);
  if std::env::var("CLOSE_FAIL").as_deref()==Ok("1"){return Err(Status(-8));}
  assert!(self.live);self.live=false;LIVE.fetch_sub(1,Ordering::SeqCst);Ok(())
 }}
 impl Drop for Owner {fn drop(&mut self){if self.live {DROP.fetch_add(1,Ordering::SeqCst);LIVE.fetch_sub(1,Ordering::SeqCst);}}}
 pub fn lifecycle()->(u64,u64,i32){(LIVE.load(Ordering::SeqCst) as u64,0,0)}
}
static AUDITS:AtomicUsize=AtomicUsize::new(0);
struct Broken;
impl Write for Broken {fn write(&mut self,_:&[u8])->io::Result<usize>{Err(io::ErrorKind::BrokenPipe.into())}fn flush(&mut self)->io::Result<()>{Ok(())}}
struct Partial {done:bool}
impl Write for Partial {fn write(&mut self,_:&[u8])->io::Result<usize>{if self.done {Err(io::ErrorKind::BrokenPipe.into())}else{self.done=true;Ok(1)}}fn flush(&mut self)->io::Result<()>{Ok(())}}
fn audit_line(v:&serde_json::Value)->Result<()> {
 AUDITS.fetch_add(1,Ordering::SeqCst);
 match std::env::var("IO").unwrap().as_str(){"broken"=>write_audit(&mut Broken,v),"partial"=>write_audit(&mut Partial{done:false},v),_=>write_audit(&mut io::stderr().lock(),v)}
}
struct Binding {artifact_sha256:String,epoch:u64}
struct Artifact {binding:Binding}
impl Artifact {fn binding(&self)->&Binding{&self.binding}}
struct Entry {expert:u32}
struct Gpu {owners:BTreeMap<u32,resident::Owner>,binding:Option<serde_json::Value>,fatal:bool,audit:bool}
fn write_audit(writer: &mut impl Write, value: &serde_json::Value) -> Result<()> {
    writeln!(writer, "{value}")
        .map_err(|error| Error::new("GPU_AUDIT_IO", format!("stderr audit write: {error}")))
}
impl Gpu {
    fn status(&mut self, code: resident::Status, operation: &str) -> Error {
        // Only numerical refusal is a documented safe post-start recoverable result.
        // All other unexpected native statuses terminate this GPU worker.
        if code.0 != -9 {
            self.fatal = true;
        }
        Error::new(
            if code.0 == -9 {
                "GPU_NUMERIC"
            } else {
                "GPU_WORKER_FATAL"
            },
            format!(
                "{operation}: native status {}; no replay or CPU fallback",
                code.0
            ),
        )
    }
    fn close_staged(&mut self, staged: &mut BTreeMap<u32, resident::Owner>) -> Result<()> {
        for owner in staged.values_mut() {
            if let Err(code) = owner.close() {
                self.fatal = true;
                // No destructor may repeat cleanup or touch an unrelated context.
                std::mem::forget(std::mem::take(staged));
                return Err(Error::new(
                    "GPU_WORKER_FATAL",
                    format!("LOAD rollback status {}; stop process", code.0),
                ));
            }
        }
        staged.clear();
        Ok(())
    }
fn refuse(&mut self,artifact:&Artifact,entry:&Entry,code:resident::Status,known:bool,staged:&mut BTreeMap<u32,resident::Owner>)->Result<()> {
                    let rollback = self.close_staged(staged);
                    rollback?;
                    return Err(if known {
                        Error::new("GPU_ADMISSION", format!("native LOAD status {}", code.0))
                    } else {
                        self.status(code, "LOAD")
                    });
}
}

fn main(){
 let n:usize=std::env::var("PREFIX").unwrap().parse().unwrap();
 let fail=std::env::var("CLOSE_FAIL").as_deref()==Ok("1");
 let audit=std::env::var("AUDIT").as_deref()==Ok("1");
 let io=std::env::var("IO").unwrap();
 let mut staged=BTreeMap::new();for &key in [17_u32,93,241][..n].iter(){staged.insert(key,resident::Owner{live:true});}
 resident::LIVE.store(n,Ordering::SeqCst);
 let mut gpu=Gpu{owners:BTreeMap::new(),binding:None,fatal:false,audit};
 let artifact=Artifact{binding:Binding{artifact_sha256:"a".repeat(64),epoch:60+n as u64}};
 let result=gpu.refuse(&artifact,&Entry{expert:[17,93,241][n]},resident::Status(-5),true,&mut staged);
 let e=result.unwrap_err();
 let closing_failed=fail&&n>0;
 assert_eq!(e.code,if closing_failed{"GPU_WORKER_FATAL"}else{"GPU_ADMISSION"});
 if !closing_failed{assert_eq!(e.message,"native LOAD status -5");}
 assert_eq!(resident::CLOSE.load(Ordering::SeqCst),if closing_failed{1}else{n},"cleanup repeated or skipped");
 assert_eq!(resident::DROP.load(Ordering::SeqCst),0,"destructor repeated cleanup");
 assert_eq!(AUDITS.load(Ordering::SeqCst),usize::from(audit&&!closing_failed),"missing immediate refusal audit");
 assert_eq!(gpu.fatal,closing_failed||(audit&&io!="healthy"));
 assert!(gpu.owners.is_empty()&&gpu.binding.is_none()&&staged.is_empty());
 if !closing_failed{assert_eq!(resident::LIVE.load(Ordering::SeqCst),0);}
 println!("{}",serde_json::json!({"scope":"CPU collaborators; not CUDA proof","prefix":n,"audit":audit,"io":io,"close_failed":closing_failed,"error":e.code,"fatal":gpu.fatal,"close_calls":resident::CLOSE.load(Ordering::SeqCst),"drop_calls":resident::DROP.load(Ordering::SeqCst),"audit_calls":AUDITS.load(Ordering::SeqCst)}));
}
