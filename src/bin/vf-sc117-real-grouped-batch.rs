//! Bounded local qualification of two-token grouped expert execution.
use std::{collections::BTreeMap, fs, time::{Duration, SystemTime, UNIX_EPOCH}};
use voltafabric::{moe::{Choice, Registry, Replica, Stamp}, native_ggml_cpu::Binding,
    native_ledger::NativeBatchLedger, routing::RoutingProblem, session::{Endpoint, Session},
    sha256, strata_native_layout::NativeLayout};

type E = Box<dyn std::error::Error>;
const INDEX_SHA: &str = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";
const SIZE: u64 = 50_292_326_400;
const WORKER_SHA: &str = "ce6786e7d35182baa64c204ff8de73c190923baa23c560e6c40df497ab079f32";
const ACT_SHA: [&str; 2] = ["c8a0744a87fade2652acfb34c802a49354379f558791bdf63e53f1f124bcc88a", "9cdd44775e6d5afd50f61662745765e1b1a5701c3f31a85c766f82efed8a3403"];
const EXPERTS: [(u32,u32,u32,&str); 4] = [
    (75,2,0,"a312f304226bc4cfb7590ea90e9d91babbba67ef0fe4686a0eae85e26a9ed74d"),
    (255,2,1,"8a17e1a2a5f9a21e11d2bae64fc99a656c810e8ed16d15bed9a09800fccd9cb5"),
    (12,3,0,"658b2b7b9c7b5055667577babea78964084e1a1dad958c546c46d513249f578e"),
    (181,3,1,"98a127792dc0ae3709e7c714225f994913404e880e61e55794c0bf4fcb4d77ea")];

fn run() -> Result<serde_json::Value,E> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 8 { return Err("usage: program PACK INDEX ACT0 ACT1 LEDGER WORKER PLAN_SHA256".into()); }
    if a[7].len()!=64 || !a[7].bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {return Err("invalid plan digest".into());}
    let index=fs::read_to_string(&a[2])?;
    if sha256::hex(index.as_bytes()) != INDEX_SHA {return Err("index digest mismatch".into());}
    if sha256::hex(&fs::read(&a[6])?) != WORKER_SHA {return Err("worker digest mismatch".into());}
    let mut inputs=Vec::new();
    for i in 0..2 {
        let b=fs::read(&a[3+i])?;
        if b.len()!=10240 || sha256::hex(&b)!=ACT_SHA[i] {return Err("activation digest or size mismatch".into());}
        let v:Vec<f32>=b.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect();
        if v.iter().any(|x| !x.is_finite()) {return Err("nonfinite activation".into());}
        inputs.push(v);
    }
    let layout=NativeLayout::parse_sc117_v3(&index,SIZE)?;
    let mut pack=fs::File::open(&a[1])?;
    let mut blobs=Vec::new();
    for (expert,rank,local,digest) in EXPERTS {
        let data=layout.read_verified_expert(&mut pack,2,expert.into(),INDEX_SHA,digest)?;
        let binding=Binding {model_revision:"sc117-local-observation".into(),rank,layer:2,expert,local,epoch:1,index_sha256:INDEX_SHA.into(),blob_sha256:digest.into()};
        blobs.push((binding,data));
    }
    let ledger=NativeBatchLedger::open(std::path::Path::new(&a[5]))?;
    let timeout=Duration::from_secs(60);
    let mut sessions=Vec::new();
    for rank in [2,3] {
        let mut s=Session::connect(&Endpoint {rank,argv:vec![a[6].clone(),"--guarded-local".into(),"--expected-exe-sha256".into(),WORKER_SHA.into()]},timeout)?;
        for (b,data) in &blobs {if b.rank==rank {s.load_native_ggml(b,&index,SIZE,&data.bytes,timeout)?;}}
        s.seal_native_ggml(2,timeout)?;
        sessions.push(s);
    }
    let choices=|ids:[u32;4], weights:[f64;4]| ids.into_iter().zip(weights).map(|(expert,weight)|Choice{expert,weight}).collect();
    let mut p=RoutingProblem {hidden:2560,experts:vec![12,75,181,255],routes:vec![choices([255,12,75,181],[0.125,0.25,0.375,0.25]),choices([181,75,12,255],[0.5,0.125,0.125,0.25])],
        registry:Registry{epoch:1,copies:EXPERTS.iter().map(|(expert,rank,local,_)|Replica{expert:*expert,rank:*rank,local:*local}).collect()},
        stamp:Stamp{run_id:format!("r553-{}-{}",std::process::id(),SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()),model_revision:"sc117-local-observation".into(),plan_id:a[7].clone(),layer:2,batch:0,epoch:1},
        preferred_owners:BTreeMap::new(),resident_order:BTreeMap::from([(2,vec![0,1]),(3,vec![0,1])])};
    let unweighted=Session::exec_native_ggml_unweighted_f32(&mut sessions,&p,&inputs,&ledger,timeout)?;
    p.stamp.batch=1;
    let collected=Session::exec_native_ggml_batch(&mut sessions,&p,&inputs,&ledger,timeout)?;
    if unweighted.len()!=2 || unweighted.iter().any(|t| t.len()!=4 || t.iter().any(|r| r.len()!=2560)) || collected.len()!=2 || collected.iter().any(|r|r.len()!=2560) {return Err("output shape mismatch".into());}
    let mut receipts=Vec::new();
    for s in sessions {
        let raw=s.traffic.native_ggml_raw_execs; let execs=s.traffic.execs;
        let r=s.close(timeout);
        if r.killed || r.error.is_some() || r.exit_code!=Some(0) || r.bye_execs!=Some(2) || raw!=2 || execs!=2 {return Err("worker shutdown or execution count failed".into());}
        receipts.push(serde_json::json!({"rank":r.rank,"pid":r.pid,"raw_execs":raw,"bye_execs":r.bye_execs,"exit_code":r.exit_code}));
    }
    let raw:Vec<Vec<Vec<u32>>>=unweighted.iter().map(|t|t.iter().map(|r|r.iter().map(|v|(*v as f32).to_bits()).collect()).collect()).collect();
    let result:Vec<Vec<u64>>=collected.iter().map(|r|r.iter().map(|v|v.to_bits()).collect()).collect();
    Ok(serde_json::json!({"run_id":p.stamp.run_id,"plan_sha256":a[7],"raw_f32_bits":raw,"collected_f64_bits":result,"receipts":receipts}))
}
fn main(){match run(){Ok(v)=>println!("{v}"),Err(e)=>{eprintln!("qualification failed: {e}");std::process::exit(1)}}}
