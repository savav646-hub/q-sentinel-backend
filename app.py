from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import os, time, secrets, base64
import numpy as np
from redis import Redis

# Crypto clasic
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

# PQC (Kyber/Dilithium) via Open Quantum Safe (liboqs)
try:
    import oqs
    OQS_AVAILABLE = True
except Exception:
    OQS_AVAILABLE = False

SESSION_TTL_SEC = int(os.getenv("SESSION_TTL_SEC","900"))
DEFAULT_N       = int(os.getenv("DEFAULT_N","4096"))
QBER_ABORT      = float(os.getenv("QBER_ABORT","0.11"))
REDIS_URL       = os.getenv("REDIS_URL","redis://redis:6379/0")
CORS_ORIGINS    = [o.strip() for o in os.getenv("CORS_ORIGINS","http://localhost:8000,http://localhost:5500,file://").split(",") if o.strip()]
PQC_KEM_ALG     = os.getenv("PQC_KEM_ALG","ML-KEM-768")     # Kyber-768
PQC_DSA_ALG     = os.getenv("PQC_DSA_ALG","ML-DSA-65")      # Dilithium-3

HKDF_INFO_AES     = b"qs:aesgcm:v1"
HKDF_INFO_PQC_AES = b"mlkem+aead:v1"

app = FastAPI(title="Q-Sentinel — PQC (Kyber/Dilithium) + QKD-sim", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

rds = Redis.from_url(REDIS_URL, decode_responses=False)
def _rkey(*parts:str)->str: return "qs:" + ":".join(parts)

# --------- BB84 (sim) ----------
_rng = lambda seed=None: np.random.default_rng(seed)
def simulate_bb84(N:int, eve:bool=False, p_noise:float=0.0, seed:Optional[int]=None):
    if N<=0: raise ValueError("N>0")
    if not(0.0<=p_noise<=0.5): raise ValueError("p_noise [0,0.5]")
    rng = _rng(seed)
    a_bits  = rng.integers(0,2,size=N,dtype=np.uint8)
    a_base  = rng.integers(0,2,size=N,dtype=np.uint8)
    b_base  = rng.integers(0,2,size=N,dtype=np.uint8)
    if eve:
        e_base = rng.integers(0,2,size=N,dtype=np.uint8)
        e_bits = np.where(e_base==a_base, a_bits, rng.integers(0,2,size=N,dtype=np.uint8))
        b_raw  = np.where(b_base==e_base,  e_bits, rng.integers(0,2,size=N,dtype=np.uint8))
    else:
        b_raw  = np.where(b_base==a_base, a_bits, rng.integers(0,2,size=N,dtype=np.uint8))
    if p_noise>0.0:
        flips = (rng.random(N)<p_noise).astype(np.uint8); b_raw ^= flips
    same = (a_base==b_base)
    b_key = b_raw[same]; a_key = a_bits[same]
    qber = 1.0 if b_key.size==0 else float(np.mean(b_key!=a_key))
    return b_key.astype(np.uint8,copy=False), qber

def bits_to_bytes(bits:np.ndarray)->bytes:
    L=(bits.size//8)*8
    return np.packbits(bits[:L]).tobytes() if L>0 else b""

def text_to_bits(txt:str)->np.ndarray:
    return np.unpackbits(np.frombuffer(txt.encode(),dtype=np.uint8))

def derive_subkey(bits:np.ndarray, salt:bytes, info:bytes, length:int=32)->bytes:
    mat = bits_to_bytes(bits)
    if not mat: raise HTTPException(400,"Cheie insuficientă.")
    hkdf = HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info)
    return hkdf.derive(mat)

# --------- Sesiuni (Redis) ----------
def sess_key(sid:str)->str: return _rkey("sess",sid)
def cur_key(sid:str)->str:  return _rkey("sess",sid,"cursor")

def create_session(bits:np.ndarray, qber:float)->str:
    sid = secrets.token_urlsafe(18); salt=os.urandom(16)
    acc = (qber<=QBER_ABORT and bits.size>=256)
    payload = {b"bits": bits_to_bytes(bits), b"qber": f"{qber:.6f}".encode(), b"salt": salt,
               b"exp": str(int(time.time()+SESSION_TTL_SEC)).encode(), b"acc": b"1" if acc else b"0"}
    rds.hset(sess_key(sid), mapping=payload); rds.expire(sess_key(sid), SESSION_TTL_SEC)
    rds.setex(cur_key(sid), SESSION_TTL_SEC, b"0")
    return sid

def get_session(sid:str)->dict:
    h = rds.hgetall(sess_key(sid))
    if not h: raise HTTPException(404,"Sesiune inexistentă sau expirată")
    if time.time()>int(h[b"exp"]):
        rds.delete(sess_key(sid)); rds.delete(cur_key(sid))
        raise HTTPException(404,"Sesiune expirată")
    return h

def consume_bits(sid:str, nbits:int)->tuple[np.ndarray,bytes]:
    ck = cur_key(sid)
    with rds.pipeline() as p:
        while True:
            try:
                p.watch(ck); cur=int(p.get(ck) or b"0"); start, end = cur, cur+nbits
                p.multi(); p.set(ck, str(end).encode()); p.execute(); break
            except Exception: continue
    packed = rds.hget(sess_key(sid), b"bits")
    all_bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
    if end>all_bits.size:
        rds.set(ck, str(cur).encode())
        raise HTTPException(400, f"Cheie insuficientă: {all_bits.size-cur} biți rămași, {nbits} necesari.")
    salt = rds.hget(sess_key(sid), b"salt")
    return all_bits[start:end], salt

# --------- Schemas ----------
class GenerateReq(BaseModel):
    N:int=Field(default=DEFAULT_N, ge=64); eve:bool=False
    p_noise:float=Field(default=0.0,ge=0.0,le=0.5)
    seed:Optional[int]=None; min_key_len:int=Field(default=256,ge=64)
class GenerateResp(BaseModel):
    session_id:str; key_len:int; qber:float; accepted:bool; preview:str; expires_in_sec:int

class EncReq(BaseModel): session_id:str; plaintext:str; aad:Optional[str]="tx"
class EncResp(BaseModel): bundle:dict; alg:str="AES-256-GCM"; used_bits:int
class DecReq(BaseModel): session_id:str; bundle:dict
class DecResp(BaseModel): plaintext:str

# PQC (Kyber/Dilithium)
def _pqc_required():
    if not OQS_AVAILABLE: raise HTTPException(501,"PQC indisponibil (pachet 'oqs' nu e prezent).")

def kem_state_key(sid:str)->str: return _rkey("pqc","kem",sid)
def pqc_aes_key_key(sid:str)->str: return _rkey("pqc","aes",sid)

class KemStartResp(BaseModel): sid:str; alg:str; public_key_b64:str; expires_in_sec:int=300
class KemFinishReq(BaseModel): sid:str; ciphertext_b64:str
class KemFinishResp(BaseModel): session_id:str; aead:str="AES-256-GCM"

class KemEncapReq(BaseModel): public_key_b64:str; alg:Optional[str]=None
class KemEncapResp(BaseModel): ciphertext_b64:str

class PqcAesEncReq(BaseModel): session_id:str; plaintext:str; aad:Optional[str]="tx"
class PqcAesEncResp(BaseModel): bundle:dict; alg:str="AES-256-GCM"
class PqcAesDecReq(BaseModel): session_id:str; bundle:dict
class PqcAesDecResp(BaseModel): plaintext:str

# Auto
class Capabilities(BaseModel):
    pqc_available: bool
    kem_algs: List[str]
    dsa_algs: List[str]
    qkd_sim: bool = True
    cipher_suites: List[str] = ["PQC_AES_256_GCM","QKD_AES_256_GCM"]

class NegotiateReq(BaseModel):
    want_pqc: bool = True
    want_qkd: bool = True
    prefer: Optional[List[str]] = None
    min_qkd_bits: int = 256
    qkd_N: int = DEFAULT_N
    p_noise: float = 0.0
    eve: bool = False

class NegotiateResp(BaseModel):
    suite: str
    ref: str

# --------- Endpoints ----------
@app.get("/health")
def health(): return {"ok": True, "service":"q-sentinel-backend", "pqc": bool(OQS_AVAILABLE)}

@app.post("/qkd/generate", response_model=GenerateResp)
def qkd_generate(req:GenerateReq):
    b_key, qber = simulate_bb84(req.N, eve=req.eve, p_noise=req.p_noise, seed=req.seed)
    acc = (qber<=QBER_ABORT and b_key.size>=req.min_key_len)
    sid = create_session(b_key, qber)
    preview = ''.join(map(str, b_key[:64].tolist()))
    return GenerateResp(session_id=sid, key_len=int(b_key.size), qber=qber, accepted=acc, preview=preview, expires_in_sec=SESSION_TTL_SEC)

@app.post("/crypto/aesgcm/encrypt", response_model=EncResp)
def aes_encrypt(req:EncReq):
    h = get_session(req.session_id)
    if h[b"acc"]!=b"1": raise HTTPException(400, "Cheie neacceptată (QBER mare sau prea scurtă).")
    need = text_to_bits(req.plaintext).size
    slice_bits, salt = consume_bits(req.session_id, need)
    aes_key = derive_subkey(slice_bits, salt, HKDF_INFO_AES, 32)
    aes = AESGCM(aes_key); nonce=os.urandom(12); aad=(req.aad or "tx").encode()
    ct_tag = aes.encrypt(nonce, req.plaintext.encode(), aad)
    bundle = {"nonce":base64.b64encode(nonce).decode(),"cipher_b64":base64.b64encode(ct_tag).decode(),"aad_b64":base64.b64encode(aad).decode()}
    return EncResp(bundle=bundle, used_bits=int(need))

@app.post("/crypto/aesgcm/decrypt", response_model=DecResp)
def aes_decrypt(req:DecReq):
    nonce=base64.b64decode(req.bundle["nonce"]); aad=base64.b64decode(req.bundle.get("aad_b64","")); ct_tag=base64.b64decode(req.bundle["cipher_b64"])
    ct_bits = np.unpackbits(np.frombuffer(ct_tag, dtype=np.uint8))
    cur = int(rds.get(_rkey("sess",req.session_id,"cursor")) or b"0")
    start = cur - ct_bits.size
    if start<0: raise HTTPException(400,"Offset nesincronizat.")
    packed = rds.hget(_rkey("sess",req.session_id), b"bits"); all_bits = np.unpackbits(np.frombuffer(packed,dtype=np.uint8))
    slice_bits = all_bits[start:start+ct_bits.size]; salt = rds.hget(_rkey("sess",req.session_id), b"salt")
    aes_key = derive_subkey(slice_bits, salt, HKDF_INFO_AES, 32)
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    pt = AESGCM(aes_key).decrypt(nonce, ct_tag, aad)
    return DecResp(plaintext=pt.decode())

# --- PQC: ML-KEM (Kyber) handshake ---
@app.post("/pqc/mlkem/server_start", response_model=KemStartResp)
def pqc_server_start():
    if not OQS_AVAILABLE: raise HTTPException(501,"PQC indisponibil")
    seed=os.urandom(32)
    with oqs.KeyEncapsulation(PQC_KEM_ALG) as kem:
        public_key = kem.generate_keypair_seed(seed)
    sid = secrets.token_urlsafe(16)
    rds.hset(_rkey("pqc","kem",sid), mapping={b"seed":seed, b"alg":PQC_KEM_ALG.encode(), b"exp":str(int(time.time()+300)).encode()})
    rds.expire(_rkey("pqc","kem",sid), 300)
    return KemStartResp(sid=sid, alg=PQC_KEM_ALG, public_key_b64=base64.b64encode(public_key).decode())

@app.post("/pqc/mlkem/server_finish", response_model=KemFinishResp)
def pqc_server_finish(req:KemFinishReq):
    if not OQS_AVAILABLE: raise HTTPException(501,"PQC indisponibil")
    h = rds.hgetall(_rkey("pqc","kem",req.sid))
    if not h: raise HTTPException(404,"Handshake expirat")
    seed=h[b"seed"]; alg=h[b"alg"].decode(); ct=base64.b64decode(req.ciphertext_b64)
    with oqs.KeyEncapsulation(alg) as kem:
        kem.generate_keypair_seed(seed); shared = kem.decap_secret(ct)
    aes_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=HKDF_INFO_PQC_AES).derive(shared)
    psid = "pqc-"+secrets.token_urlsafe(16)
    rds.setex(_rkey("pqc","aes",psid), SESSION_TTL_SEC, aes_key)
    return KemFinishResp(session_id=psid)

@app.post("/pqc/mlkem/encapsulate", response_model=KemEncapResp)
def pqc_client_encapsulate(req:KemEncapReq):
    if not OQS_AVAILABLE: raise HTTPException(501,"PQC indisponibil")
    alg = req.alg or PQC_KEM_ALG
    pk = base64.b64decode(req.public_key_b64)
    with oqs.KeyEncapsulation(alg) as kem:
        ct, _ = kem.encap_secret(pk)
    return KemEncapResp(ciphertext_b64=base64.b64encode(ct).decode())

@app.get("/capabilities", response_model=Capabilities)
def capabilities():
    kem_algs, dsa_algs = [], []
    if OQS_AVAILABLE:
        try:
            kem_algs = getattr(oqs, "get_enabled_kems", lambda: [])()
            dsa_algs = getattr(oqs, "get_enabled_sigs", lambda: [])()
        except Exception:
            pass
    return Capabilities(pqc_available=bool(OQS_AVAILABLE), kem_algs=kem_algs, dsa_algs=dsa_algs)

def _negotiate_impl(req:NegotiateReq)->NegotiateResp:
    prefer = req.prefer or ["PQC_AES_256_GCM","QKD_AES_256_GCM"]
    if req.want_pqc and OQS_AVAILABLE and ("PQC_AES_256_GCM" in prefer):
        with oqs.KeyEncapsulation(PQC_KEM_ALG) as kem:
            pk = kem.generate_keypair()
            ct, _client = kem.encap_secret(pk)
            ss = kem.decap_secret(ct)
        aes_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"mlkem+aead:auto").derive(ss)
        psid = "pqc-"+secrets.token_urlsafe(16)
        rds.setex(_rkey("pqc","aes",psid), SESSION_TTL_SEC, aes_key)
        return NegotiateResp(suite="PQC_AES_256_GCM", ref=psid)
    if req.want_qkd and ("QKD_AES_256_GCM" in prefer):
        b_key, qber = simulate_bb84(req.qkd_N, eve=req.eve, p_noise=req.p_noise)
        if (qber<=QBER_ABORT) and (b_key.size>=req.min_qkd_bits):
            sid = create_session(b_key, qber)
            return NegotiateResp(suite="QKD_AES_256_GCM", ref=sid)
        raise HTTPException(503, f"QKD insuficient (QBER={qber:.4f})")
    raise HTTPException(406,"Nicio suită disponibilă")

@app.post("/negotiate", response_model=NegotiateResp)
def negotiate(req:NegotiateReq): return _negotiate_impl(req)

if __name__=="__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
